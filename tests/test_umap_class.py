import numpy as np
import pytest
import scipy.sparse as sp

from umap_algo.umap_class import umap_mapping


def make_model(X_train_: np.ndarray = np.array([]), Y_train_: np.ndarray = np.array([]), **kwargs):
    """
    Function that creates model umap using the default parameters.
    The default parameters are not used to prevent tests
    from passing when changes to their values should cause them to fail.

    Parameters
    -----------
    X_train_ : trained sets
    Y_train_ : trained embeddings

    Default value :
    ---------------
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    KNN_metric: str = "euclidean",
    KNN_method: str = "exact",
    self.a = 1.9
    self.b = 0.79

    """
    model = umap_mapping(**kwargs)
    model.Y_train_ = Y_train_
    model.X_train_ = X_train_
    return model


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestInitializeWithBarycenter:
    def test_weighted_barycenter_is_correct(self):
        """
        GIVEN simple trained embeddings and matrix of weights
            where a new point has 2 neighbors.
        WHEN  call _initialize_with_barycenter,
        THEN  test if the output is the barycenter
        """
        # GIVEN
        Y_train = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
        weights = sp.csr_matrix(
            np.array(
                [
                    [1.0, 3.0, 0.0],
                ]
            )
        )
        model = make_model(Y_train_=Y_train)

        # WHEN
        Y_new = model._initialize_with_barycenter(weights)

        # THEN
        expected = (1.0 * Y_train[0] + 3.0 * Y_train[1]) / (1.0 + 3.0)
        np.testing.assert_allclose(Y_new[0], expected)

    def test_fallback_to_global_mean_when_no_neighbors(self):
        """
        GIVEN simple trained embeddings and matrix of weights
              where a new point has no neighbors,
        WHEN  call _initialize_with_barycenter,
        THEN  test the output is the mean of Y_train.
        """
        # GIVEN
        Y_train = np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
            ]
        )
        weights = sp.csr_matrix(np.zeros((1, 2)))
        model = make_model(Y_train_=Y_train)

        # WHEN
        Y_new = model._initialize_with_barycenter(weights)

        # THEN
        expected = np.array([2.0, 0.0])
        np.testing.assert_allclose(Y_new[0], expected)

    def test_output_shape_matches_m_new_points_and_n_components(self):
        """
        GIVEN a training embedding with n_components dimensions and m new points with valid weights,
        WHEN call _initialize_with_barycenter,
        THEN  the output should have shape = (m, n_components).
        """
        # GIVEN
        n_train, n_components, m = 10, 3, 4
        rng = np.random.default_rng(42)
        Y_train = rng.standard_normal((n_train, n_components))

        W = np.zeros((m, n_train))
        for i in range(m):
            neighbors = rng.choice(n_train, size=2, replace=False)
            W[i, neighbors] = rng.uniform(0.1, 1.0, size=2)
        weights = sp.csr_matrix(W)
        model = make_model(Y_train_=Y_train)

        # WHEN
        Y_new = model._initialize_with_barycenter(weights)

        # THEN
        assert Y_new.shape == (m, n_components)


# ── TestCrossWeights ───────────────────────────────────────────────────────────


class TestCrossWeights:
    def test_weights_are_in_zero_one_range(self):
        """
        GIVEN valid indices and distances between 2 new points and a training set of 5 points,
        WHEN  call _cross_weights,
        THEN  all values in the weight matrix must lie in [0, 1] since they are fuzzy probabilities
            derived from a decreasing exponential.
        """
        # GIVEN
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((5, 3))
        Y_train = rng.standard_normal((5, 2))
        model = make_model(Y_train_=Y_train, X_train_=X_train)

        indices = np.array([[0, 2], [1, 3]])
        distances = np.array([[0.5, 1.2], [0.3, 0.9]])
        rho = distances[:, 0]
        sigma = np.array([1.0, 1.0])

        # WHEN
        W = model._cross_weights(indices, distances, rho, sigma)

        # THEN
        assert W.data.min() >= 0.0
        assert W.data.max() <= 1.0

    def test_output_shape_is_m_new_by_n_train(self):
        """
        GIVEN m=3 new points with k=2 neighbors each among n_train=8 points,
        WHEN  call _cross_weights,
        THEN  the output matrix should have shape=(m, n_train).
        """
        # GIVEN
        rng = np.random.default_rng(1)
        m, k, n_train = 3, 2, 8
        X_train = rng.standard_normal((n_train, 4))
        Y_train = rng.standard_normal((n_train, 2))
        model = make_model(Y_train_=Y_train, X_train_=X_train)

        indices = rng.integers(0, n_train, size=(m, k))
        distances = rng.uniform(0.1, 2.0, size=(m, k))
        rho = distances[:, 0]
        sigma = np.ones(m)

        # WHEN
        W = model._cross_weights(indices, distances, rho, sigma)

        # THEN
        assert W.shape == (m, n_train)

    def test_cross_weights_exact_values(self):
        """
        GIVEN 2 new points with k=2 neighbors each, matrix of distances, rho and sigma
        WHEN  call _cross_weights,
        THEN  check exact values.

        calculated output :
        ---------------
        Formula : w = exp(-max(0, d - rho) / sigma)

        Point 0 → neighbors [1, 3], distances [0.5, 1.5], rho=0.5, sigma=1.0
            w[0,1] = exp(-max(0, 0.5 - 0.5) / 1.0) = exp(0)
            w[0,3] = exp(-max(0, 1.5 - 0.5) / 1.0) = exp(-1)

        Point 1 → neighbors [0, 2], distances [1.0, 2.0], rho=1.0, sigma=2.0
            w[1,0] = exp(-max(0, 1.0 - 1.0) / 2.0) = exp(0)
            w[1,2] = exp(-max(0, 2.0 - 1.0) / 2.0) = exp(-0.5)
        """
        # GIVEN
        model = make_model(X_train_=np.zeros((4, 2)))

        indices = np.array([[1, 3], [0, 2]])
        distances = np.array([[0.5, 1.5], [1.0, 2.0]])
        rho = np.array([0.5, 1.0])
        sigma = np.array([1.0, 2.0])

        # WHEN
        W = model._cross_weights(indices, distances, rho, sigma)

        # THEN
        expected = {
            (0, 1): 1.0,
            (0, 3): np.exp(-1.0),
            (1, 0): 1.0,
            (1, 2): np.exp(-0.5),
        }
        for (row, col), val in expected.items():
            np.testing.assert_allclose(W[row, col], val, rtol=1e-6, err_msg=f"incorrect values in W[{row},{col}]")

        W_dense = W.toarray()
        for r in range(2):
            for c in range(4):
                if (r, c) not in expected:
                    assert W_dense[r, c] == 0.0, f"W[{r},{c}] should be nul"


# ── TestTransform ──────────────────────────────────────────────────────────────


class TestTransform:
    def test_raises_if_fit_transform_not_called(self):
        """
        GIVEN instantiate umap_mapping without fit_transform,
        WHEN  call .transform,
        THEN  a RuntimeError should be raised.
        """
        # GIVEN
        model = make_model()

        # WHEN / THEN
        with pytest.raises(RuntimeError, match="fit_transform"):
            model.transform(np.random.randn(3, 4))

    def test_iris_dataset(self, iris_split):
        """
        GIVEN the iris dataset split into train/test set
        WHEN call .transform
        THEN the ouptut should have the shape of X_test projected onto
            a space of model.n_components dimensions
        """
        # GIVEN
        X_train = iris_split["X_train"]
        X_test = iris_split["X_test"]
        model = make_model()
        model.fit_transform(X_train)

        # WHEN
        embedding_test = model.transform(X_test)

        # THEN
        assert np.shape(embedding_test) == (np.shape(X_test)[0], model.n_components)


# ── TestAttractiveForce ──────────────────────────────────────────────────────────────


class TestAttractiveForce:
    @pytest.mark.parametrize(
        "y_i, y_j, weight_ij, expected_sign_dim",
        [
            (np.array([2.0, 0.0]), np.array([0.0, 0.0]), 1.0, (0, "<")),
            (np.array([-2.0, 0.0]), np.array([0.0, 0.0]), 1.0, (0, ">")),
            (np.array([0.0, 3.0]), np.array([0.0, 0.0]), 1.0, (1, "<")),
            (np.array([0.0, -3.0]), np.array([0.0, 0.0]), 1.0, (1, ">")),
            (np.array([2.0, 0.0]), np.array([0.0, 0.0]), 2.0, (0, "<")),
        ],
    )
    def test_direction_pulls_yi_toward_yj(self, y_i, y_j, weight_ij, expected_sign_dim):
        """
        GIVEN two distinct points y_i and y_j with a positive weight,
        WHEN  attractive_force is called,
        THEN  the gradient should pull y_i toward y_j.
        """
        # GIVEN
        model = make_model()
        dim, sign = expected_sign_dim

        # WHEN
        grad = model.attractive_force(y_i=y_i, y_j=y_j, weight_ij=weight_ij)

        # THEN
        if sign == "<":
            assert grad[dim] < 0.0
        else:
            assert grad[dim] > 0.0

    def test_zero_weight_gives_zero_gradient(self):
        """
        GIVEN two distincts points,
        WHEN  call attractive_force,
        THEN  the gradient should be nul.
        """
        # GIVEN
        model = make_model()
        y_i = np.array([1.0, 2.0])
        y_j = np.array([4.0, 6.0])

        # WHEN
        grad = model.attractive_force(y_i=y_i, y_j=y_j, weight_ij=0.0)

        # THEN
        np.testing.assert_allclose(grad, np.zeros(2))

    @pytest.mark.parametrize(
        "y_i, y_j, w1, w2",
        [
            (np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.5, 1.0),
            (np.array([2.0, 1.0]), np.array([1.0, 1.0]), 0.25, 1.0),
            (np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.1, 0.5),
            (np.array([1.0, 1.0]), np.array([0.0, 0.0]), 1.0, 2.0),
        ],
        ids=[
            "original_case",
            "different_points_w025_w1",
            "vertical_points_w01_w05",
            "diagonal_points_w1_w2",
        ],
    )
    def test_gradient_scales_with_weight(self, y_i, y_j, w1, w2):
        """
        GIVEN two fixed points and two weights w1 < w2,
        WHEN  call attractive_force with each weights,
        THEN  the gradient should be proportional with respect to the weights.
        """
        # GIVEN
        model = make_model()

        # WHEN
        grad1 = model.attractive_force(y_i=y_i, y_j=y_j, weight_ij=w1)
        grad2 = model.attractive_force(y_i=y_i, y_j=y_j, weight_ij=w2)

        # THEN
        np.testing.assert_allclose(grad2, grad1 * (w2 / w1), rtol=1e-6)


# ── TestRepulsiveForce ──────────────────────────────────────────────────────────────


class TestRepulsiveForce:
    @pytest.mark.parametrize(
        "y_i, y_j, expected_direction",
        [
            (np.array([2.0, 0.0]), np.array([0.0, 0.0]), "right"),
            (np.array([-2.0, 0.0]), np.array([0.0, 0.0]), "left"),
            (np.array([0.0, 2.0]), np.array([0.0, 0.0]), "up"),
            (np.array([0.0, -2.0]), np.array([0.0, 0.0]), "down"),
            (np.array([1.0, 1.0]), np.array([0.0, 0.0]), "top-right"),
            (np.array([-1.0, -1.0]), np.array([0.0, 0.0]), "bottom-left"),
        ],
        ids=[
            "right_of_yj",
            "left_of_yj",
            "above_yj",
            "below_yj",
            "top_right_diagonal",
            "bottom_left_diagonal",
        ],
    )
    def test_direction_pushes_yi_away_from_yj(self, y_i, y_j, expected_direction):
        """
        GIVEN y_i positioned relative to y_j with zero weight (maximum repulsion),
        WHEN repulsive_force is invoked,
        THEN the gradient should point away from y_j, pushing y_i in the opposite direction.
        """
        # GIVEN
        model = make_model()

        # WHEN
        grad = model.repulsive_force(y_i=y_i, y_j=y_j, weight_ij=0.0)

        # THEN
        if expected_direction == "right":
            assert grad[0] > 0.0, f"Expected positive x gradient, got {grad[0]}"
        elif expected_direction == "left":
            assert grad[0] < 0.0, f"Expected negative x gradient, got {grad[0]}"
        elif expected_direction == "up":
            assert grad[1] > 0.0, f"Expected positive y gradient, got {grad[1]}"
        elif expected_direction == "down":
            assert grad[1] < 0.0, f"Expected negative y gradient, got {grad[1]}"
        elif expected_direction == "top-right":
            assert grad[0] > 0.0 and grad[1] > 0.0, ValueError(f"Expected positive x and y gradients, got {grad}")
        elif expected_direction == "bottom-left":
            assert grad[0] < 0.0 and grad[1] < 0.0, ValueError(f"Expected negative x and y gradients, got {grad}")

    def test_weight_one_gives_zero_gradient(self):
        """
        GIVEN weight_ij=1.0,
        WHEN  call repulsive_force,
        THEN  the gradient should be nul.
        """
        # GIVEN
        model = make_model()
        y_i = np.array([1.0, 3.0])
        y_j = np.array([4.0, 7.0])

        # WHEN
        grad = model.repulsive_force(y_i, y_j, weight_ij=1.0)

        # THEN
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)

    @pytest.mark.parametrize(
        "y_j, positions",
        [
            # Original case: points moving away along x-axis
            (np.array([0.0, 0.0]), [np.array([0.5, 0.0]), np.array([2.0, 0.0]), np.array([5.0, 0.0])]),
            # Points moving away along y-axis
            (np.array([0.0, 0.0]), [np.array([0.0, 0.5]), np.array([0.0, 2.0]), np.array([0.0, 5.0])]),
            # Points moving away diagonally
            (np.array([0.0, 0.0]), [np.array([0.5, 0.5]), np.array([1.5, 1.5]), np.array([3.0, 3.0])]),
            # Different origin point
            (np.array([1.0, 1.0]), [np.array([1.5, 1.0]), np.array([3.0, 1.0]), np.array([6.0, 1.0])]),
            # Smaller increments
            (np.array([0.0, 0.0]), [np.array([0.1, 0.0]), np.array([0.5, 0.0]), np.array([1.0, 0.0])]),
        ],
        ids=[
            "x_axis_movement",
            "y_axis_movement",
            "diagonal_movement",
            "offset_origin",
            "small_increments",
        ],
    )
    def test_force_decreases_as_points_move_apart(self, y_j, positions):
        """
        GIVEN zero weight and three positions of y_i increasingly far from y_j,
        WHEN  repulsive_force is called for each position,
        THEN  the gradient norm should decrease as distance increases
            (repulsion weakens with distance).
        """
        # GIVEN
        model = make_model()

        # WHEN
        norms = [np.linalg.norm(model.repulsive_force(y_i, y_j, weight_ij=0.0)) for y_i in positions]

        # THEN
        assert norms[0] > norms[1] > norms[2], (
            f"Expected decreasing norms as distance increases, but got {norms[0]:.4f} > {norms[1]:.4f} > {norms[2]:.4f}"
        )


# ── TestFindAbParams ──────────────────────────────────────────────────────────────


class TestFindAbParams:
    @pytest.mark.parametrize("min_dist", [0.05, 0.1, 0.2, 0.5])
    def test_returns_positive_a_and_b(self, min_dist):
        """
        GIVEN a sparse distance matrix with varied values,
        WHEN calling find_ab_params,
        THEN a and b must be strictly positive (physical constraint of the model).
        """

        # GIVEN
        model = make_model(min_dist=min_dist)

        d_vals = np.linspace(0.05, 3.0, 50)
        D = sp.csr_matrix((d_vals, (np.arange(50), np.arange(50))), shape=(50, 50))

        # WHEN
        a, b = model.find_ab_params(D)

        # THEN
        assert a > 0.0, f"a should be positive for min_dist={min_dist}"
        assert b > 0.0, f"b should be positive for min_dist={min_dist}"

    def test_fitted_curve_approximates_target_psi(self):
        """
        GIVEN a distance matrix and its target curve ψ,
        WHEN find_ab_params is used to fit a and b,
        THEN the curve 1 / (1 + a * d^(2b)) should closely match ψ.
        """
        # GIVEN
        model = make_model(min_dist=0.1)
        d_vals = np.linspace(0.05, 3.0, 100)
        psi = np.where(d_vals <= model.min_dist, 1.0, np.exp(-(d_vals - model.min_dist)))
        D = sp.csr_matrix((d_vals, (np.arange(100), np.arange(100))), shape=(100, 100))

        # WHEN
        a, b = model.find_ab_params(D)
        fitted = 1.0 / (1.0 + a * d_vals ** (2 * b))

        # THEN
        rmse = np.sqrt(np.mean((fitted - psi) ** 2))
        assert rmse < 0.05, f"RMSE too high : {rmse:.4f}"


# ── TestSpectralEmbedding ──────────────────────────────────────────────────────────────


class TestSpectralEmbedding:
    def test_output_shape_is_n_samples_by_n_components(self):
        """
        GIVEN a symmetrical weights matrix, valid for 8 points,
        WHEN  call spectral_embedding with n_components=2,
        THEN  the output should have the shape equals to (8, 2).
        """
        # GIVEN
        model = make_model(n_components=2)
        rng = np.random.default_rng(0)
        n = 8
        A = rng.uniform(0.1, 1.0, (n, n))
        W = sp.csr_matrix((A + A.T) / 2)

        # WHEN
        Y = model.spectral_embedding(W)

        # THEN
        assert Y.shape == (n, 2)

    def test_two_disconnected_clusters_are_separated(self):
        """
        GIVEN a graph with two perfectly separated connected components
                (block-diagonal),
        WHEN spectral_embedding is called with n_components=2,
        THEN the points of the two clusters should be separated along at least
                one axis of the latent space.
        """
        # GIVEN
        model = make_model(n_components=2)
        block = np.ones((4, 4)) - np.eye(4)
        W_dense = np.block(
            [
                [block, np.zeros((4, 4))],
                [np.zeros((4, 4)), block],
            ]
        )
        W = sp.csr_matrix(W_dense)

        # WHEN
        Y = model.spectral_embedding(W)

        # THEN
        mean_c0 = Y[:4].mean(axis=0)
        mean_c1 = Y[4:].mean(axis=0)
        assert np.linalg.norm(mean_c0 - mean_c1) > 0.1

    def test_fully_connected_clique_embedding(self):
        """
        GIVEN a fully connected graph (clique) of 4 nodes,
        WHEN spectral_embedding is called with n_components=2,
        THEN all non-trivial eigenvectors should have the same eigenvalue,
        so the embedding should be constant along any axis (up to rotation/scaling).
        """
        # GIVEN
        model = make_model(n_components=2)
        W_dense = np.ones((4, 4)) - np.eye(4)
        W = sp.csr_matrix(W_dense)

        # WHEN
        Y = model.spectral_embedding(W)

        # Then
        # orthonomral columns
        np.testing.assert_almost_equal(Y.T @ Y, np.eye(2), decimal=6)

        # each column should be orthogonal to (1,1,1,1) whose eigen value is 0
        ones = np.ones(4)
        np.testing.assert_almost_equal(Y.T @ ones, np.zeros(2), decimal=6)

        # each column is an eigen vector of the normalized symetrical laplacian of the graph
        #  associated with the eigenvalue 4/3
        deg = np.array(np.sum(W_dense, axis=1)).flatten()  # shape (4,)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L_sym = D_inv_sqrt @ (np.diag(deg) - W_dense) @ D_inv_sqrt

        expected_eigenvalue = 4.0 / 3.0

        for i in range(2):
            v = Y[:, i]
            Lv = L_sym @ v
            # check: ||Lv||² == λ² ||v||² and Lv ⟂ null space
            np.testing.assert_almost_equal(
                np.linalg.norm(Lv),
                expected_eigenvalue * np.linalg.norm(v),
                decimal=6,
                err_msg=f"The column {i} is not in the eigenspace {expected_eigenvalue}",
            )
            np.testing.assert_almost_equal(
                ones @ Lv, 0.0, decimal=6, err_msg=f"L_sym @ v[:,{i}] must not have any component along the null vector"
            )
