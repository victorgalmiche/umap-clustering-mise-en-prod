import numpy as np
import scipy.sparse as sp
import pytest
from unittest.mock import patch
from umap_algo.umap_class import umap_mapping


def make_model(X_train_: np.ndarray = np.array([]), Y_train_: np.ndarray = np.array([])):
    """Crée un mock de umap_mapping avec Y_train_ défini."""
    model = umap_mapping()
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
        Y_train = np.array([
            [0.0, 0.0],   
            [2.0, 2.0],   
            [10.0, 10.0]  
        ])
        weights = sp.csr_matrix(np.array([
            [1.0, 3.0, 0.0],
        ]))
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
        Y_train = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
        ])
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
            np.testing.assert_allclose(
                W[row, col], val, rtol=1e-6, err_msg=f"incorrect values in W[{row},{col}]"
            )

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
        model = umap_mapping()
 
        # WHEN / THEN
        with pytest.raises(RuntimeError, match="fit_transform"):
            model.transform(np.random.randn(3, 4))
 
    def test_output_shape_matches_m_new_points_and_n_components(self):
        """
        GIVEN un modèle entraîné (X_train_, Y_train_ définis) et m=3 nouveaux
              points dans le même espace de features,
        WHEN  on appelle transform en mockant exact_knn_all_points pour éviter
              toute dépendance extérieure,
        THEN  le tableau retourné doit avoir la forme (m, n_components).
        """
        # GIVEN
        rng = np.random.default_rng(42)
        n_train, n_features, n_components = 20, 4, 2
        m = 3
        k = 5  # n_neighbors par défaut du modèle
 
        X_train = rng.standard_normal((n_train, n_features))
        Y_train = rng.standard_normal((n_train, n_components))
        X_new = rng.standard_normal((m, n_features))
 
        model = umap_mapping(n_neighbors=k, n_components=n_components)
        model.X_train_ = X_train
        model.Y_train_ = Y_train
 
        # Distances et indices fictifs mais cohérents (m × k)
        fake_indices = rng.integers(0, n_train, size=(m, k))
        fake_distances = rng.uniform(0.1, 2.0, size=(m, k))
        # On trie par distance croissante pour que rho = distances[:, 0] soit correct
        order = np.argsort(fake_distances, axis=1)
        fake_distances = np.take_along_axis(fake_distances, order, axis=1)
        fake_indices = np.take_along_axis(fake_indices,   order, axis=1)
 
        # WHEN — on mocke exact_knn_all_points pour isoler transform
        with patch(
            "umap_algo.umap_class.exact_knn_all_points",
            return_value=(fake_indices, fake_distances),
        ):
            Y_new = model.transform(X_new, n_epochs=5)
 
        # THEN
        assert Y_new.shape == (m, n_components)
 
    def test_new_points_stay_near_neighbors_after_optimization(self):
        """
        GIVEN un modèle entraîné dont les points d'entraînement sont regroupés
              en deux clusters bien séparés dans Y_train,
              et un nouveau point clairement rattaché au cluster 1,
        WHEN  on appelle transform avec peu d'époques,
        THEN  le nouveau point doit rester proche du centre du cluster 1
              et loin du cluster 2.
        """
        # GIVEN
        rng = np.random.default_rng(7)
        n_per_cluster = 10
        k = 4
 
        # Cluster 0 autour de (-5, 0), cluster 1 autour de (5, 0) dans Y_train
        Y_cluster0 = rng.standard_normal((n_per_cluster, 2)) + np.array([-5.0, 0.0])
        Y_cluster1 = rng.standard_normal((n_per_cluster, 2)) + np.array([5.0, 0.0])
        Y_train = np.vstack([Y_cluster0, Y_cluster1])
 
        X_train = rng.standard_normal((2 * n_per_cluster, 4))
        X_new = rng.standard_normal((1, 4))
 
        model = umap_mapping(n_neighbors=k, n_components=2)
        model.X_train_ = X_train
        model.Y_train_ = Y_train
 
        # Nouveau point connecté uniquement aux voisins du cluster 1 (indices 10-13)
        fake_indices = np.array([[10, 11, 12, 13]])
        fake_distances = np.array([[0.2, 0.4, 0.6, 0.8]])
 
        with patch(
            "umap_algo.umap_class.exact_knn_all_points",
            return_value=(fake_indices, fake_distances),
        ):
            Y_new = model.transform(X_new, n_epochs=30)
 
        # THEN — le point transformé doit être plus proche du centre du cluster 1
        center_cluster0 = Y_cluster0.mean(axis=0)
        center_cluster1 = Y_cluster1.mean(axis=0)
        dist_to_c0 = np.linalg.norm(Y_new[0] - center_cluster0)
        dist_to_c1 = np.linalg.norm(Y_new[0] - center_cluster1)
        assert dist_to_c1 < dist_to_c0, (
            f"Le point devrait être proche du cluster 1 "
            f"(dist_c1={dist_to_c1:.2f} < dist_c0={dist_to_c0:.2f})"
        )

# ── TestAttractiveForce ──────────────────────────────────────────────────────────────


class TestAttractiveForce:

    def test_direction_pulls_yi_toward_yj(self):
        """
        GIVEN deux points y_i et y_j distincts avec un poids positif,
        WHEN  on appelle attractive_force,
        THEN  le gradient doit pointer de y_i vers y_j (signe opposé à y_i - y_j).
        """
        # GIVEN
        model = umap_mapping()
        y_i = np.array([2.0, 0.0])
        y_j = np.array([0.0, 0.0])

        # WHEN
        grad = model.attractive_force(y_i, y_j, weight_ij=1.0)

        # THEN — gradient négatif sur x car y_i est à droite de y_j
        assert grad[0] < 0.0

    def test_zero_weight_gives_zero_gradient(self):
        """
        GIVEN deux points distincts mais un poids nul,
        WHEN  on appelle attractive_force,
        THEN  le gradient doit être le vecteur zéro.
        """
        # GIVEN
        model = umap_mapping()
        y_i = np.array([1.0, 2.0])
        y_j = np.array([4.0, 6.0])

        # WHEN
        grad = model.attractive_force(y_i, y_j, weight_ij=0.0)

        # THEN
        np.testing.assert_allclose(grad, np.zeros(2))

    def test_gradient_scales_with_weight(self):
        """
        GIVEN deux points fixes et deux poids w1 < w2,
        WHEN  on appelle attractive_force avec chacun des poids,
        THEN  la norme du gradient doit être proportionnelle au poids.
        """
        # GIVEN
        model = umap_mapping()
        y_i = np.array([1.0, 0.0])
        y_j = np.array([0.0, 0.0])
        w1, w2 = 0.5, 1.0

        # WHEN
        grad1 = model.attractive_force(y_i, y_j, weight_ij=w1)
        grad2 = model.attractive_force(y_i, y_j, weight_ij=w2)

        # THEN
        np.testing.assert_allclose(grad2, grad1 * (w2 / w1), rtol=1e-6)


class TestRepulsiveForce:

    def test_direction_pushes_yi_away_from_yj(self):
        """
        GIVEN y_i à droite de y_j et un poids nul (répulsion maximale),
        WHEN  on appelle repulsive_force,
        THEN  le gradient doit pointer dans le sens opposé à y_j,
              c'est-à-dire vers la droite (composante x positive).
        """
        # GIVEN
        model = umap_mapping()
        y_i = np.array([2.0, 0.0])
        y_j = np.array([0.0, 0.0])

        # WHEN
        grad = model.repulsive_force(y_i, y_j, weight_ij=0.0)

        # THEN
        assert grad[0] > 0.0

    def test_weight_one_gives_zero_gradient(self):
        """
        GIVEN un poids weight_ij=1.0 (les deux points sont voisins certains),
        WHEN  on appelle repulsive_force,
        THEN  le gradient doit être nul car (1 - weight_ij) = 0.
        """
        # GIVEN
        model = umap_mapping()
        y_i = np.array([1.0, 3.0])
        y_j = np.array([4.0, 7.0])

        # WHEN
        grad = model.repulsive_force(y_i, y_j, weight_ij=1.0)

        # THEN
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)

    def test_force_decreases_as_points_move_apart(self):
        """
        GIVEN un poids nul et trois positions de y_i de plus en plus éloignées de y_j,
        WHEN  on appelle repulsive_force pour chacune,
        THEN  la norme du gradient doit décroître à mesure que la distance augmente
              (la répulsion s'atténue avec l'éloignement).
        """
        # GIVEN
        model = umap_mapping()
        y_j = np.array([0.0, 0.0])
        positions = [np.array([0.5, 0.0]), np.array([2.0, 0.0]), np.array([5.0, 0.0])]

        # WHEN
        norms = [np.linalg.norm(model.repulsive_force(y_i, y_j, weight_ij=0.0))
                 for y_i in positions]

        # THEN
        assert norms[0] > norms[1] > norms[2]


class TestFindAbParams:

    def test_returns_positive_a_and_b(self):
        """
        GIVEN une matrice de distances sparse avec des valeurs variées,
        WHEN  on appelle find_ab_params,
        THEN  a et b doivent être strictement positifs (contrainte physique du modèle).
        """
        # GIVEN
        model = umap_mapping(min_dist=0.1)
        d_vals = np.linspace(0.05, 3.0, 50)
        D = sp.csr_matrix(
            (d_vals, (np.arange(50), np.arange(50))), shape=(50, 50)
        )

        # WHEN
        a, b = model.find_ab_params(D)

        # THEN
        assert a > 0.0
        assert b > 0.0

    def test_fitted_curve_approximates_target_psi(self):
        """
        GIVEN une matrice de distances et la courbe cible psi associée,
        WHEN  on ajuste a et b via find_ab_params,
        THEN  la courbe 1/(1 + a*d^(2b)) doit approximer psi avec une erreur faible.
        """
        # GIVEN
        model = umap_mapping(min_dist=0.1)
        d_vals = np.linspace(0.05, 3.0, 100)
        psi    = np.where(d_vals <= model.min_dist, 1.0, np.exp(-(d_vals - model.min_dist)))
        D = sp.csr_matrix(
            (d_vals, (np.arange(100), np.arange(100))), shape=(100, 100)
        )

        # WHEN
        a, b = model.find_ab_params(D)
        fitted = 1.0 / (1.0 + a * d_vals ** (2 * b))

        # THEN
        rmse = np.sqrt(np.mean((fitted - psi) ** 2))
        assert rmse < 0.05, f"RMSE trop élevé : {rmse:.4f}"

    def test_larger_min_dist_shifts_curve_right(self):
        """
        GIVEN deux modèles avec des min_dist différents (0.1 vs 0.5),
        WHEN  on ajuste a et b pour chacun,
        THEN  un min_dist plus grand doit produire un b plus petit
              (la courbe décroît moins vite, reflétant un espace plus étalé).
        """
        # GIVEN
        d_vals = np.linspace(0.05, 3.0, 100)
        D = sp.csr_matrix(
            (d_vals, (np.arange(100), np.arange(100))), shape=(100, 100)
        )
        model_tight  = umap_mapping(min_dist=0.1)
        model_spread = umap_mapping(min_dist=0.5)

        # WHEN
        _, b_tight  = model_tight.find_ab_params(D)
        _, b_spread = model_spread.find_ab_params(D)

        # THEN
        assert b_spread < b_tight


class TestSpectralEmbedding:

    def test_output_shape_is_n_samples_by_n_components(self):
        """
        GIVEN une matrice de poids symétrique valide pour 8 points,
        WHEN  on appelle spectral_embedding avec n_components=2,
        THEN  la sortie doit avoir la forme (8, 2).
        """
        # GIVEN
        model = umap_mapping(n_components=2)
        rng = np.random.default_rng(0)
        n = 8
        A = rng.uniform(0.1, 1.0, (n, n))
        W = sp.csr_matrix((A + A.T) / 2)   # symétrique

        # WHEN
        Y = model.spectral_embedding(W)

        # THEN
        assert Y.shape == (n, 2)

    def test_embedding_is_real_valued(self):
        """
        GIVEN une matrice de poids symétrique définie positive,
        WHEN  on appelle spectral_embedding,
        THEN  toutes les valeurs de l'embedding doivent être réelles (pas de partie imaginaire).
        """
        # GIVEN
        model = umap_mapping(n_components=2)
        rng = np.random.default_rng(1)
        n = 10
        A = rng.uniform(0.1, 1.0, (n, n))
        W = sp.csr_matrix((A + A.T) / 2)

        # WHEN
        Y = model.spectral_embedding(W)

        # THEN
        assert np.isrealobj(Y), "L'embedding contient des valeurs complexes."

    def test_two_disconnected_clusters_are_separated(self):
        """
        GIVEN un graphe avec deux composantes connexes parfaitement séparées
              (bloc-diagonale),
        WHEN  on appelle spectral_embedding avec n_components=2,
        THEN  les points des deux clusters doivent être séparés sur au moins
              un axe de l'espace latent.
        """
        # GIVEN — bloc-diagonale : deux cliques de 4 sommets sans lien entre elles
        model = umap_mapping(n_components=2)
        block = np.ones((4, 4)) - np.eye(4)
        W_dense = np.block([
            [block,           np.zeros((4, 4))],
            [np.zeros((4, 4)), block           ],
        ])
        W = sp.csr_matrix(W_dense)

        # WHEN
        Y = model.spectral_embedding(W)

        # THEN — les moyennes des deux clusters doivent différer sur au moins un axe
        mean_c0 = Y[:4].mean(axis=0)
        mean_c1 = Y[4:].mean(axis=0)
        assert np.linalg.norm(mean_c0 - mean_c1) > 0.1