import numpy as np
import pytest

from umap_algo.knn import exact_knn_all_points


class TestExactKnnAllPoints:
    """Test suite for exact_knn_all_points function."""

    @pytest.mark.parametrize(
        "point_idx, expected_indices, expected_distances", 
        [
            (0, [1, 2], [1.0, 2.0]),
            (2, [1, 3], [1.0, 1.0]),  # distances will be sorted for comparison
            (4, [3, 2], [1.0, 2.0])
        ]
    )
    def test_exact_neighbors_and_distance(self, point_idx, expected_indices, expected_distances):
        """
        GIVEN a known 1-D grid, the nearest neighbours and their distances,
        WHEN calling exact_knn_all_points,
        THEN the desired neighbours should be found.
        """
        # GIVEN
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

        # WHEN
        indices_all, distances_all = exact_knn_all_points(X, k=2)
        indices = indices_all[point_idx]
        distances = distances_all[point_idx]

        # THEN
        if point_idx == 2:
            assert set(indices) == set(expected_indices), f"Point {point_idx}: neighbours mismatch"
            np.testing.assert_allclose(sorted(distances), sorted(expected_distances))
        else:
            assert list(indices) == expected_indices, f"Point {point_idx}: neighbours mismatch"
            np.testing.assert_allclose(distances, expected_distances)

    def test_exact_neighbors_and_distances_with_train(self):
        """
        GIVEN Query a single point against a known training set and verify
            both indices and distances are pixel-perfect.
        WHEN call exact_knn_all_points
        THEN 
            Training set: (0,0), (1,0), (3,3)
            Query point : (0.1, 0.1)  → closest are idx 0 (d≈0.141), idx 1 (d≈0.9)
        """
        # GIVEN
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 3.0]])
        X_query = np.array([[0.1, 0.1]])

        # WHEN
        indices, distances = exact_knn_all_points(X_query, k=2, X_train=X_train)

        # THEN 
        expected_indices = [np.int64(0), np.int64(1)]
        expected_distances = [
            np.sqrt(0.1**2 + 0.1**2),        
            np.sqrt(0.9**2 + 0.1**2),         
        ]
        assert list(indices[0]) == expected_indices
        np.testing.assert_allclose(distances[0], expected_distances, atol=1e-6)

    # ------------------------------------------------------------------
    # Output-shape tests
    # ------------------------------------------------------------------

    def test_output_shape_no_train(self):
        """
        GIVEN a random 10x3 dataset X,
        WHEN calling exact_knn_all_points on X with k=3 and no training data,
        THEN the returned indices and distances must have shape (10, 3).
        """
        X = np.random.rand(10, 3)
        indices, distances = exact_knn_all_points(X, k=3)
        assert indices.shape == (10, 3)
        assert distances.shape == (10, 3)

    def test_output_shape_with_train(self):
        """
        GIVEN a random 20x3 training dataset X_train and 5x3 query dataset X_query,
        WHEN calling exact_knn_all_points on X_query with k=4 using X_train,
        THEN the returned indices and distances must have shape (5, 4).
        """
        X_train = np.random.rand(20, 3)
        X_query = np.random.rand(5, 3)
        indices, distances = exact_knn_all_points(X_query, k=4, X_train=X_train)
        assert indices.shape == (5, 4)
        assert distances.shape == (5, 4)

    # ------------------------------------------------------------------
    # Self-exclusion: without X_train a point must never be its own neighbour
    # ------------------------------------------------------------------

    def test_no_self_in_neighbors(self):
        """
        GIVEN a random 15x2 dataset X,
        WHEN calling exact_knn_all_points on X with k=3,
        THEN no point should appear among its own nearest neighbors.
        """
        X = np.random.rand(15, 2)
        indices, distances = exact_knn_all_points(X, k=3)
        for i, row in enumerate(indices):
            assert i not in row, f"point {i} must not appear among its own neighbours"

    def test_distances_are_positive_no_train(self):
        """
        GIVEN a random 10x2 dataset X,
        WHEN calling exact_knn_all_points on X with k=3,
        THEN all returned distances must be strictly positive (self-exclusion enforced).
        """
        X = np.random.rand(10, 2)
        _, distances = exact_knn_all_points(X, k=3)
        assert np.all(distances > 0), "self-exclusion: all distances must be strictly positive"

    # ------------------------------------------------------------------
    # k clipping when X_train is smaller than requested k
    # ------------------------------------------------------------------

    def test_k_clipped_to_train_size(self):
        """
        GIVEN k > len(X_train), 
        WHEN call exact_knn_all_points
        THEN k must be silently clipped.
        """
        X_train = np.random.rand(3, 2)
        X_query = np.random.rand(2, 2)
        indices, distances = exact_knn_all_points(X_query, k=10, X_train=X_train)
        # Only 3 training points available → at most 3 neighbours
        assert indices.shape[1] == 3
        assert distances.shape[1] == 3

    # ------------------------------------------------------------------
    # Metric forwarding
    # ------------------------------------------------------------------

    def test_manhattan_metric_exact(self):
        """
        GIVEN Simple 2-D case with Manhattan distance.
                Points: A=(0,0), B=(1,0), C=(0,2)
        WHEN Call exact_knn_all_points
        THEN For A: d_manhattan(A,B)=1, d_manhattan(A,C)=2 → neighbour order B, C
        """
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        indices, distances = exact_knn_all_points(X, k=2, metric="manhattan")

        assert list(indices[0]) == [1, 2]
        np.testing.assert_allclose(distances[0], [1.0, 2.0])
