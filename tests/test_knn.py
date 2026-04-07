import numpy as np
import pytest

from umap_algo.knn import exact_knn_all_points


class TestExactKnnAllPoints:
    """Test suite for exact_knn_all_points function."""

    def test_positive_knn_distances():
        """ Distanes should be positive """
        # Given
        X = np.random.rand(8, 4)
        k = 3

        # When
        _, distances = exact_knn_all_points(X, k)

        # Then
        assert np.all(distances > 0)
    
    @pytest.mark.parametrize(
        "point_idx, expected_indices, expected_distances",
        [
            (0, [1, 2], [1.0, 2.0]),
            (2, [1, 3], [1.0, 1.0]),  
            (4, [3, 2], [1.0, 2.0]),
        ],
    )
    def test_exact_neighbors_and_distances_no_train(point_idx, expected_indices, expected_distances):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        indices, distances = exact_knn_all_points(X, k=2)

        assert indices.shape == (5, 2)
        assert distances.shape == (5, 2)

        if point_idx == 2:
            # cas symétrique → ordre non garanti
            assert set(indices[point_idx]) == set(expected_indices)
            np.testing.assert_allclose(sorted(distances[point_idx]), sorted(expected_distances))
        else:
            assert list(indices[point_idx]) == expected_indices
            np.testing.assert_allclose(distances[point_idx], expected_distances)


    def test_exact_neighbors_and_distances_no_train(self):
        """
        For a known 1-D grid the nearest neighbours and their distances
        must be exactly predictable.

        Point 0 (x=0): nearest is point 1 (d=1), then point 2 (d=2)
        Point 2 (x=2): nearest are points 1 & 3 (d=1 each), then 0 & 4 (d=2)
        Point 4 (x=4): nearest is point 3 (d=1), then point 2 (d=2)
        """
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        indices, distances = exact_knn_all_points(X, k=2)

        assert indices.shape == (5, 2)
        assert distances.shape == (5, 2)

        # --- point 0 (x=0) ---
        assert list(indices[0]) == [1, 2], "point 0: neighbours should be 1 then 2"
        np.testing.assert_allclose(distances[0], [1.0, 2.0])

        # --- point 2 (x=2) – symmetric middle point ---
        assert set(indices[2]) == {1, 3}, "point 2: neighbours should be 1 and 3"
        np.testing.assert_allclose(sorted(distances[2]), [1.0, 1.0])

        # --- point 4 (x=4) ---
        assert list(indices[4]) == [3, 2], "point 4: neighbours should be 3 then 2"
        np.testing.assert_allclose(distances[4], [1.0, 2.0])

    def test_exact_neighbors_and_distances_with_train(self):
        """
        Query a single point against a known training set and verify
        both indices and distances are pixel-perfect.

        Training set: (0,0), (1,0), (0,1), (3,3)
        Query point : (0.1, 0.1)  → closest are idx 0 (d≈0.141), idx 1 (d≈0.9), idx 2 (d≈0.9)
        """
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
        X_query = np.array([[0.1, 0.1]])

        indices, distances = exact_knn_all_points(X_query, k=3, X_train=X_train)

        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)

        expected_indices = [0, 1, 2]          # sorted by distance
        expected_distances = [
            np.sqrt(0.1**2 + 0.1**2),         # ≈ 0.1414
            np.sqrt(0.9**2 + 0.1**2),         # ≈ 0.9055
            np.sqrt(0.1**2 + 0.9**2),         # ≈ 0.9055
        ]

        assert list(indices[0]) == expected_indices
        np.testing.assert_allclose(distances[0], expected_distances, atol=1e-6)

    # ------------------------------------------------------------------
    # Output-shape tests
    # ------------------------------------------------------------------

    def test_output_shape_no_train(self):
        X = np.random.rand(10, 3)
        indices, distances = exact_knn_all_points(X, k=3)
        assert indices.shape == (10, 3)
        assert distances.shape == (10, 3)

    def test_output_shape_with_train(self):
        X_train = np.random.rand(20, 3)
        X_query = np.random.rand(5, 3)
        indices, distances = exact_knn_all_points(X_query, k=4, X_train=X_train)
        assert indices.shape == (5, 4)
        assert distances.shape == (5, 4)

    # ------------------------------------------------------------------
    # Self-exclusion: without X_train a point must never be its own neighbour
    # ------------------------------------------------------------------

    def test_no_self_in_neighbors(self):
        X = np.random.rand(15, 2)
        indices, distances = exact_knn_all_points(X, k=3)
        for i, row in enumerate(indices):
            assert i not in row, f"point {i} must not appear among its own neighbours"

    def test_distances_are_positive_no_train(self):
        X = np.random.rand(10, 2)
        _, distances = exact_knn_all_points(X, k=3)
        assert np.all(distances > 0), "self-exclusion: all distances must be strictly positive"

    # ------------------------------------------------------------------
    # k clipping when X_train is smaller than requested k
    # ------------------------------------------------------------------

    def test_k_clipped_to_train_size(self):
        """When k > len(X_train), k must be silently clipped."""
        X_train = np.random.rand(3, 2)
        X_query = np.random.rand(2, 2)
        indices, distances = exact_knn_all_points(X_query, k=10, X_train=X_train)
        # Only 3 training points available → at most 3 neighbours
        assert indices.shape[1] == 3
        assert distances.shape[1] == 3

    # ------------------------------------------------------------------
    # Distances must be non-decreasing along each row
    # ------------------------------------------------------------------

    def test_distances_sorted_no_train(self):
        X = np.random.rand(20, 4)
        _, distances = exact_knn_all_points(X, k=5)
        for row in distances:
            assert list(row) == sorted(row), "distances must be in ascending order"

    def test_distances_sorted_with_train(self):
        X_train = np.random.rand(30, 4)
        X_query = np.random.rand(10, 4)
        _, distances = exact_knn_all_points(X_query, k=5, X_train=X_train)
        for row in distances:
            assert list(row) == sorted(row), "distances must be in ascending order"

    # ------------------------------------------------------------------
    # Metric forwarding
    # ------------------------------------------------------------------

    def test_manhattan_metric_exact(self):
        """
        Simple 2-D case with Manhattan distance.

        Points: A=(0,0), B=(1,0), C=(0,2)
        For A: d_manhattan(A,B)=1, d_manhattan(A,C)=2 → neighbour order B, C
        """
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        indices, distances = exact_knn_all_points(X, k=2, metric="manhattan")

        assert list(indices[0]) == [1, 2]
        np.testing.assert_allclose(distances[0], [1.0, 2.0])

    # ------------------------------------------------------------------
    # Single-point edge case (no X_train)
    # ------------------------------------------------------------------

    def test_single_query_point_with_train(self):
        X_train = np.array([[0.0], [5.0], [10.0]])
        X_query = np.array([[4.9]])
        indices, distances = exact_knn_all_points(X_query, k=1, X_train=X_train)
        assert indices[0, 0] == 1          # closest training point is index 1 (x=5)
        np.testing.assert_allclose(distances[0, 0], 0.1, atol=1e-6)