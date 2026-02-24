import numpy as np
import pytest

from umapverse.umap_algo.knn import exact_knn_all_points


def test_positive_knn_distances():
    """ Distanes should be positive """
    # Given
    X = np.random.rand(8, 4)
    k = 3

    # When
    _, distances = exact_knn_all_points(X, k)

    # Then 
    assert np.all(distances > 0)