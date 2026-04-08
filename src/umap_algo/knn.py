import numpy as np
from sklearn.neighbors import KDTree
from typing import Tuple


def exact_knn_all_points(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    X_train: np.ndarray = np.array([]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the k nearest neighbors for all points in dataset X using KDTree.

    Parameters
    ----------
    X : ndarray (N, d) - dataset
    k : int - number of neighbors
    metric : str - distance (euclidean, manhattan, etc.)
    X_train : array-like shape (n_samples, n_features) - training set


    Returns
    -------
    indices : ndarray (N, k) - indices of the k nearest neighbors
    distances : ndarray (N, k) - associated distances
    """

    if X_train.size > 0:
        tree = KDTree(X_train, metric=metric)
        k = min(k, X_train.shape[0])
        distances, indices = tree.query(X, k=k)
        return indices, distances
    else:
        tree = KDTree(X, metric=metric)
        # k+1 car le point lui-même est retourné
        distances, indices = tree.query(X, k=k + 1)
        # On enlève le point lui-même (distance nulle)
        return indices[:, 1:], distances[:, 1:]
