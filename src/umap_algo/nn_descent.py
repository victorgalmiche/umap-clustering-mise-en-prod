"""Algorithme basé sur le papier Dong, W., Moses, C., & Li, K. (2011, March).
Efficient k-nearest neighbor graph construction for generic similarity measures.
In Proceedings of the 20th international conference on World wide web (pp. 577-586).
Keeping the name of variables disregarding pylint recommendations.
"""

import heapq

import numpy as np
from scipy.spatial.distance import pdist, squareform


def approx_knn_all_points(X: np.ndarray, k: int, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les k plus proches voisins approximés pour tous les points du dataset.
    S'appuie sur l'algorithme nn_descent.

    Parameters
    ----------
    X : ndarray (N, d)
        Dataset
    k : int
        Nombre de voisins
    metric : str
        Distance (euclidean, manhattan, etc.)

    Returns
    -------
    indices : ndarray (N, k)
        Indices des k plus proches voisins
    distances : ndarray (N, k)
        Distances associées
    """

    distance_matrix = squareform(pdist(X, metric=metric))
    sigma = -distance_matrix

    return nn_descent(X, sigma, k)


def nn_descent(V: np.ndarray, sigma: np.ndarray, K: int, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:  # pylint: disable=too-many-locals,too-many-branches
    """Algorithme 1 de l'article
    Inputs:
        V: ndarray (N, d), the dataset
        sigma: ndarray(N, N), similarity oracle
        K: int, number of neighbors
    Outputs:
        indices: liste des + proches voisins pour chacun des points du dataset
        distances: liste des distances associées
    """
    N, _ = np.shape(V)
    B = [[] for _ in range(N)]
    for v in range(N):
        samples = np.random.choice([u for u in range(N) if u != v], K, replace=False)
        for u in samples:
            heapq.heappush(B[v], (-np.inf, u))

    for _ in range(max_iter):
        R = reverse(B)
        B_bar = [[] for _ in range(N)]
        for v in range(N):
            for w, u in B[v]:
                heapq.heappush(B_bar[v], (w, u))
            for w, u in R[v]:
                heapq.heappush(B_bar[v], (w, u))

        c = 0

        for v in range(N):
            for _, u1 in B_bar[v]:
                for _, u2 in B_bar[u1]:
                    if u2 == v:
                        continue  # Ne pas ajouter v à la liste de ses voisins
                    w = sigma[v, u2]
                    c += update_nn(B[v], (w, u2))

        if c == 0:
            break

    graph = [sorted([(u, -w) for w, u in heap], key=lambda x: x[1]) for heap in B]

    indices = [[] for _ in range(N)]
    distances = [[] for _ in range(N)]
    for v in range(N):
        for u, d in graph[v]:
            indices[v].append(u)
            distances[v].append(d)

    return np.array(indices), np.array(distances)


def reverse(B: list[list[tuple[float, int]]]) -> list[list[tuple[float, int]]]:
    """
    Compute the reverse k-NN graph: for each node, list the points
    that have it as one of their k nearest neighbors.
    """
    N = len(B)
    R = [[] for _ in range(N)]
    for u in range(N):
        for w, v in B[u]:
            R[v].append((w, u))
    return R


def update_nn(H: list[tuple[float, int]], x: tuple[float, int]) -> int:
    """
    Try to insert a candidate neighbor ``x = (similarity, index)`` into
    the max-size-K min-heap ``H``. Returns 1 if the heap was updated,
    0 otherwise (duplicate or similarity too low).
    """

    w, u = x

    # Si u est déjà dans le tas
    for _, v in H:
        if v == u:
            return 0

    # Sinon, si il a une similarité assez grande, on l'ajoute
    if H[0][0] < w:
        heapq.heapreplace(H, x)
        return 1

    return 0
