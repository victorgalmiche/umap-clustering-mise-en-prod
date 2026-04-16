import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN


def run_umap_api(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    knn_metric: str = "euclidean",
    knn_method: str = "approx",
    n_epochs: int = 200,
    mode: str = "umap"
) -> np.ndarray:
    """
    Call the API to run umap.
    mode: umap or train
    """
    assert mode in ["umap", "train"], ValueError("Invalid mode")

    # url = f"http://0.0.0.0:8000/{mode}"
    url = f"https://umap-api-mmvs.lab.sspcloud.fr/{mode}"

    csv_buffer = df.to_csv(index=False).encode()

    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }

    data = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": min_dist,
        "knn_metric": knn_metric,
        "knn_method": knn_method,
        "x_client_source": "streamlit"
    }

    response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    data = response.json()

    if mode == "umap":
        return np.array(data["embedding"])
    if mode == "train":
        return np.array(data["embedding"]), data["access_key"]


def run_kmeans(X, n_clusters):
    return KMeans(n_clusters).fit(X).labels_


def run_hdbscan(X, min_cluster_size):
    return HDBSCAN(min_cluster_size).fit(X).labels_
