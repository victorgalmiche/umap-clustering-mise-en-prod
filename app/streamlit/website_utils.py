import requests
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_covtype, fetch_openml
from sklearn.cluster import KMeans, HDBSCAN


def run_umap_api(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    knn_metric: str = "euclidean",
    knn_method: str = "approx"
) -> np.ndarray:
    """
    Call the API to run umap.
    """
    # url = "http://0.0.0.0:8000/umap"
    url = "https://umap-api-mmvs.lab.sspcloud.fr/umap"

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
    }

    response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    data = response.json()

    return np.array(data["embedding"])


def run_kmeans(X, n_clusters):
    return KMeans(n_clusters).fit(X).labels_


def run_hdbscan(X, min_cluster_size):
    return HDBSCAN(min_cluster_size).fit(X).labels_


@st.cache_data
def load_iris_data():
    return load_iris(as_frame=True)["data"]


@st.cache_data
def load_covtype_data(n_points=500):
    covtype = fetch_covtype()
    X = pd.DataFrame(covtype.data)
    return X.iloc[:n_points]


@st.cache_data
def load_fashion_mnist_data(n_points=500, selected_classes=(0, 1, 2)):
    fashion = fetch_openml(name="Fashion-MNIST", version=1, parser="auto")
    X = fashion.data
    y = fashion.target.astype(int)
    mask = y.isin(selected_classes)
    return X[mask].iloc[:n_points]


@st.cache_data
def load_miniboone_data(n_points=500):
    miniboone = fetch_openml(data_id=41150, as_frame=True, parser="auto")
    return miniboone.data.iloc[:n_points]


DATASET_LOADERS = {
    "Iris": load_iris_data,
    "Covtype": load_covtype_data,
    "Fashion-MNIST": load_fashion_mnist_data,
    "MiniBooNE": load_miniboone_data,
}


def load_dataset(name: str):
    return DATASET_LOADERS[name]()