import streamlit as st


def default_umap_params():
    if "n_components" not in st.session_state:
        st.session_state.n_components = 2
    if "n_neighbors" not in st.session_state:
        st.session_state.n_neighbors = 15
    if "n_epochs" not in st.session_state:
        st.session_state.n_epochs = 150
    if "min_dist" not in st.session_state:
        st.session_state.min_dist = 0.1
    if "knn_metric" not in st.session_state:
        st.session_state.knn_metric = "euclidean"
    if "knn_method" not in st.session_state:
        st.session_state.knn_method = "approx"


def select_umap_params(n_valid_columns):
    params = dict()

    params["n_components"] = st.sidebar.slider(
        "n_components",
        2,
        min(n_valid_columns, 10),
        key="n_components"
    )
    params["n_neighbors"] = st.sidebar.slider(
        "n_neighbors",
        2,
        100,
        key="n_neighbors"
    )
    params["n_epochs"] = st.sidebar.slider(
        "n_epochs",
        10,
        300,
        key="n_epochs"
    )
    params["min_dist"] = st.sidebar.slider(
        "min_dist",
        0.01,
        1.0,
        key="min_dist"
    )
    params["knn_metric"] = st.sidebar.selectbox(
        "KNN metric",
        ['euclidean', 'chebyshev', 'minkowski'],
        index=0,
        key="knn_metric"
    )
    params["knn_method"] = st.sidebar.selectbox(
        "KNN method",
        ["exact", "approx"],
        index=0,
        key="knn_method"
    )

    return params


def select_clustering_method(key_suffix):
    return st.selectbox(
            "Method",
            ["KMeans", "HDBSCAN"],
            key="clusering_method" + key_suffix
    )


def select_clustering_param(clustering_method, n_samples, key_suffix):
    if clustering_method == "KMeans":
        return st.slider("n_clusters", 2, min(20, n_samples), 2, key="n_clusters_" + key_suffix)
    else:
        return st.slider("min_cluster_size", 2, min(50, n_samples-1), 5, key="min_cluster_size" + key_suffix)
