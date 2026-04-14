import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

# Placeholder imports for your custom implementations
from src.umap_algo.umap_class import umap_mapping
from app.streamlit.website_utils import run_umap_api, run_kmeans, run_hdbscan

st.set_page_config(
    page_title="UMAP + Clustering App",
    layout="wide",
    menu_items={"About": "Version 0.1"}
)

st.title("UMAP Dimensionality Reduction & Clustering")

# -----------------------------
# 1. Data loading
# -----------------------------
st.sidebar.header("Data")

data_file = st.sidebar.file_uploader("Upload data to embed (CSV)", type=["csv"])
# TODO: add target
# TODO: add parquet


@st.cache_data
def load_default_data():
    return load_iris(as_frame=True)["data"]


if data_file is not None:
    data_to_embed = pd.read_csv(data_file)
else:
    st.sidebar.info("Using default dataset")
    data_to_embed = load_default_data()

st.write("### Dataset preview")
st.dataframe(data_to_embed.head())

# -----------------------------
# 2. UMAP parameters
# -----------------------------
st.sidebar.header("UMAP Parameters")

n_neighbors = st.sidebar.slider("n_neighbors", 2, 100, 2)
n_components = st.sidebar.slider("n_components", 2, min(data_to_embed.shape[1]-1, 10), 2)
min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1)

knn_metric = st.sidebar.selectbox("KNN metric", ["euclidean", "manhattan", "cosine"], index=0)
knn_method = st.sidebar.selectbox("KNN method", ["exact", "approx"], index=0)

# -----------------------------
# 3. Animation option
# -----------------------------
show_animation = False

if n_neighbors == 2:
    show_animation = st.sidebar.checkbox("Show optimization animation")

# -----------------------------
# 4. Run UMAP
# -----------------------------
st.header("Run UMAP")

if st.button("Run UMAP"):
    X_df = data_to_embed.select_dtypes(include=[np.number])

    with st.spinner("Running UMAP..."):
        if show_animation:
            X = X_df.values
            reducer = umap_mapping(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                KNN_metric=knn_metric,
                KNN_method=knn_method
            )
            anim = reducer.animate_optimization(X)
            st.pyplot(anim)
            st.warning("Animation does not scale and does not return embeddings")
            embedding = run_umap_api(
                df=X_df,
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                knn_metric=knn_metric,
                knn_method=knn_method
            )

        else:
            embedding = run_umap_api(
                df=X_df,
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                knn_metric=knn_metric,
                knn_method=knn_method
            )

    st.session_state["embedding"] = embedding
    st.success("UMAP completed")

# -----------------------------
# 5. Visualization
# -----------------------------
if "embedding" in st.session_state:
    embedding = st.session_state["embedding"]
    st.header("UMAP Visualization")

    if embedding.shape[1] == 2:
        plot_df = pd.DataFrame(embedding, columns=["x", "y"])
        fig = px.scatter(plot_df, x="x", y="y")
        st.plotly_chart(fig, use_container_width=True)

    download_df = pd.DataFrame(embedding, columns=[f"dim_{i}" for i in range(embedding.shape[1])])
    csv = download_df.to_csv(index=False).encode()

    st.download_button("Download embedding", csv, "embedding.csv", "text/csv")

    # -----------------------------
    # 6. Clustering
    # -----------------------------
    st.header("Clustering")

    clustering_method = st.selectbox("Method", ["KMeans", "HDBSCAN"])

    if clustering_method == "KMeans":
        n_clusters = st.slider("n_clusters", 2, 20, 5)
    else:
        min_cluster_size = st.slider("min_cluster_size", 2, 50, 5)

    if st.button("Run Clustering"):
        with st.spinner("Clustering..."):
            if clustering_method == "KMeans":
                labels = run_kmeans(embedding, n_clusters)
            else:
                labels = run_hdbscan(embedding, min_cluster_size)

        st.session_state["labels"] = labels
        st.success("Clustering completed")

# -----------------------------
# 7. Visualization with clusters
# -----------------------------
if "embedding" in st.session_state and "labels" in st.session_state:
    embedding = st.session_state["embedding"]
    labels = st.session_state["labels"]

    st.header("Clustered Visualization")

    if embedding.shape[1] == 2:
        plot_df = pd.DataFrame({
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "cluster": labels.astype(str)
        })
        fig = px.scatter(plot_df, x="x", y="y", color="cluster")
        st.plotly_chart(fig, use_container_width=True)

    result_df = pd.DataFrame(embedding, columns=[f"dim_{i}" for i in range(embedding.shape[1])])
    result_df["cluster"] = labels

    csv = result_df.to_csv(index=False).encode()
    st.download_button("Download embedding + clusters", csv, "embedding_clusters.csv", "text/csv")
