import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

from app.streamlit.website_utils import run_umap_api, run_kmeans, run_hdbscan

DATASET_OPTIONS = {
    "Digits (Images of handwritten digits)": load_digits,
    "Iris (Flower specifications)": load_iris,
    "Wine (specifications)": load_wine,
    "Breast Cancer (specifications)": load_breast_cancer
}


def render():

    # -----------------------------
    # 1. Data loading
    # -----------------------------

    st.sidebar.header("Data Configuration")

    def reset_state():
        if "embedding" in st.session_state:
            del st.session_state["embedding"]
        if "labels" in st.session_state:
            del st.session_state["labels"]

    # Choix de la source
    data_source = st.sidebar.radio(
        "Data Selection",
        ["Standard Datasets", "Upload CSV"],
        on_change=reset_state
    )

    @st.cache_data
    def load_and_sample(df):
        """Limit to 500 lines, deterministic"""
        return df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)

    target_column = None

    if data_source == "Upload CSV":
        data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if data_file is not None:
            raw_df = pd.read_csv(data_file)

            # Limit the size for fast run of UMAP
            if len(raw_df) >= 500:
                st.warning("Too many rows (>= 500), data will be sampled.")
                data_to_embed = load_and_sample(raw_df)
            else:
                data_to_embed = raw_df

            # The user choses target column
            target_column = st.sidebar.selectbox(
                "Select the target column",
                options=[None] + list(data_to_embed.columns),
                help="This column will be ignored by UMAP but used for visualization."
            )
        else:
            st.sidebar.warning("Waiting for file...")
            st.stop()
    else:
        selected_name = st.sidebar.selectbox(
            "Chose a dataset",
            list(DATASET_OPTIONS.keys()),
            on_change=reset_state
        )
        loader = DATASET_OPTIONS[selected_name]
        data = loader(as_frame=True)

        df_full = data["data"].copy()
        df_full["target"] = data["target"]

        data_to_embed = load_and_sample(df_full)
        target_column = "target"

    st.write("### Dataset preview")
    st.dataframe(data_to_embed.head())

    if (data_to_embed.shape[1] <= 3 and target_column) or data_to_embed.shape[1] <= 2:
        st.error("Please provide data with at least 3 columns to embed into smaller dimension.")

    # -----------------------------
    # 2. UMAP parameters
    # -----------------------------
    st.sidebar.header("UMAP Parameters")

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

    n_components = st.sidebar.slider(
        "n_components",
        2,
        min(data_to_embed.shape[1] - 1, 10),
        key="n_components"
    )
    n_neighbors = st.sidebar.slider(
        "n_neighbors",
        2,
        100,
        key="n_neighbors"
    )
    n_epochs = st.sidebar.slider(
        "n_epochs",
        10,
        300,
        key="n_epochs"
    )
    min_dist = st.sidebar.slider(
        "min_dist",
        0.0,
        1.0,
        key="min_dist"
    )
    knn_metric = st.sidebar.selectbox(
        "KNN metric",
        ["euclidean", "manhattan", "cosine"],
        index=0,
        key="knn_metric"
    )
    knn_method = st.sidebar.selectbox(
        "KNN method",
        ["exact", "approx"],
        index=0,
        key="knn_method"
    )

    # -----------------------------
    # 3. Run UMAP
    # -----------------------------
    st.header("Run UMAP")

    save_model = st.checkbox("Do you want to save your model?")

    if st.button("Run UMAP"):
        X_df = data_to_embed.drop(columns=[target_column])
        X_df = X_df.select_dtypes(include=[np.number])

        with st.spinner("Running UMAP..."):
            if save_model:
                embedding, access_key = run_umap_api(
                    df=X_df,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    knn_metric=knn_metric,
                    knn_method=knn_method,
                    n_epochs=n_epochs,
                    mode="train"
                )

            else:
                embedding = run_umap_api(
                    df=X_df,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    knn_metric=knn_metric,
                    knn_method=knn_method,
                    n_epochs=n_epochs,
                    mode="umap"
                )

        st.session_state["embedding"] = embedding

        if save_model:
            st.success(f"UMAP completed, key to load model (for transform): {access_key}")
        else:
            st.success("UMAP completed")

    if "embedding" in st.session_state:
        embedding = st.session_state["embedding"]

        # -----------------------------
        # 4. Metrics
        # -----------------------------
        trust = trustworthiness(
            data_to_embed,
            embedding,
            metric=knn_metric,
            n_neighbors=n_neighbors
        )
        st.metric(label="UMAP Trustworthiness", value=f"{trust:.4f}")

        # -----------------------------
        # 5. Visualization
        # -----------------------------
        st.header("UMAP Visualization")

        if embedding.shape[1] == 2:
            plot_df = pd.DataFrame(embedding, columns=["x", "y"])
            plot_df[target_column] = data_to_embed[target_column].astype(str)
            fig = px.scatter(plot_df, x="x", y="y", color=target_column)
            st.plotly_chart(fig, use_container_width=True)

        download_df = pd.DataFrame(
            embedding,
            columns=[f"dim_{i}" for i in range(embedding.shape[1])]
        )
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

    if "embedding" in st.session_state and "labels" in st.session_state:
        embedding = st.session_state["embedding"]
        labels = st.session_state["labels"]

        # -----------------------------
        # 7. Metrics
        # -----------------------------
        sil_score = silhouette_score(embedding, labels)
        st.metric(label="Silhouette Score", value=f"{sil_score:.4f}")

        # -----------------------------
        # 8. Visualization with clusters
        # -----------------------------
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
        st.download_button(
            "Download embedding + clusters",
            csv,
            "embedding_clusters.csv", "text/csv"
        )
