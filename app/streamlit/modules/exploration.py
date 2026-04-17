import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

import app.streamlit.utils.embeddings as emb_utils
import app.streamlit.utils.data_preprocessing as data_utils
import app.streamlit.utils.hyperparameters as param_utils
import app.streamlit.utils.visualization as plot_utils


def render():
    # -----------------------------
    # 1. Data configuration
    # -----------------------------
    st.sidebar.header("Data Configuration")

    data_source = data_utils.fetch_data_source()

    target_column = None

    if data_source == "Upload CSV":
        data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        data_to_embed, target_column = data_utils.fetch_csv_file(data_file)
    else:
        data_to_embed, target_column = data_utils.fetch_default_data()

    st.write("### Dataset preview")
    st.dataframe(data_to_embed.head())

    n_valid_columns = data_to_embed.select_dtypes(include=[np.number]).shape[1]
    if (target_column is not None) and (target_column in data_to_embed):
        n_valid_columns -= 1

    if n_valid_columns <= 2:
        st.error("Please provide data with at least 3 columns to embed into smaller dimension.")

    # -----------------------------
    # 2. UMAP parameters
    # -----------------------------
    st.sidebar.header("UMAP Parameters")

    param_utils.default_umap_params()
    umap_params = param_utils.select_umap_params(n_valid_columns)

    # -----------------------------
    # 3. Run UMAP
    # -----------------------------
    st.header("Run UMAP")

    save_model = st.checkbox("Do you want to save your model? (Experimental)")

    if st.button("Run UMAP"):
        if target_column is not None:
            X_df = data_to_embed.drop(columns=[target_column])
        else:
            X_df = data_to_embed.copy()
        X_df = X_df.select_dtypes(include=[np.number])

        with st.spinner("Running UMAP..."):
            if save_model:
                embedding, access_key = emb_utils.run_umap_api(
                    df=X_df,
                    n_neighbors=umap_params["n_neighbors"],
                    n_components=umap_params["n_components"],
                    min_dist=umap_params["min_dist"],
                    knn_metric=umap_params["knn_metric"],
                    knn_method=umap_params["knn_method"],
                    n_epochs=umap_params["n_epochs"],
                    mode="train"
                )

            else:
                embedding = emb_utils.run_umap_api(
                    df=X_df,
                    n_neighbors=umap_params["n_neighbors"],
                    n_components=umap_params["n_components"],
                    min_dist=umap_params["min_dist"],
                    knn_metric=umap_params["knn_metric"],
                    knn_method=umap_params["knn_method"],
                    n_epochs=umap_params["n_epochs"],
                    mode="umap"
                )

        st.session_state["embedding"] = embedding

        if save_model:
            st.success(f"UMAP completed, key to load model (for transform):    {access_key}")
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
            metric=umap_params["knn_metric"],
            n_neighbors=umap_params["n_neighbors"]
        )
        st.metric(label="UMAP Trustworthiness", value=f"{trust:.4f}")

        # -----------------------------
        # 5. Visualization
        # -----------------------------
        st.header("UMAP Visualization")

        plot_utils.show_embeddings(
            embedding=embedding,
            data_to_embed=data_to_embed,
            target_column=target_column
        )

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

        clustering_method = param_utils.select_clustering_method(key_suffix="_explore")
        clustering_param = param_utils.select_clustering_param(
            clustering_method=clustering_method,
            n_samples=len(embedding),
            key_suffix="_explore")

        if st.button("Run Clustering", key="run_clustering"):
            with st.spinner("Clustering..."):
                if clustering_method == "KMeans":
                    labels = emb_utils.run_kmeans(embedding, clustering_param)
                else:
                    labels = emb_utils.run_hdbscan(embedding, clustering_param)

            st.session_state["labels"] = labels
            st.success("Clustering completed")

    if "embedding" in st.session_state and "labels" in st.session_state:
        embedding = st.session_state["embedding"]
        labels = st.session_state["labels"]

        # -----------------------------
        # 7. Metrics
        # -----------------------------
        if len(set(labels)) >= 2:
            sil_score = silhouette_score(embedding, labels)
            st.metric(label="Silhouette Score", value=f"{sil_score:.4f}")

        # -----------------------------
        # 8. Visualization with clusters
        # -----------------------------
        st.header("Clustered Visualization")

        plot_utils.show_clusters(embedding=embedding, labels=labels)

        result_df = pd.DataFrame(embedding, columns=[f"dim_{i}" for i in range(embedding.shape[1])])
        result_df["cluster"] = labels

        csv = result_df.to_csv(index=False).encode()
        st.download_button(
            "Download embedding + clusters",
            csv,
            "embedding_clusters.csv", "text/csv"
        )
