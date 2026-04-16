import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score
from app.streamlit.website_utils import run_umap_transform, run_hdbscan, run_kmeans


def render():
    """
    Render the UI for the /transform endpoint (Model Inference).
    """
    st.subheader("🚀 Project New Data")
    st.markdown("""
    Use a previously generated **Access Key** to project new data points into
    your existing UMAP latent space.
    """)

    @st.cache_data
    def load_and_sample(df):
        """Limit to 500 lines, deterministic"""
        return df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)

    # --- 1. Security & Key ---
    st.info("Enter your secure token to access your private model.")

    # Pre-fill the key if it exists in session_state from the Training tab
    saved_key = st.session_state.get("access_key", "")
    access_key = st.text_input(
        "Access Key",
        value=saved_key,
        type="password",
        help="The key generated during the /train process."
    )

    # --- 2. New Data Upload ---
    st.divider()
    new_data_file = st.file_uploader(
        "Upload new samples (CSV)",
        type=["csv"],
        help="Ensure columns match the training dataset (numerical features)."
    )

    if new_data_file is not None:
        df_new = pd.read_csv(new_data_file)

        # Limit the size for fast run of UMAP
        if len(df_new) >= 500:
            st.warning("Too many rows (>= 500), data will be sampled.")
            df_new = load_and_sample(df_new)

        target_column = st.selectbox(
            "Select the target column",
            options=[None] + list(df_new.columns),
            help="This column will be ignored by UMAP but used for visualization."
        )

    else:
        st.warning("Waiting for data...")
        st.stop()

    # --- 3. Hyperparameters ---
    n_epochs_trans = st.slider(
        "Optimization Epochs",
        min_value=1,
        max_value=200,
        value=100,
        help="Number of iterations for the projection refinement."
    )

    # --- 4. Execution & Results ---
    if st.button("Apply Transformation", type="primary"):
        if not access_key:
            st.error("Missing Access Key.")
            return

        if new_data_file is None:
            st.error("Please upload a CSV file.")
            return

        # Data preparation
        if target_column is not None:
            df_to_transform = df_new.drop(columns=[target_column])
        else:
            df_to_transform = df_new.copy()

        with st.spinner("Contacting UMAP API for transformation..."):
            new_emb = run_umap_transform(
                df=df_to_transform,
                access_key=access_key,
                n_epochs=n_epochs_trans
            )

        # Store results in session state
        st.session_state["new_embedding"] = new_emb
        st.session_state["new_df_preview"] = df_new

        st.success(f"Successfully transformed {len(df_new)} points.")

    if "new_embedding" in st.session_state:
        new_emb = st.session_state["new_embedding"]

        # TODO: add trustworthiness (need to modify API)

        # --- 5. Visualization ---
        st.write("### Projection Results")

        if new_emb.shape[1] == 2:
            plot_df = pd.DataFrame(new_emb, columns=["x", "y"])
            if target_column:
                plot_df[target_column] = df_new[target_column].astype(str)
            fig = px.scatter(plot_df, x="x", y="y", color=target_column)
            st.plotly_chart(fig, width='stretch')

        # Download result
        csv_data = pd.DataFrame(
            new_emb,
            columns=[f"dim_{i}" for i in range(new_emb.shape[1])]
        ).to_csv(index=False).encode()

        st.download_button(
            "Download Projected Points",
            csv_data,
            "projected_data.csv",
            "text/csv"
        )

        # -----------------------------
        # 6. Clustering
        # -----------------------------
        st.header("Clustering")

        new_clustering_method = st.selectbox(
            "Method",
            ["KMeans", "HDBSCAN"],
            key="new_clustering_method"
        )

        if new_clustering_method == "KMeans":
            new_n_clusters = st.slider(
                "n_clusters",
                2,
                20,
                5
            )
        else:
            new_min_cluster_size = st.slider(
                "min_cluster_size",
                2,
                50,
                5
            )

        if st.button("Run Clustering", key="new_run_clustering"):
            with st.spinner("Clustering..."):
                if new_clustering_method == "KMeans":
                    new_labels = run_kmeans(new_emb, new_n_clusters)
                else:
                    new_labels = run_hdbscan(new_emb, new_min_cluster_size)

            st.session_state["new_labels"] = new_labels
            st.success("Clustering completed")

    if "new_embedding" in st.session_state and "new_labels" in st.session_state:
        new_emb = st.session_state["new_embedding"]
        new_labels = st.session_state["new_labels"]

        # -----------------------------
        # 7. Metrics
        # -----------------------------
        sil_score = silhouette_score(new_emb, new_labels)
        st.metric(label="Silhouette Score", value=f"{sil_score:.4f}")

        # -----------------------------
        # 8. Visualization with clusters
        # -----------------------------
        st.header("Clustered Visualization")

        if new_emb.shape[1] == 2:
            plot_df = pd.DataFrame({
                "x": new_emb[:, 0],
                "y": new_emb[:, 1],
                "cluster": new_labels.astype(str)
            })
            fig = px.scatter(plot_df, x="x", y="y", color="cluster")
            st.plotly_chart(fig, width='stretch')

        result_df = pd.DataFrame(new_emb, columns=[f"dim_{i}" for i in range(new_emb.shape[1])])
        result_df["cluster"] = new_labels

        csv = result_df.to_csv(index=False).encode()
        st.download_button(
            "Download new embedding + clusters",
            csv,
            "embedding_clusters.csv", "text/csv"
        )