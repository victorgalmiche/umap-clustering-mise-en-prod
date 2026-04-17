import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score

import app.streamlit.utils.embeddings as emb_utils
import app.streamlit.utils.data_preprocessing as data_utils
import app.streamlit.utils.hyperparameters as param_utils
import app.streamlit.utils.visualization as plot_utils


def render():
    """
    Render the UI for the /transform endpoint (Model Inference).
    """
    st.subheader("🚀 Project New Data")
    st.markdown("""
    Use a previously generated **Access Key** to project new data points into
    your existing UMAP latent space.
    """)

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
        df_new, new_target_column = data_utils.fetch_csv_file(new_data_file)

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
        if new_target_column is not None:
            df_to_transform = df_new.drop(columns=[new_target_column])
        else:
            df_to_transform = df_new.copy()

        with st.spinner("Contacting UMAP API for transformation..."):
            new_emb = emb_utils.run_umap_transform(
                df=df_to_transform,
                access_key=access_key,
                n_epochs=n_epochs_trans
            )

        # Store results in session state
        st.session_state["new_embedding"] = new_emb

        st.success(f"Successfully transformed {len(df_new)} points.")

    if "new_embedding" in st.session_state:
        new_emb = st.session_state["new_embedding"]

        # --- 5. Visualization ---
        st.write("### Projection Results")

        plot_utils.show_embeddings(
            embedding=new_emb,
            data_to_embed=df_new,
            target_column=new_target_column
        )

        # Download result
        new_csv_data = pd.DataFrame(
            new_emb,
            columns=[f"dim_{i}" for i in range(new_emb.shape[1])]
        ).to_csv(index=False).encode()

        st.download_button(
            "Download Projected Points",
            new_csv_data,
            "projected_data.csv",
            "text/csv"
        )

        # -----------------------------
        # 6. Clustering
        # -----------------------------
        st.header("Clustering")

        new_clustering_method = param_utils.select_clustering_method(key_suffix="_transform")
        new_clustering_param = param_utils.select_clustering_param(
            clustering_method=new_clustering_method,
            n_samples=len(new_emb),
            key_suffix="_transform"
        )

        if st.button("Run Clustering", key="new_run_clustering"):
            with st.spinner("Clustering..."):
                if new_clustering_method == "KMeans":
                    new_labels = emb_utils.run_kmeans(new_emb, new_clustering_param)
                else:
                    new_labels = emb_utils.run_hdbscan(new_emb, new_clustering_param)

            st.session_state["new_labels"] = new_labels
            st.success("Clustering completed")

    if "new_embedding" in st.session_state and "new_labels" in st.session_state:
        new_emb = st.session_state["new_embedding"]
        new_labels = st.session_state["new_labels"]

        # -----------------------------
        # 7. Metrics
        # -----------------------------
        if len(set(new_labels)) >= 2:
            sil_score = silhouette_score(new_emb, new_labels)
            st.metric(label="Silhouette Score", value=f"{sil_score:.4f}")

        # -----------------------------
        # 8. Visualization with clusters
        # -----------------------------
        st.header("Clustered Visualization")

        plot_utils.show_clusters(embedding=new_emb, labels=new_labels)

        result_df = pd.DataFrame(new_emb, columns=[f"dim_{i}" for i in range(new_emb.shape[1])])
        result_df["cluster"] = new_labels

        csv = result_df.to_csv(index=False).encode()
        st.download_button(
            "Download new embedding + clusters",
            csv,
            "embedding_clusters.csv", "text/csv"
        )
