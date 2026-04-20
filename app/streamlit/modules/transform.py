import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score

import app.streamlit.utils.embeddings as emb_utils
import app.streamlit.utils.data_preprocessing as data_utils
import app.streamlit.utils.hyperparameters as param_utils
import app.streamlit.utils.visualization as plot_utils


def render()->None:
    """
    Render the UI for the /transform endpoint (Model Inference).
    """
    st.markdown("""
    Apply a saved UMAP manifold to **unseen data**. This ensures that new samples are 
    positioned consistently relative to your original training set.
    """)

    # --- 1. Security & Key ---
    st.subheader("🔑 Authentication")
    saved_key = st.session_state.get("access_key", "")
    access_key = st.text_input(
        "Access Key",
        value=saved_key,
        type="password",
        help="The unique ID generated when you saved your model in the Train tab."
    )
    if not access_key:
        st.warning("Please enter an Access Key to proceed.")

    # --- 2. New Data Upload ---
    st.subheader("1. Upload New Samples")
    st.info("💡 **Requirement:** Your CSV must contain the same numerical features (columns and column names) as your training data.")

    new_data_file = st.file_uploader(
        "Select CSV file",
        type=["csv"],
        help="Upload the new data points you wish to project."
    )

    if new_data_file is not None:
        df_new, new_target_column = data_utils.fetch_csv_file(new_data_file, suffix_key="_transform")
        with st.expander("Preview new data"):
            st.dataframe(df_new.head())
    else:
        st.info("Waiting for a CSV file to be uploaded...")
        st.stop()

    # --- 3. Hyperparameters ---
    st.subheader("2. Projection Settings")
    n_epochs_trans = st.slider(
        "Optimization Epochs",
        min_value=1,
        max_value=200,
        value=100,
        help="How many iterations to run to find the best placement for new points. More epochs = higher precision but slower."
    )

    # --- 4. Execution & Results ---
    if st.button("✨ Apply Transformation", type="primary", use_container_width=True):
        if not access_key:
            st.error("Access Key is required to fetch the stored model.")
            return

        # Data preparation
        if new_target_column is not None:
            df_to_transform = df_new.drop(columns=[new_target_column])
        else:
            df_to_transform = df_new.copy()

        with st.spinner("🔄 Projecting points into latent space..."):
            try:
                new_emb = emb_utils.run_umap_transform(
                    df=df_to_transform,
                    access_key=access_key,
                    n_epochs=n_epochs_trans
                )
                st.session_state["new_embedding"] = new_emb
                st.success(f"Successfully transformed {len(df_new)} points.")
            except Exception as e:
                st.error(f"Transformation failed. Ensure your key is correct and data columns match. Error: {e}")

    if "new_embedding" in st.session_state:
        new_emb = st.session_state["new_embedding"]
        
        st.divider()

        # --- 5. Visualization ---
        st.subheader("3. Projection Results")
        
        col_viz, col_dl = st.columns([3, 1])
        
        with col_viz:
            plot_utils.show_embeddings(
                embedding=new_emb,
                data_to_embed=df_new,
                target_column=new_target_column
            )

        with col_dl:
            st.write("#### Export")
            new_csv_data = pd.DataFrame(
                new_emb,
                columns=[f"dim_{i}" for i in range(new_emb.shape[1])]
            ).to_csv(index=False).encode()

            st.download_button(
                "📥 Download CSV",
                new_csv_data,
                "projected_data.csv",
                "text/csv",
                use_container_width=True
            )

        # -----------------------------
        # 6. Clustering
        # -----------------------------
        st.divider()
        st.header("4. Post-Projection Clustering")
        st.write("Identify groups within the newly projected data.")

        c1, c2 = st.columns(2)
        with c1:
            new_clustering_method = param_utils.select_clustering_method(key_suffix="_transform")
        with c2:
            new_clustering_param = param_utils.select_clustering_param(
                clustering_method=new_clustering_method,
                n_samples=len(new_emb),
                key_suffix="_transform"
            )

        if st.button("🪄 Run Clustering", key="new_run_clustering", use_container_width=True):
            with st.spinner("Finding patterns..."):
                if new_clustering_method == "KMeans":
                    new_labels = emb_utils.run_kmeans(new_emb, new_clustering_param)
                else:
                    new_labels = emb_utils.run_hdbscan(new_emb, new_clustering_param)

            st.session_state["new_labels"] = new_labels

    if "new_embedding" in st.session_state and "new_labels" in st.session_state:
        new_emb = st.session_state["new_embedding"]
        new_labels = st.session_state["new_labels"]

        # -----------------------------
        # 7. Metrics & Final Viz
        # -----------------------------
        if len(set(new_labels)) >= 2:
            sil_score = silhouette_score(new_emb, new_labels)
            st.metric(
                label="Cluster Quality (Silhouette Score)", 
                value=f"{sil_score:.4f}",
                help="Measures how similar an object is to its own cluster compared to other clusters."
            )

        plot_utils.show_clusters(embedding=new_emb, labels=new_labels)

        result_df = pd.DataFrame(new_emb, columns=[f"dim_{i}" for i in range(new_emb.shape[1])])
        result_df["cluster"] = new_labels

        final_csv = result_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Projected Clusters",
            final_csv,
            "projected_clusters.csv", 
            "text/csv",
            use_container_width=True
        )
