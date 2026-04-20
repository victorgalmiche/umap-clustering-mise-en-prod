from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd

DATASET_OPTIONS = {
    "Digits (Images of handwritten digits)": load_digits,
    "Iris (Flower specifications)": load_iris,
    "Wine (specifications)": load_wine,
    "Breast Cancer (specifications)": load_breast_cancer,
}


def reset_state():
    if "embedding" in st.session_state:
        del st.session_state["embedding"]
    if "labels" in st.session_state:
        del st.session_state["labels"]


@st.cache_data
def load_and_sample(df):
    """Limit to 500 lines, deterministic"""
    return df.sample(n=min(499, len(df)), random_state=42).reset_index(drop=True)


def fetch_data_source():
    return st.sidebar.radio("Data Selection", ["Standard Datasets", "Upload CSV"], on_change=reset_state)


def fetch_csv_file(
    data_file: UploadedFile | None,
    suffix_key: str,
) -> tuple[pd.DataFrame, str | None]:

    if not suffix_key:
        raise ValueError("key_suffix is required to avoid duplicate Streamlit element keys.")
    if data_file is not None:
        raw_df = pd.read_csv(data_file)

        # Limit the size for fast run of UMAP
        if len(raw_df) >= 500:
            st.warning("Too many rows (>= 500), data will be sampled.")
            data_to_embed = load_and_sample(raw_df)
        else:
            data_to_embed = raw_df

        # The user choses target column
        target_column = st.selectbox(
            "Select the target column",
            options=[None] + list(data_to_embed.columns),
            help="This column will be ignored by UMAP but used for visualization.",
            key="target_column" + suffix_key,
        )
    else:
        st.sidebar.warning("Waiting for file...")
        st.stop()

    return data_to_embed, target_column


def fetch_default_data():
    selected_name = st.sidebar.selectbox("Chose a dataset", list(DATASET_OPTIONS.keys()), on_change=reset_state)
    loader = DATASET_OPTIONS[selected_name]
    data = loader(as_frame=True)

    df_full = data["data"].copy()
    df_full["target"] = data["target"]

    data_to_embed = load_and_sample(df_full)
    target_column = "target"

    return data_to_embed, target_column
