import streamlit as st
import requests
from modules import exploration, transform

# --- CONFIGURATION ---
API_URL = "https://umap-api-mmvs.lab.sspcloud.fr/"

st.set_page_config(
    page_title="UMAP + Clustering App",
    layout="wide",
    menu_items={"About": "Version 0.1"}
)


# --- API STATUS SIDEBAR ---
def check_api_status():
    try:
        # Use a timeout so the UI doesn't hang if the API is down
        # API calls mlflow and this can take ~2s, use a 4s timeout
        response = requests.get(API_URL, timeout=4)
        if response.status_code == 200:
            return "online"
        return "offline"
    except Exception:
        return "offline"


with st.sidebar:
    st.header("System Status")
    status = check_api_status()

    if status == "online":
        st.success("API Connected", icon="🟢")
    else:
        st.error("API Offline", icon="🔴")


# --- MAIN INTERFACE ---
st.title("UMAP Dimensionality Reduction & Clustering")
st.markdown("""
        Use this tool to project high-dimensional data into 2D or 3D space and identify natural groupings using UMAP and unsupervised clustering.
    """)

tab_exp, tab_trans = st.tabs([
    "🔍 Exploration (/umap) and training (/train)",
    "🚀 Projection (/transform) - Experimental"
])

with tab_exp:
    exploration.render()

with tab_trans:
    transform.render()
