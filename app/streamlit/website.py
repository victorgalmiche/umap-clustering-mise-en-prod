import streamlit as st

from modules import exploration, transform

st.set_page_config(
    page_title="UMAP + Clustering App",
    layout="wide",
    menu_items={"About": "Version 0.1"}
)

st.title("UMAP Dimensionality Reduction & Clustering")

# On crée 3 onglets pour garder l'ancien et ajouter le nouveau
tab_exp, tab_trans = st.tabs([
    "🔍 Exploration (/umap)",
    "🚀 Projection (/transform)"
])

with tab_exp:
    # On passe les dataframes nécessaires au module
    exploration.render()

with tab_trans:
    # Ici on n'a pas forcément besoin de data_to_embed car on upload 
    # de nouveaux points, mais on peut passer des infos de contexte.
    transform.render()
