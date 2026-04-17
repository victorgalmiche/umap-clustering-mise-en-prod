import pandas as pd
import streamlit as st
import plotly.express as px


def show_embeddings(embedding, data_to_embed, target_column):
    if embedding.shape[1] == 2:
        plot_df = pd.DataFrame(embedding, columns=["x", "y"])
        if target_column:
            plot_df[target_column] = data_to_embed[target_column].astype(str)
        fig = px.scatter(plot_df, x="x", y="y", color=target_column)
        st.plotly_chart(fig, width='stretch')


def show_clusters(embedding, labels):
    if embedding.shape[1] == 2:
        plot_df = pd.DataFrame({
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "cluster": labels.astype(str)
        })
        fig = px.scatter(plot_df, x="x", y="y", color="cluster")
        st.plotly_chart(fig, width='stretch')
