import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def show_embeddings(
    embedding: np.ndarray,
    data_to_embed: pd.DataFrame,
    target_column: str | None,
) -> None:
    """
    Display a 2D scatter plot of embeddings, optionally colored by a target column.

    Renders the plot directly in the Streamlit app. If the embedding is not
    2-dimensional, a warning is shown.

    Parameters
    ----------
    embedding : 2D array of shape (n_samples, n_components)
    data_to_embed : Original DataFrame the embedding was computed from.
    target_column : Name of the column in ``data_to_embed`` used to color the points.
        If None, all points are drawn with a single color.
    """

    shape = embedding.shape[1]
    if shape == 2:
        plot_df = pd.DataFrame(embedding, columns=["x", "y"])
        if target_column:
            plot_df[target_column] = data_to_embed[target_column].astype(str)
        fig = px.scatter(plot_df, x="x", y="y", color=target_column)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning(f"Visualization is only supported in 2D (current dimensions : {shape}).")


def show_clusters(embedding: np.ndarray, labels: np.ndarray):
    """
    Display a 2D scatter plot of embeddings colored by cluster label.

    Renders the plot directly in the Streamlit app. If the embedding is not
    2-dimensional, a warning is shown.

    Parameters
    ----------
    embedding : 2D array of shape (n_samples, n_components) containing the embedded
        coordinates. Only n_components == 2 is supported for plotting.
    labels : 1D array of shape (n_samples,) with the cluster label of each point.
    """

    shape = embedding.shape[1]
    if shape == 2:
        plot_df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "cluster": labels.astype(str)})
        fig = px.scatter(plot_df, x="x", y="y", color="cluster")
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning(f"Visualization is only supported in 2D (current dimensions : {shape}).")
