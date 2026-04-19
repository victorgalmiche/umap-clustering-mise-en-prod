"""
UMAP Service API utils 
"""
import os
import io
import polars as pl
import logging
import umap

from pathlib import Path
from pydantic import BaseModel
from fastapi import UploadFile, HTTPException, Form
from typing import Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker

logger = logging.getLogger(Path(__file__).stem)


async def validate_and_read_csv(file: UploadFile) -> Tuple[pl.DataFrame, bytes]:
    """
    Validate that an uploaded file is a CSV and read its content.
 
    Parameters
    ----------
    file : UploadFile
        The file uploaded via the FastAPI endpoint.
 
    Returns
    -------
    Tuple[pl.DataFrame, bytes]
        Contains:
        - The Polars DataFrame built from the CSV content
        - The raw bytes content of the file (useful for size monitoring)
 
    Raises
    ------
    HTTPException
        400 if the file is not a CSV or exceeds the line limit.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    return _get_polars_from_request(content), content


def _get_polars_from_request(content: bytes) -> pl.DataFrame:
    """
    Convert raw CSV bytes from a POST request into a Polars DataFrame.
    keeping numerical columns
 
    Enforces a hard limit on the number of rows to cap compute resources.
 
    Parameters
    ----------
    content : bytes
        Raw CSV file content read from the HTTP request.
 
    Returns
    -------
    pl.DataFrame
        Polars DataFrame parsed from the CSV content.
 
    Raises
    ------
    HTTPException
        400 if the CSV contains 500 lines or more.
    """

    df = pl.read_csv(io.BytesIO(content)).select(pl.selectors.numeric())

    if df.height >= 500:
        raise HTTPException(status_code=400, detail="CSV file must have less than 500 lines.")

    if df.width < 3:
        raise HTTPException(
            status_code=400,
            detail="CSV file must have at least 3 numerical columns."
        )
    return df


def get_experiment_path(base_name: str, client_source: Optional[str] = None) -> str:
    """
    Generate the MLflow experiment path based on the environment and client source.
 
    Defaults to the APP_ENV environment variable, or 'dev' if not set.
 
    Parameters
    ----------
    base_name : str
        Base name of the MLflow experiment (e.g. 'umap-training').
    client_source : str, optional
        Identifier of the calling client. If provided, it overrides APP_ENV.
 
    Returns
    -------
    str
        MLflow experiment path formatted as '/{env}/{base_name}'.
    """
    env = client_source if client_source else os.getenv("APP_ENV", "dev")
    return f"/{env}/{base_name}"


def fit_umap_model(
    df: pl.DataFrame, 
    params: dict, 
    tracker: ExperimentTracker
) -> Tuple[Any, StandardScaler, Any, Any]:
    """
    Train a UMAP model using a dictionary of parameters.
 
    Tries the custom `umap_mapping` implementation first, and falls back to
    the `umap-learn` library if the custom training fails.
 
    Parameters
    ----------
    df : pl.DataFrame
        Input dataset to fit the UMAP model on.
    params : dict
        Dictionary of UMAP parameters, including:
            n_neighbors : int
                Number of neighbors for KNN.
            n_components : int
                Output embedding dimension.
            min_dist : float
                Minimum distance in low-dimensional space.
            knn_metric : str
                Distance metric: 'euclidean', 'manhattan', etc.
            knn_method : str
                KNN method: 'exact' or 'approx'.
            n_epochs : int
                Optimization epochs.
    tracker : ExperimentTracker
        MLflow tracker used to log training metrics.
 
    Returns
    -------
    Tuple[Any, StandardScaler, Any, Any]
        Contains:
        - model: The fitted UMAP model (custom or umap-learn fallback)
        - scaler: The fitted StandardScaler
        - dataset_standardized: The standardized input dataset
        - Y: The low-dimensional embedding of the training data
    """
    scaler = StandardScaler()
    dataset_standardized = scaler.fit_transform(df.to_pandas())

    model = umap_mapping(
        n_neighbors=params.get("n_neighbors"),
        n_components=params.get("n_components"),
        min_dist=params.get("min_dist"),
        KNN_metric=params.get("knn_metric"),
        KNN_method=params.get("knn_method"),
    )

    try:
        result = model.fit_transform(X=dataset_standardized, n_epochs=params.get("n_epochs"))
        Y = result[0] if isinstance(result, tuple) else result
        tracker.log_metrics({"training_success": 1})
    except Exception as e:
        logger.warning(f"Custom UMAP failed: {e}. Falling back to umap-learn.")
        model = umap.UMAP(
            n_neighbors=params.get("n_neighbors"),
            n_components=params.get("n_components"),
            min_dist=params.get("min_dist"),
            metric=params.get("knn_metric")
        )
        Y = model.fit_transform(dataset_standardized)
        tracker.log_metrics({"training_success": 0, "fallback": 1})

    return model, scaler, dataset_standardized, Y


class UmapParameters(BaseModel):
    n_neighbors: int
    n_components: int
    min_dist: float
    knn_metric: str
    knn_method: str
    n_epochs: int

    def get_umap_params(cfg: Any) -> Any:
        """
        Build a FastAPI dependency that injects Hydra config defaults into form fields.
    
        Returns a callable to be used with `Depends()` so that each form field
        defaults to the value from `cfg.umap` while remaining user-overridable.
    
        Parameters
        ----------
        cfg : Any
            Hydra configuration object exposing `cfg.umap` with UMAP defaults.
    
        Returns
        -------
        Callable[..., UmapParameters]
            A dependency function that returns a `UmapParameters` instance built
            from the request form fields.
        """        
        def dependency(
            n_neighbors: int = Form(cfg.umap.n_neighbors, description="Number of neighbors for KNN"),
            n_components: int = Form(cfg.umap.n_components, description="Target dimension"),
            min_dist: float = Form(cfg.umap.min_dist, description="Minimum distance in the embedding"),
            knn_metric: str = Form(cfg.umap.KNN_metric, description="Distance metric"),
            knn_method: str = Form(cfg.umap.KNN_method, description="KNN search method: 'exact' or 'approx'"),
            n_epochs: int = Form(cfg.umap.n_epochs_train, description="Optimization iterations"),
        ) -> UmapParameters:
            return UmapParameters(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                knn_metric=knn_metric,
                knn_method=knn_method,
                n_epochs=n_epochs
            )
        return dependency


def prepare_umap_params(df: pl.DataFrame, params: UmapParameters) -> tuple[dict, int, int]:
    """
    Merge UMAP parameters with the DataFrame dimensions.
 
    Parameters
    ----------
    df : pl.DataFrame
        Input dataset used to extract the number of samples and features.
    params : UmapParameters
        UMAP parameters received from the request.
 
    Returns
    -------
    Tuple[dict, int, int]
        Contains:
        - umap_params: The full parameters dictionary, including `n_samples` and `n_features`
        - n_samples: Number of rows in the DataFrame
        - n_features: Number of columns in the DataFrame
    """

    umap_params = params.model_dump()
    
    n_samples, n_features = df.shape[0], df.shape[1]
    
    umap_params.update({
        "n_samples": n_samples,
        "n_features": n_features
    })
    
    return umap_params, n_samples, n_features