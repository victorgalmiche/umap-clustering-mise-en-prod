import os
import io
import polars as pl
import logging
import Path
import umap

from pydantic import BaseModel
from fastapi import UploadFile, HTTPException, Form
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

from src.umap_algo.umap_class import umap_mapping


logger = logging.getLogger(Path(__file__).stem)


async def validate_and_read_csv(file: UploadFile) -> Tuple[pl.DataFrame, bytes]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.file.read()
    return _get_polars_from_request(content), content


def _get_polars_from_request(content: bytes) -> pl.DataFrame:
    """
    Get CSV data from the POST request
    Convert to a Polars dataframe
    raise exception if more than 500 lines (reason: limit compute ressources)
    """

    df = pl.read_csv(io.BytesIO(content))

    if df.height >= 500:
        raise HTTPException(status_code=400, detail="CSV file must have less than 500 lines.")

    return df


def get_experiment_path(base_name: str, client_source: Optional[str] = None) -> str:
    """
    Generates the MLflow experiment path based on the environment and client source.
    Defaults to the APP_ENV variable or 'dev'.
    """
    env = client_source if client_source else os.getenv("APP_ENV", "dev")
    return f"/{env}/{base_name}"


def fit_umap_model(df, params, tracker):
    """
    Entraîne le modèle UMAP en utilisant un dictionnaire de paramètres.
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
        Y = model.fit_transform(X=dataset_standardized, n_epochs=params.get("n_epochs"))
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
        tracker.log_metrics({"training_success": 0})
        tracker.log_params({"fallback": True})

    return model, scaler, dataset_standardized, Y


class UmapParameters(BaseModel):
    n_neighbors: int
    n_components: int
    min_dist: float
    knn_metric: str
    knn_method: str
    n_epochs: int

    def get_umap_params(cfg):
        """
        Générateur de dépendance pour injecter cfg dans les formulaires FastAPI.
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


def prepare_umap_params(df, params: UmapParameters) -> tuple[dict, int, int]:
    """
    Fusionne les paramètres UMAP avec les dimensions du DataFrame.
    Retourne le dictionnaire complet, n_samples et n_features.
    """
    umap_params = params.model_dump()
    
    n_samples, n_features = df.shape[0], df.shape[1]
    
    umap_params.update({
        "n_samples": n_samples,
        "n_features": n_features
    })
    
    return umap_params, n_samples, n_features