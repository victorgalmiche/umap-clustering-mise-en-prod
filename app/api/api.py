"""
UMAP Service API with Hydra Configuration
"""

import io
import os
import secrets
import logging
from typing import Optional
from pathlib import Path

import polars as pl
import umap
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header
from sklearn.preprocessing import StandardScaler
import hydra

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker, UmapStorage

# Initialize Hydra globally
hydra.initialize(version_base=None, config_path="../../config")
cfg = hydra.compose(config_name="main")

logger = logging.getLogger(Path(__file__).stem)

# In-memory model cache: access_key -> (model, scaler, dataset_standardized, Y)
model_cache = {}

# API Metadata for Swagger UI (/docs)
tags_metadata = [
    {"name": "General", "description": "System health and welcome information."},
    {"name": "Model Management",
     "description": "Training and projection operations using secure access keys."},
    {"name": "Legacy", "description": "One-shot UMAP projections without persistence."},
]

app = FastAPI(
    title="UMAP Management API",
    description="""
    This API provides high-performance dimension reduction services.

    ### Workflow:
    1. **Train** a model by uploading a CSV. Receive a secure `access_key`.
    2. **Transform** new data using the manifold learned during training via your `access_key`.
    3. **Track** results automatically in MLflow.
    """,
    version="0.2.0",
    openapi_tags=tags_metadata
)


def get_experiment_path(base_name: str, client_source: Optional[str] = None) -> str:
    """
    Generates the MLflow experiment path based on the environment and client source.
    Defaults to the APP_ENV variable or 'dev'.
    """
    env = client_source if client_source else os.getenv("APP_ENV", "dev")
    return f"/{env}/{base_name}"


@app.get("/", tags=["General"], summary="Welcome endpoint")
def show_welcome_page():
    """Returns basic API metadata."""
    return {
        "api": "UMAP API",
        "version": cfg.api.version,
        "status": "ready"
    }


@app.post(
    "/train",
    summary="Train a UMAP model",
    tags=["Model Management"]
)
async def train_model(
    file: UploadFile = File(...),
    n_neighbors: int = Form(cfg.umap.n_neighbors,
                            description="Number of neighbors for KNN"),
    n_components: int = Form(cfg.umap.n_components,
                             description="Target dimension"),
    min_dist: float = Form(cfg.umap.min_dist,
                           description="Minimum distance in the embedding"),
    knn_metric: str = Form(cfg.umap.KNN_metric,
                           description="Distance metric"),
    knn_method: str = Form(cfg.umap.KNN_method,
                           description="KNN search method: 'exact' or 'approx'"),
    n_epochs: int = Form(cfg.umap.n_epochs_train,
                         description="Optimization iterations"),
    x_client_source: Optional[str] = Header(None)
):
    """
    Train a UMAP model on provided CSV and return a secure access key.

    The access key is a random token that grants access to transform data
    without exposing the underlying model or training data to other users.

    Parameters
    ----------
    file : UploadFile
        CSV file for training
    n_neighbors : int
        Number of neighbors for KNN (default: 15)
    n_components : int
        Output embedding dimension (default: 2)
    min_dist : float
        Minimum distance in low-dimensional space (default: 0.1)
    knn_metric : str
        Distance metric: 'euclidean', 'manhattan', etc. (default: 'euclidean')
    knn_method : str
        KNN method: 'exact' or 'approx' (default: 'approx')
    n_epochs : int
        Optimization epochs (default: 200)
    x_client_source : str (optional)
        To identify the caller for monitoring purposes

    Returns
    -------
    dict
        Contains:
        - access_key: Secure random token for /transform
        - embedding_shape: Shape of training embedding
        - n_samples: Number of training samples
        - message: Usage instructions
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Data ingestion
    content = await file.read()
    df = pl.read_csv(io.BytesIO(content))
    n_samples, n_features = df.shape

    # MLflow setup
    exp_path = get_experiment_path("umap-training", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path,
        run_name=f"train-{file.filename}",
        run_tags={"env": os.getenv("APP_ENV", "dev"), "source": x_client_source or "api"}
    )

    with tracker.run():
        tracker.log_params({
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "knn_metric": knn_metric,
            "knn_method": knn_method,
            "n_epochs": n_epochs,
            "n_samples": n_samples,
            "n_features": n_features,
        })

        # Preprocessing
        scaler = StandardScaler()
        dataset_standardized = scaler.fit_transform(df.to_pandas())

        # Training logic
        model = umap_mapping(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            KNN_metric=knn_metric,
            KNN_method=knn_method,
        )

        try:
            Y = model.fit_transform(dataset_standardized, n_epochs=n_epochs)
            tracker.log_metrics({"training_success": 1})
        except Exception as e:
            logger.warning(f"Custom UMAP failed: {e}. Falling back to umap-learn.")
            model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric=knn_metric
            )
            Y = model.fit_transform(dataset_standardized)
            tracker.log_metrics({"training_success": 0})
            tracker.log_params({"fallback": True})

        # MLflow persistence
        tracker.log_pyfunc_model(
            pyfunc_model=UmapStorage(model),
            artifact_path="umap_model",
            registered_model_name="umap_model_registry",
            X_train=dataset_standardized,
            Y_train=Y
        )

        tracker.log_metrics({
            "output_shape_0": Y.shape[0],
            "output_shape_1": Y.shape[1],
        })

        # Caching and access control
        access_key = secrets.token_urlsafe(32)
        model_cache[access_key] = (model, scaler, dataset_standardized, Y)

    return {
        "access_key": access_key,
        "embedding_shape": Y.shape,
        "n_samples": n_samples,
        "n_features": n_features,
        "message": "Model cached. Use the access_key for the /transform endpoint."
    }


@app.post(
    "/transform",
    summary="Transform new data",
    tags=["Model Management"]
)
async def transform_data(
    access_key: str = Form(..., description="The key provided by the /train endpoint"),
    file: UploadFile = File(...),
    n_epochs: int = Form(cfg.umap.n_epochs_transform,
                         description="Optimization epochs for the projection"),
    x_client_source: Optional[str] = Header(None)
):
    """
    Transform new data using a previously trained UMAP model.

    Uses the secure access key from /train endpoint to identify the model.
    Only the person with the access key can transform data with that model.

    Parameters
    ----------
    access_key : str
        Secure token received from /train endpoint
    file : UploadFile
        CSV file with new data
    n_epochs : int
        Optimization epochs for refining new embeddings (default: 100)
    x_client_source : str (optional)
        To identify the caller for monitoring purposes

    Returns
    -------
    dict
        Contains embedding and metadata
    """
    if access_key not in model_cache:
        raise HTTPException(status_code=403, detail="Invalid access_key.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Retrieve objects from cache
    model, scaler, _, _ = model_cache[access_key]

    # Processing
    content = await file.read()
    df = pl.read_csv(io.BytesIO(content))
    X_new_scaled = scaler.transform(df.to_pandas())

    exp_path = get_experiment_path("umap-transform", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path,
        run_name="transform-execution",
        run_tags={"env": os.getenv("APP_ENV", "dev")}
    )

    with tracker.run():
        tracker.log_params({"n_epochs": n_epochs, "n_samples": df.shape[0]})
        try:
            Y_new = model.transform(X_new_scaled, n_epochs=n_epochs)
            tracker.log_metrics({"transform_success": 1})
        except Exception as e:
            tracker.log_metrics({"transform_success": 0})
            raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

        tracker.log_metrics({
            "output_shape_0": Y_new.shape[0],
            "output_shape_1": Y_new.shape[1],
        })

    return {
        "embedding": Y_new.tolist(),
        "embedding_shape": Y_new.shape,
        "n_samples": df.shape[0],
    }


@app.post(
    "/umap",
    summary="One-shot UMAP projection",
    tags=["Legacy"]
)
async def apply_umap(
    file: UploadFile = File(...),
    n_neighbors: int = Form(cfg.umap.n_neighbors,
                            description="Number of neighbors for KNN"),
    n_components: int = Form(cfg.umap.n_components,
                             description="Target dimension"),
    min_dist: float = Form(cfg.umap.min_dist,
                           description="Minimum distance in the embedding"),
    knn_metric: str = Form(cfg.umap.KNN_metric,
                           description="Distance metric"),
    knn_method: str = Form(cfg.umap.KNN_method,
                           description="KNN search method: 'exact' or 'approx'"),
    n_epochs: int = Form(cfg.umap.n_epochs_train,
                         description="Optimization iterations"),
    x_client_source: Optional[str] = Header(None)
):
    """
    Accept a CSV file via multipart/form-data and return the UMAP projection.

    **Legacy endpoint** - For new usage, use /train and /transform instead for better performance
    and privacy.

    Parameters
    ----------
    file : UploadFile
        CSV file corresponding to the data to fit-transform
    n_neighbors : int
        Number of neighbors for KNN (default: 15)
    n_components : int
        Output embedding dimension (default: 2)
    min_dist : float
        Minimum distance in low-dimensional space (default: 0.1)
    knn_metric : str
        Distance metric: 'euclidean', 'manhattan', etc. (default: 'euclidean')
    knn_method : str
        KNN method: 'exact' or 'approx' (default: 'approx')
    n_epochs : int
        Optimization epochs (default: 200)
    x_client_source : str (optional)
        To identify the caller for monitoring purposes

    Returns
    -------
    dict
        JSON object with embedding
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    df = pl.read_csv(io.BytesIO(content))

    exp_path = get_experiment_path("umap-legacy", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path,
        run_name=f"legacy-run-{file.filename}",
        run_tags={"mode": "legacy", "env": os.getenv("APP_ENV", "dev")}
    )

    with tracker.run():
        tracker.log_params({"n_neighbors": n_neighbors, "n_samples": df.shape[0]})

        scaler = StandardScaler()
        dataset_standardized = scaler.fit_transform(df.to_pandas())

        model = umap_mapping(
            n_neighbors=n_neighbors, n_components=n_components,
            min_dist=min_dist, KNN_metric=knn_metric, KNN_method=knn_method,
        )

        try:
            result = model.fit_transform(dataset_standardized)
            dataset_transformed = result[0] if isinstance(result, tuple) else result
            tracker.log_metrics({"success": 1})
        except Exception as e:
            logger.warning(f"Custom fallback: {e}")
            model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist)
            dataset_transformed = model.fit_transform(dataset_standardized)
            tracker.log_metrics({"success": 0, "fallback": 1})

        tracker.log_pyfunc_model(
            pyfunc_model=UmapStorage(model),
            artifact_path="legacy_model",
            registered_model_name="umap_legacy_models",
            X_train=dataset_standardized,
            Y_train=dataset_transformed
        )

    return {"embedding": dataset_transformed.tolist()}
