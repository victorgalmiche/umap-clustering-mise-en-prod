"""
UMAP Service API with Hydra Configuration
"""

import os
import secrets
import logging
import time
import hydra
import umap
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Request, Depends

from app.api.modules.api_utils import (
    validate_and_read_csv, 
    get_experiment_path, 
    fit_umap_model, 
    UmapParameters, 
    prepare_umap_params,
)

from src.adapter.mlflow_tracker import ExperimentTracker, UmapStorage
from src.adapter.monitoring import get_monitor

# Initialize Hydra globally
hydra.initialize(version_base=None, config_path="../../config")
cfg = hydra.compose(config_name="main")
umap_parameters = UmapParameters.get_umap_params(cfg)

logger = logging.getLogger(Path(__file__).stem)

# In-memory model cache: access_key -> (model, scaler, dataset_standardized, Y)
model_cache = {}

# API Metadata for Swagger UI (/docs)
tags_metadata = [
    {"name": "General", "description": "System health and welcome information."},
    {"name": "Model Management",
     "description": "Training and projection operations using secure access keys (experimental)."},
    {"name": "Production", "description": "One-shot UMAP projections without persistence."},
]

app = FastAPI(
    title="UMAP Management API",
    description="""
    This API provides dimension reduction services.

    ### Usage :
    - `/umap`: upload a CSV. Receive low-dimensional embeddings

    ### In development
    1. **Train** a model by uploading a CSV. Receive a secure `access_key`.
    2. **Transform** new data using the manifold learned during training via your `access_key`.
    3. **Track** results automatically in MLflow.
    Service is not guaranteed.
    """,
    version="0.5.1",
    openapi_tags=tags_metadata
)

# Initialize monitoring
monitor = get_monitor()


@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware to track request latency and errors."""
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        monitor.log_request(endpoint, method, response.status_code, latency_ms)
        return response
    except Exception:
        monitor.log_request_error(endpoint, method)
        raise


@app.get("/", tags=["General"], summary="Welcome endpoint")
def show_welcome_page():
    """Returns basic API metadata."""
    return {"api": "UMAP API", "version": cfg.api.version, "status": "ready"}


@app.get("/health", tags=["General"], summary="Health check endpoint")
def health_check():
    """
    Returns API health status and monitoring information.

    Used by orchestration systems (Kubernetes, Docker, etc.) to verify service availability.
    """
    return {
        "status": "healthy",
        "version": cfg.api.version,
        "cached_models": len(model_cache),
        "environment": os.getenv("APP_ENV", "dev"),
    }


@app.post("/train", summary="Train a UMAP model", tags=["Model Management"])
async def train_model(
    file: UploadFile = File(...),
    params: UmapParameters = Depends(umap_parameters),
    x_client_source: Optional[str] = Header(None),
):
    """
    Train a UMAP model on provided CSV and return a secure access key.

    The access key is a random token that grants access to transform data
    without exposing the underlying model or training data to other users.

    Parameters
    ----------
    file : UploadFile
        CSV file for training
    params : UmapParameters
        including :
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

    df, content = await validate_and_read_csv(file=file)

    umap_params, n_samples, n_features = prepare_umap_params(df=df, params=params)

    # Log input size metrics
    monitor.log_input_size("/train", len(content), n_samples, n_features)

    # MLflow setup
    tracker = ExperimentTracker(
        experiment_name=get_experiment_path("umap-training", x_client_source),
        run_name=f"train-{file.filename}",
        run_tags={"env": os.getenv("APP_ENV", "dev"), "source": x_client_source or "api"},
    )

    with tracker.run():
        tracker.log_params(umap_params)

        model, scaler, dataset_standardized, Y = fit_umap_model(df, umap_params, tracker)

        tracker.log_pyfunc_model(
            pyfunc_model=UmapStorage(model),
            artifact_path="umap_model",
            registered_model_name="umap_model_registry",
            X_train=dataset_standardized,
            Y_train=Y,
        )

        tracker.log_metrics(
            {
                "output_shape_0": Y.shape[0],
                "output_shape_1": Y.shape[1],
            }
        )

    access_key = secrets.token_urlsafe(32)
    model_cache[access_key] = (model, scaler, dataset_standardized, Y)

    # Log cache status outside the training run to avoid nested MLflow runs.
    monitor.log_cache_status(len(model_cache))

    return {
        "access_key": access_key,
        "embedding_shape": Y.shape,
        "n_samples": n_samples,
        "n_features": n_features,
        "message": "Model cached. Use the access_key for the /transform endpoint.",
        "embedding": Y.tolist(),
    }


@app.post("/transform", summary="Transform new data", tags=["Model Management"])
async def transform_data(
    access_key: str = Form(..., description="The key provided by the /train endpoint"),
    file: UploadFile = File(...),
    n_epochs: int = Form(cfg.umap.n_epochs_transform, description="Optimization epochs for the projection"),
    x_client_source: Optional[str] = Header(None),
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
        monitor.log_error("/transform", "invalid_access_key")
        raise HTTPException(status_code=403, detail="Invalid access_key.")

    model, scaler, _, _ = model_cache[access_key]

    df, content = await validate_and_read_csv(file=file)

    monitor.log_input_size("/transform", len(content), df.shape[0], df.shape[1])
    
    X_new_scaled = scaler.transform(df.to_pandas())

    tracker = ExperimentTracker(
        experiment_name=get_experiment_path("umap-transform", x_client_source), 
        run_name="transform-execution", 
        run_tags={"env": os.getenv("APP_ENV", "dev")}
    )

    transform_error: Optional[Exception] = None

    with tracker.run():
        tracker.log_params({"n_epochs": n_epochs, "n_samples": df.shape[0]})
        try:
            if isinstance(model, umap.UMAP):
                Y_new = model.transform(X_new_scaled)
            else:
                Y_new = model.transform(X_new_scaled, n_epochs=n_epochs)
            tracker.log_metrics({"transform_success": 1})
        except Exception as e:
            tracker.log_metrics({"transform_success": 0})
            transform_error = e

        if transform_error is None:
            tracker.log_metrics(
                {
                    "output_shape_0": Y_new.shape[0],
                    "output_shape_1": Y_new.shape[1],
                }
            )

    if transform_error is not None:
        monitor.log_error("/transform", "computation_error", is_critical=True)
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(transform_error)}")

    return {
        "embedding": Y_new.tolist(),
        "embedding_shape": Y_new.shape,
        "n_samples": df.shape[0],
    }


@app.post(
    "/umap",
    summary="One-shot UMAP projection",
    tags=["Production"]
)
async def apply_umap(
    file: UploadFile = File(...),
    params: UmapParameters = Depends(umap_parameters),
    x_client_source: Optional[str] = Header(None),
):
    """
    Accept a CSV file via multipart/form-data and return the UMAP projection.

    **Legacy endpoint** - For new usage, use /train and /transform instead for better performance
    and privacy.

    Parameters
    ----------
    file : UploadFile
        CSV file corresponding to the data to fit-transform
    params: UmapParameters
        including :
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

    df, content = await validate_and_read_csv(file=file)

    umap_params, n_samples, n_features = prepare_umap_params(df, params)

    monitor.log_input_size("/umap", len(content), n_samples, n_features)

    tracker = ExperimentTracker(
        experiment_name=get_experiment_path("umap-legacy", x_client_source),
        run_name=f"legacy-run-{file.filename}",
        run_tags={"mode": "legacy", "env": os.getenv("APP_ENV", "dev")},
    )

    with tracker.run():
        tracker.log_params(umap_params)

        model, scaler, dataset_standardized, dataset_transformed = fit_umap_model(
            df, umap_params, tracker,
        )

    return {"embedding": dataset_transformed.tolist()}
