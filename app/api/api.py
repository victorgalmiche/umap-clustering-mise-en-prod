"""
UMAP Service API with Hydra Configuration
"""

import io
import os
import secrets
import logging
import time
from typing import Optional
from pathlib import Path

import polars as pl
import umap
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Request
from sklearn.preprocessing import StandardScaler
import hydra

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker, UmapStorage
from src.adapter.monitoring import get_monitor

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
    version="0.4",
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
    run_name = f"{method}-{endpoint.lstrip('/')}"

    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        # Log successful requests
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("method", method)
            mlflow.set_tag("status_code", response.status_code)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("success", 1 if response.status_code < 400 else 0)

        return response
    except Exception:
        # Log errors
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("method", method)
            mlflow.log_metric("error", 1)
        raise


def get_experiment_path(base_name: str, client_source: Optional[str] = None) -> str:
    """
    Generates the MLflow experiment path based on the environment and client source.
    Defaults to the APP_ENV variable or 'dev'.
    """
    env = client_source if client_source else os.getenv("APP_ENV", "dev")
    return f"/{env}/{base_name}"


def get_polars_from_request(content):
    """
    Get CSV data from the POST request
    Convert to a Polars dataframe
    Keep only numerical columns
    raise exception if more than 500 lines (reason: limit compute ressources)
    """

    df = pl.read_csv(io.BytesIO(content)).select(pl.selectors.numeric())

    if df.height >= 500:
        raise HTTPException(
            status_code=400,
            detail="CSV file must have less than 500 lines (ressources limit)."
            )

    if df.width < 3:
        raise HTTPException(
            status_code=400,
            detail="CSV file must have at least 3 numerical columns."
            )

    return df

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
    n_neighbors: int = Form(cfg.umap.n_neighbors, description="Number of neighbors for KNN"),
    n_components: int = Form(cfg.umap.n_components, description="Target dimension"),
    min_dist: float = Form(cfg.umap.min_dist, description="Minimum distance in the embedding"),
    knn_metric: str = Form(cfg.umap.KNN_metric, description="Distance metric"),
    knn_method: str = Form(cfg.umap.KNN_method, description="KNN search method: 'exact' or 'approx'"),
    n_epochs: int = Form(cfg.umap.n_epochs_train, description="Optimization iterations"),
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

    # Data ingestion
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    df = get_polars_from_request(content)
    n_samples, n_features = df.shape

    # Log input size metrics
    monitor.log_input_size("/train", len(content), n_samples, n_features)

    # MLflow setup
    exp_path = get_experiment_path("umap-training", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path,
        run_name=f"train-{file.filename}",
        run_tags={"env": os.getenv("APP_ENV", "dev"), "source": x_client_source or "api"},
    )

    with tracker.run():
        tracker.log_params(
            {
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_dist": min_dist,
                "knn_metric": knn_metric,
                "knn_method": knn_method,
                "n_epochs": n_epochs,
                "n_samples": n_samples,
                "n_features": n_features,
            }
        )

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
            model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=knn_metric)
            Y = model.fit_transform(dataset_standardized)
            tracker.log_metrics({"training_success": 0})
            tracker.log_params({"fallback": True})

        # MLflow persistence
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

    # Caching and access control
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

    if not file.filename.lower().endswith(".csv"):
        monitor.log_error("/transform", "invalid_csv_format")
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Retrieve objects from cache
    model, scaler, _, _ = model_cache[access_key]

    # Processing
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    df = get_polars_from_request(content)

    # Log input size metrics
    monitor.log_input_size("/transform", len(content), df.shape[0], df.shape[1])
    

    X_new_scaled = scaler.transform(df.to_pandas())

    exp_path = get_experiment_path("umap-transform", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path, run_name="transform-execution", run_tags={"env": os.getenv("APP_ENV", "dev")}
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
    n_neighbors: int = Form(cfg.umap.n_neighbors, description="Number of neighbors for KNN"),
    n_components: int = Form(cfg.umap.n_components, description="Target dimension"),
    min_dist: float = Form(cfg.umap.min_dist, description="Minimum distance in the embedding"),
    knn_metric: str = Form(cfg.umap.KNN_metric, description="Distance metric"),
    knn_method: str = Form(cfg.umap.KNN_method, description="KNN search method: 'exact' or 'approx'"),
    n_epochs: int = Form(cfg.umap.n_epochs_train, description="Optimization iterations"),
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
        monitor.log_error("/umap", "invalid_csv_format")
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    content = await file.read()
    df = get_polars_from_request(content)

    # Log input size metrics
    monitor.log_input_size("/umap", len(content), df.shape[0], df.shape[1])

    exp_path = get_experiment_path("umap-legacy", x_client_source)
    tracker = ExperimentTracker(
        experiment_name=exp_path,
        run_name=f"legacy-run-{file.filename}",
        run_tags={"mode": "legacy", "env": os.getenv("APP_ENV", "dev")},
    )

    with tracker.run():
        tracker.log_params({"n_neighbors": n_neighbors, "n_samples": df.shape[0]})

        scaler = StandardScaler()
        dataset_standardized = scaler.fit_transform(df.to_pandas())

        model = umap_mapping(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            KNN_metric=knn_metric,
            KNN_method=knn_method,
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
            Y_train=dataset_transformed,
        )

    return {"embedding": dataset_transformed.tolist()}
