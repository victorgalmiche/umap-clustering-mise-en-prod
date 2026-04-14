"""A simple API to expose our implementation of UMAP"""

import io
import secrets
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import polars as pl
import umap
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import mlflow
import os

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker, UmapStorage

logger = logging.getLogger(Path(__file__).stem)

# In-memory model cache: access_key -> (model, scaler, X_train, Y_train)
model_cache = {}

# MLflow tracking
mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_server:
    mlflow.set_tracking_uri(mlflow_server)

app = FastAPI(
    title="UMAP API",
    description="Dimension reduction with UMAP algorithm"
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "UMAP API",
        "Model_name": "UMAP",
        "Model_version": "0.2",
    }


@app.post(
    "/train",
    summary="Train a UMAP model and get a secure access key",
    tags=["Model Management"]
)
async def train_model(
    file: UploadFile = File(...),
    n_neighbors: int = Form(15),
    n_components: int = Form(2),
    min_dist: float = Form(0.1),
    knn_metric: str = Form("euclidean"),
    knn_method: str = Form("approx"),
    n_epochs: int = Form(200),
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

    Returns
    -------
    dict
        Contains:
        - access_key: Secure random token for /transform
        - embedding_shape: Shape of training embedding
        - n_samples: Number of training samples
        - message: Usage instructions
    """
    # Validate file
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files accepted. Received: {file.filename}",
        )

    # Read data
    content = await file.read()
    buffer = io.BytesIO(content)
    logger.info("Reading training data...")
    df = pl.read_csv(buffer)

    n_samples = df.shape[0]
    n_features = df.shape[1]

    # Start MLflow run
    experiment_name = "umap-training"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "knn_metric": knn_metric,
            "knn_method": knn_method,
            "n_epochs": n_epochs,
            "n_samples": n_samples,
            "n_features": n_features,
        })

        # Preprocess
        logger.info("Preprocessing data...")
        scaler = StandardScaler()
        dataset_standardized = scaler.fit_transform(df.to_pandas())

        # Train model
        logger.info(f"Training UMAP on {n_samples} samples...")
        model = umap_mapping(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            KNN_metric=knn_metric,
            KNN_method=knn_method,
        )

        try:
            result = model.fit_transform(dataset_standardized, n_epochs=n_epochs)

            mlflow.log_metric("training_success", 1)
        except Exception as e:
            logger.warning(f"Custom UMAP failed: {e}. Using umap-learn...")
            model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric=knn_metric
            )
            Y = model.fit_transform(dataset_standardized)
            mlflow.log_metric("training_success", 0)
            mlflow.log_param("fallback_to_umap_learn", True)

        # Log model with MLflow
        pyfunc_model = UmapStorage(model)
        mlflow.pyfunc.log_model(
            artifact_path="umap_model",
            python_model=pyfunc_model,
            artifacts={
                "X_train": None,  # Will be saved below
                "Y_train": None,
            },
        )

        # Log metrics
        mlflow.log_metrics({
            "output_shape_0": Y.shape[0],
            "output_shape_1": Y.shape[1],
        })

        # Generate secure access key
        access_key = secrets.token_urlsafe(32)
        
        # Cache model
        model_cache[access_key] = (model, scaler, dataset_standardized, Y)
        logger.info(f"Model trained and cached with access key (secure)")

    return {
        "access_key": access_key,
        "message": "Use this key to transform new data with /transform endpoint",
        "embedding_shape": Y.shape,
        "n_samples": n_samples,
        "n_features": n_features,
    }


@app.post(
    "/transform",
    summary="Transform new data using a trained model",
    tags=["Model Management"]
)
async def transform_data(
    access_key: str = Form(...),
    file: UploadFile = File(...),
    n_epochs: int = Form(100),
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

    Returns
    -------
    dict
        Contains embedding and metadata
    """
    # Validate access_key
    if access_key not in model_cache:
        raise HTTPException(
            status_code=403,
            detail="Invalid access_key. Use the key returned from /train endpoint.",
        )

    # Validate file
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files accepted. Received: {file.filename}",
        )

    # Retrieve cached model and scaler
    model, scaler, X_train_scaled, Y_train = model_cache[access_key]

    # Read new data
    content = await file.read()
    buffer = io.BytesIO(content)
    logger.info("Reading new data...")
    df = pl.read_csv(buffer)

    n_samples_new = df.shape[0]

    # Preprocess new data using the same scaler
    logger.info("Preprocessing new data...")
    X_new_scaled = scaler.transform(df.to_pandas())

    # Start MLflow run for transform
    with mlflow.start_run():
        mlflow.log_params({
            "n_epochs": n_epochs,
            "n_samples_new": n_samples_new,
            "operation": "transform",
        })

        # Transform using trained model
        logger.info(f"Transforming {n_samples_new} new samples...")
        try:
            Y_new = model.transform(X_new_scaled, n_epochs=n_epochs)
            mlflow.log_metric("transform_success", 1)
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            mlflow.log_metric("transform_success", 0)
            raise HTTPException(status_code=500, detail=str(e))

        mlflow.log_metrics({
            "output_shape_0": Y_new.shape[0],
            "output_shape_1": Y_new.shape[1],
        })

    return {
        "embedding": Y_new.tolist(),
        "embedding_shape": Y_new.shape,
        "n_samples": n_samples_new,
    }


@app.post(
    "/umap",
    summary="Return the UMAP projection of an uploaded CSV file (legacy)",
    response_description="a string representation of the UMAP projection",
    tags=["Legacy"]
)
async def apply_umap(
    file: UploadFile = File(...),
    n_neighbors: int = Form(15),
    n_components: int = Form(2),
    min_dist: float = Form(0.1),
    knn_metric: str = Form("euclidean"),
    knn_method: str = Form("approx"),
):
    """
    Accept a CSV file via multipart/form-data and return the UMAP projection.
    
    **Legacy endpoint** - For new usage, use /train and /transform instead for better performance and privacy.

    Parameters
    ----------
    file : UploadFile
        The CSV file uploaded by the client.

    Returns
    -------
    dict
        JSON object with embedding
    """
    # Basic file‑type check
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. "
            f"Received: {file.filename}",
        )

    # Read the entire file into memory
    content = await file.read()
    buffer = io.BytesIO(content)

    logger.info("Reading data ...")
    df = pl.read_csv(buffer)

    # Get parameters
    logger.info("Get model parameters ...")
    scaler = StandardScaler()
    model = umap_mapping(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        KNN_metric=knn_metric,
        KNN_method=knn_method,
    )

    logger.info("Fitting model...")
    dataset_standardized = scaler.fit_transform(df.to_pandas())
    try:
        result = model.fit_transform(dataset_standardized)
        # Handle both return formats: Y or (Y, anim)
        if isinstance(result, tuple):
            dataset_transformed = result[0]
        else:
            dataset_transformed = result

    except Exception as e:
        logger.warning(f"Custom UMAP failed: {e}. Falling back to umap-learn...")
        model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=knn_metric
        )

        dataset_transformed = model.fit_transform(dataset_standardized)

    # Return the transformed dataset
    logger.info("All done, returning transformed data")
    repr_json = {"embedding": dataset_transformed.tolist()}

    return repr_json
