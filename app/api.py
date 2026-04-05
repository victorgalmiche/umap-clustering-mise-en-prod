"""A simple API to expose our implementation of UMAP"""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import polars as pl
import hydra
import os
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import logging
from pathlib import Path

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker

logger = logging.getLogger(Path(__file__).stem)

app = FastAPI(
    title="UMAP API",
    description="Application de réduction de dimension avec l'algorithme UMAP"
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "UMAP API",
        "Model_name": "UMAP",
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["umap"]) # keep for testing the api
async def predict(
    sex: str = "female", age: float = 29.0, fare: float = 16.5, embarked: str = "S"
) -> str:
    """ """

    df = pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )

    prediction = "Survived 🎉" 

    return prediction

@hydra.main()
@app.post(
    "/umap",
    summary="Return the UMAP projection of an uploaded CSV file",
    response_description="a string representation of the UMAP projection",
)
async def umap(file: UploadFile = File(...)) -> str:
    """
    Accept a CSV file via multipart/form‑data and return the UMAP projection.

    Parameters
    ----------
    file : UploadFile
        The CSV file uploaded by the client.

    Returns
    -------
    JSONResponse
        JSON object with a single key ``preview`` containing a list of
        row dictionaries.

    Raises
    ------
    HTTPException
        * 400 – if the file is not a CSV or cannot be parsed.
    """
    # Basic file‑type check
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. "
            f"Received: {file.filename}",
        )

    try:
        # Read the entire file into memory – suitable for small‑to‑medium CSVs.
        # For huge files consider a streaming/partial read approach.
        content = await file.read()
        buffer = io.BytesIO(content)

        # Polars can read from a file‑like object.
        logger.info("Reading data ...")
        df = pl.read_csv(buffer)
        
        # Get parameters
        logger.info("Get model parameters ...")
        scaler = StandardScaler()
        with hydra.initialize(version_base=None, config_path="../config"):
            cfg = hydra.compose(config_name="main")
        hyperparameters = cfg.umap
        model = umap_mapping(
            n_neighbors=hyperparameters.n_neighbors,
            n_components=hyperparameters.n_components,
            min_dist=hyperparameters.min_dist,
            KNN_metric=hyperparameters.KNN_metric,
            KNN_method=hyperparameters.KNN_method,
        )
    
        logger.info("Fitting model...")
        dataset_standardized = scaler.fit_transform(df.to_pandas())
        dataset_transformed = model.fit_transform(dataset_standardized)

        # Return the transformed dataset
        logger.info("All done, returning transformed data")
        repr_str = dataset_transformed.__str__()     # or df.__str__()

    except Exception as exc: 
        # Polars raises a variety of exceptions on malformed data.
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {exc}",
        ) from exc

    return repr_str #JSONResponse(content={"preview": preview})

