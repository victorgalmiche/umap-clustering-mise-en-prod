"""A simple API to expose our implementation of UMAP"""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import polars as pl
import umap
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

from src.umap_algo.umap_class import umap_mapping

logger = logging.getLogger(Path(__file__).stem)

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
    "/umap",
    summary="Return the UMAP projection of an uploaded CSV file",
    response_description="a string representation of the UMAP projection",
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

    Parameters
    ----------
    file : UploadFile
        The CSV file uploaded by the client.

    Returns
    -------
    JSONResponse
        JSON object with a single key ``embedding`` containing a list of embeddings.

    Raises
    ------
    HTTPException
        * 400 – if the file is not a CSV.
    """
    # Basic file‑type check
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. "
            f"Received: {file.filename}",
        )
    # TODO: handle parquet files

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
        dataset_transformed = model.fit_transform(dataset_standardized)

    except Exception:
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

    return repr_json         # JSONResponse(content={"embedding": list of embeddings})
