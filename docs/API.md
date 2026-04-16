# UMAP Management API

A production-ready FastAPI application for UMAP dimensionality reduction. It supports stateful model management, secure access keys for data projection, and automated experiment tracking via MLflow.

## Key Features
* **Stateful Training**: Train a model once, receive a secure token, and use it later for consistent projections.
* **MLflow Integration**: Automated logging of parameters, metrics, and models.
* **Environment Isolation**: Separate experiments for `dev`, `prod`, and `streamlit`.
* **Robust Fallback**: Automatically switches to `umap-learn` if the custom implementation fails.

---

## Running the App Locally

### 1. Configure Environment
Create a `.env` file in the root directory:
```text
MLFLOW_TRACKING_URI=your_mlflow_server_url
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
APP_ENV=dev
```

### 2. Launch the Service
```bash
uv run uvicorn app.api.api:app
```
The service will be reachable at `http://127.0.0.1:8000`.  
Explore the interactive documentation at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

### 1. Training (`POST /train`)
Upload a CSV file to train a new UMAP manifold.
* **Inputs**: CSV file, UMAP hyperparameters (`n_neighbors`, `min_dist`, etc.).
* **Output**: A secure `access_key`.
* **Side Effect**: Logs parameters and the trained model as a PyFunc artifact in MLflow.

### 2. Projection (`POST /transform`)
Apply an existing model to new data.
* **Inputs**: Secure `access_key` and a CSV file with new data.
* **Output**: Low-dimensional coordinates (embedding).
* **Benefit**: Ensures the projection is consistent with the original training manifold.

### 3. Legacy One-Shot (`POST /umap`)
Classic fit-transform operation. Does not provide an access key or state persistence.

### 4. Health Check (`GET /`)
Returns API version and status.

---

## Experiment Tracking (MLflow)

The API uses a dynamic experiment naming convention based on the `APP_ENV` variable and the `X-Client-Source` header:
* **Path Pattern**: `/{environment}/{operation-type}`
* **Environments**: Runs are grouped into `/dev/`, `/prod_user/`, or `/streamlit/` to keep the dashboard organized.

---

## Technical Notes

### Automated Fallback
The API prioritizes the custom `umap_mapping` implementation. If an error occurs during fitting:
1. It logs a `training_success: 0` metric to MLflow.
2. It sets a `fallback: True` parameter.
3. It completes the request using the standard `umap-learn` library to ensure service continuity.

### Security
Model data is isolated in an in-memory cache. Access to a specific model's transform capability is restricted to users holding the unique `access_key` generated at training time.

### Dependencies
Managed via `uv`. Key libraries include `polars` for fast I/O, `scikit-learn` for preprocessing, and `mlflow` for lifecycle management.