[![PR Tests](https://github.com/victorgalmiche/umap-clustering-mise-en-prod/actions/workflows/tests.yaml/badge.svg)](https://github.com/victorgalmiche/umap-clustering-mise-en-prod/actions/workflows/tests.yaml)

# UMAP demonstration platform

`umap-clustering-mise-en-prod` is a project that applies Uniform Manifold Approximation and Projection (UMAP) — a modern non-linear dimensionality reduction algorithm — to real-world datasets and combines it with clustering techniques to uncover structure in high-dimensional data.

UMAP is known for producing meaningful low-dimensional embeddings that preserve local and some global structure of the original data, making it useful not just for visualization but also as a preprocessing step before clustering.

This project implements
- a backend API : a user can send a CSV file and obtain a low-dimensional embedding of his dataset. The backend is deployed at : `https://umap-api-mmvs.lab.sspcloud.fr`.
- a front-end website : a friendly interface to send your CSV file to the API and display the results. The frond-end allows the user to set parameters easily and is deployed at `https://umap-streamlit-mmvs.lab.sspcloud.fr`

Note that there are restrictions on the dataset size that can be currently processed. CSV files should be less than 2M, with no more than 500 lines and only numerical columns.


# For Users


## About UMAP

UMAP (Uniform Manifold Approximation and Projection) is a method that:

* Constructs a graph of nearest neighbors in high dimension.

* Optimizes a low-dimensional embedding that preserves this structure.

* Is widely used for visualization and as a preprocessing step for clustering and other tasks.


## Using the website

The website at `https://umap-streamlit-mmvs.lab.sspcloud.fr` provides a user-friendly interface to our UMAP implementation. There are 4 main steps : 

- **Dataset selection** : use one of the demo datasets (digits, iris, etc.) or upload your own in CSV format (left panel, step 1.)
- **UMAP parameters** : leave the default parameters or use the sliders to experiment yourself (left panel, step 2.)
- **Dimensionality reduction** : click the "run UMAP" button to process the dataset. After a few seconds, the website displays the 2D embeddings and offers to download the result in CSV format.
- **Clustering** (optional) : this step unfolds after dimensionality reduction. Two methods (k-means and HDBSCAN) can be applied to the low-dimensional embeddings, and the result can be downloaded in CSV format.

**Save the projection function on the server** (Experimental, use at your own risk). The API and the website currently allow to save the projection function in order to apply it on other data. When the "Save model" checkbox is checked, the "run UMAP" button also returns an access key. Use this in the "Projection" tab to apply the saved model.


## Using the API directly

### 1. Projection (`POST /umap`)
Upload a CSV file, receive low-dimensional embeddings (classic fit-transform). Does not provide an access key or state persistence.

### 2. Training (`POST /train`)
Upload a CSV file to train a new UMAP manifold.
* **Inputs**: CSV file, UMAP hyperparameters (`n_neighbors`, `min_dist`, etc.).
* **Output**: A secure `access_key` and embeddings.
* **Side Effect**: Logs parameters and the trained model as a PyFunc artifact in MLflow.

### 3. Projection (`POST /transform`)
Apply an existing model to new data.
* **Inputs**: Secure `access_key` and a CSV file with new data.
* **Output**: Low-dimensional coordinates (embedding).
* **Benefit**: Ensures the projection is consistent with the original training manifold.

### 4. Health Check (`GET /`)
Returns API version and status.


# For Developpers

## Installation / local API

1. Clone the repository:
```
git clone https://github.com/victorgalmiche/umap-clustering-mise-en-prod.git

cd umap-clustering-mise-en-prod
```

2. Install dependencies:
```
pip install uv
uv sync
```

3. Serve the API locally:
```bash
uv run uvicorn app.api.api:app
```
The service will be reachable at `http://127.0.0.1:8000`.  
Explore the interactive documentation at `http://127.0.0.1:8000/docs`.


## Project structure

```
.github/workflows : tests, linting, build Docker images and push to DockerHub
app/api : FastAPI API backend
app/streamlit : Streamlit frontend
docs : documentation
src : source code for UMAP
```

## Continuous Integration

On every push to Github, we use Actions to run tests and linters. If the tests are successful and the push was on the `main` branch, two Docker images are built and pushed to DockerHub : 
- `slithiaote/umap-api` : runs the FastAPI server
- `slithiaote/umap-streamlit` : runs the Streamlit server


## Continuous Deployment on SSPcloud

Deployment is handled by ArgoCD based on the `https://github.com/victorgalmiche/umap-deployment` repository. 
- `slithiaote/umap-api` is deployed to `https://umap-api-mmvs.lab.sspcloud.fr`
- `slithiaote/umap-streamlit` is deployed to `https://umap-streamlit-mmvs.lab.sspcloud.fr`

## Monitoring and model repository

A MLflow service is deployed in our project's namespace on SSPcloud. This service is used to 
- monitor calls and ressource usage of the API,
- store access keys and the corresponding projection function.

## Documentation and contributing

The `docs` directory documents each part of the project : 
```
docs/API.md
docs/CICD.md
docs/CONTRIBUTING.md
docs/MONITORING.md
docs/STREAMLIT.md
docs/UMAP.md
```
Please refer to these for further details.

In particular, `docs/CONTRIBUTING.md` proposes a step-by-step procedure for contributions to a repository that is semi-automatically deployed.

Note that the Streamlit frontend expects a running MLflow service and a running API service and that some URLs are currently hard-coded. Version v2.0 will make it easier to modify the deployment settings.


# 📚 References that helped us build our algorithms

1. McInnes, Leland; Healy, John; UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

2. Wei, Dong; Charikar, Moses; Kai, Li; Efficient k-nearest neighbor graph constructionfor generic similarity measures.

3. [ENSAE Mise en production course](https://ensae-reproductibilite.github.io/)


