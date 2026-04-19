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

**Save the reduction function on the server** (Experimental, use at your own risk). The API and the website currently allow to save the reduction function in order to apply it on other data. When the "Save model" checkbox is checked, the "run UMAP" button also returns an access key. Use this in the "Projection" tab to apply the saved model.


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
docs
src
```

## Continuous Integration

## Continuous Deployment on SSPcloud

Deployment is handled by ArgoCD based on the `https://github.com/victorgalmiche/umap-deployment` repository. 

Note that the Streamlit frontend expects a running MLflow service and a running API service and that some URLs are currently hard-coded. 



# 📚 References that helped us build our algorithms

1. McInnes, Leland; Healy, John; UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

2. Wei, Dong; Charikar, Moses; Kai, Li; Efficient k-nearest neighbor graph constructionfor generic similarity measures.

3. [ENSAE Mise en production course](https://ensae-reproductibilite.github.io/)


----
(sort this into the outline)
## 📊 Typical Workflow

### 1. Load high-dimensional dataset
 Prepare your dataset with relevant features.

### 2. Dimensionality Reduction via UMAP
Reduce to a lower dimension (e.g., 2D or 10D) while preserving structure.

### 3. Apply Clustering
Run clustering algorithms such as K-Means, DBSCAN, or density-based methods over the UMAP output.

### 4. Evaluate Results
Use clustering metrics (e.g., silhouette score) to compare and validate performance.

## 🚀 Features

### 📉 Dimensionality Reduction
Uses UMAP to reduce high-dimensional data into a lower-dimensional space where structure can be more easily visualized. UMAP algorithm can be taken from this repository or from the library `umap-learn`.

### 🔍 Clustering Analysis
Applies clustering algorithms (e.g., K-Means, DBSCAN) on UMAP embeddings to identify meaningful groups in the data.

### 📊 Comparison with Other Techniques
Enables comparison of UMAP results with classical methods like PCA or t-SNE.



# Project Structure
| Folder              | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `test/`             | Scripts and notebooks for inspecting datasets and testing algorithms.   |
| `umap_algo/`        | Core UMAP implementation and embedding pipelines.                       |
| `umap_comparisons/` | Comparisons between UMAP and other dimensionality reduction approaches. |




## 📁 Datasets (Planned)

We plan to demonstrate UMAP clustering on the following datasets:

* Socio-economic and health description of countries (~9 dimensions)

* NYC Yellow Taxi Trip Data (~18 dimensions)

* French cities socio-economic indicators (~54 dimensions)

Each dataset showcases challenges like varying dimensions and features.



