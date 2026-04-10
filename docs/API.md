# UMAP API

A FastAPI application that uses the UMAP algorithm to return a low-dimension projection of the dataset provided as a CSV file.

# Running the app locally

```bash
uv run uvicorn app.api:app
```

The service will be reachable at `http://127.0.0.1:8000`.  
Open `http://127.0.0.1:8000/docs` to explore the API in Swagger UI.

Open `http://127.0.0.1:8000/predict` to test that the API is live.

Send a POST request to `http://127.0.0.1:8000/umap` to test the API with a CSV file. An example is provided in `tests/test_api.sh`. This takes a few seconds to compute.

# How it works

Code is in `app/api.py`.

The source code defines entrypoints, which correspond an URL. The client sends a HTTP request to that URL.

Note that to send files, the client needs to send a POST request.

Several parameters can be added, including:
- `n_neighbors`: 15 by default,
- `n_components`: 2 by default,
- `min_dist`: 0.1 by default,
- `knn_metric`: "euclidean" by default,
- `knn_method`: "approx" by default,

# Notes

When the hand-made umap algorithm does not work, the API switches to the umap-learn algorithm. The KNN is then automatically approximated and not exact.

