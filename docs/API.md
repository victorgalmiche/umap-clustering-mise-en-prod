# UMAP API

A FastAPI application that currently returns the first five data rows of an uploaded CSV file. Intended : use the UMAP algorithm to return a low-dimension projection of the dataset

# Running the app locally

```bash
uv run uvicorn app.api:app
```

The service will be reachable at `http://127.0.0.1:8000`.  
Open `http://127.0.0.1:8000/docs` to explore the API in Swagger UI.

Open `http://127.0.0.1:8000/predict` to test that the API is live.

Send a POST request to `http://127.0.0.1:8000/umap` to test the API with a CSV file. An example is provided in `tests/test_api.sh`.


