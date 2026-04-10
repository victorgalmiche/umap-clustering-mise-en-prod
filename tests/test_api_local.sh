#!/bin/bash

# Test the umap API with curl

# do not forget to launch the server with
# uv run uvicorn app.api.api:app

curl -X POST "http://127.0.0.1:8000/umap" \
     -H "accept: text/plain" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@tests/iris.csv"