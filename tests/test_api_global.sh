#!/bin/bash

# Test the umap API with curl
# This script expects the API to be up and running

curl -X POST "https://umap-api-mmvs.lab.sspcloud.fr/umap" \
     -H "accept: text/plain" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@tests/iris.csv"