# Introduction

The API runs on a server on the sspcloud. We need to describe this server (source code and software environment) and describe how the sspcloud can make the API available to the rest of the world.

# Continuous Integration

## Docker image

The API logic is encapsulated in a Docker image. This means that we describe a complete virtual machine that accepts POST requests and returns results.

- `Dockerfile` describes the Docker image. It installs the python libraries and indicates which commands to use to start the web server.
- A github workflow in `.github/workflows/api_image.yml` is triggered upon push to the `seb-api` branch. On each push, the github action builds the Docker image and sends it to the DockerHub repository `${{username}}/umap-api:latest`. The variable is filled with a Github secret.

# Continuous Deployment

- Deployment is handled by ArgoCD based on the `https://github.com/victorgalmiche/umap-deployment` repository
- configuration is in `deployment`. The API is deployed to `https://umap-api-mmvs.lab.sspcloud.fr`
- ArgoCD detects pushes to this repository, and updates the deployment.

