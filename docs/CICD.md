# Introduction

The API runs on a server on the sspcloud. We need to describe this server (source code and software environment) and describe how the sspcloud can make the API available to the rest of the world.

# Continuous Integration

## API Docker image

The API logic is encapsulated in a Docker image. This means that we describe a complete virtual machine that accepts POST requests and returns results.

- The logic is implemented using FastAPI and described in `app/api/`
- `Dockerfile` describes the Docker image. It installs the python libraries and indicates which commands to use to start the web server.
- A github workflow in `.github/workflows/api_image.yml` is triggered upon push to Github. On each push, the github action builds the Docker image and sends it to the DockerHub repository `slithiaote/umap-api:latest`. The username is filled with a Github secret in the action (we did not setup a shared DockerHub repository).

## Streamlit Docker image

Another Docker image is required for the Streamlit front-end.

- The logic is implemented in Streamlit and described in `app/streamlit`
- The DockerHub repository is `slithiaote/umap-streamlit`

# Continuous Deployment

- Deployment is handled by ArgoCD based on the `https://github.com/victorgalmiche/umap-deployment` repository. Configuration is in `deployment`. 
- The backend API is deployed to `https://umap-api-mmvs.lab.sspcloud.fr`
- The front-end streamlit API is deployed to `https://umap-streamlit-mmvs.lab.sspcloud.fr`
- ArgoCD detects pushes to this repository, and updates the deployment.

