# How to develop in the context of an automatically deployed application

Aka how to test the app without making noisy commits to the main branch. A proposal follows.

# Create a new branch

- git branch
- git checkout

# Develop inside the new branch

- **local test** : `uv run uvicorn`. This works because a proxy system is in place on onyxia and the service / port is reachable in a web browser
- git commit
- git push
- CI process builds the Docker image and pushes it to **Docker Hub**, even for test branches. 
- Deployment is not affected because it uses hard-coded tags.
- **test in Docker** : `kubectl run -it api-ml --image=your_image` can be used to test image.

# Get ready to merge

- create a pull request. Note that a temporary pull request can be created on Github.
- bump tags for the Docker images
- you can still make changes after tagging the Docker images, until these are pulled / deployed, cf below.
- merge the pull request (this triggers a push to DockerHub)

# Deploy by pushing to the Deployment repo

- Developper manually updates the tags in the deployment repository.
- Argo CD detects changes in the deployment configuration
- Argo CD pulls the new Docker image into onyxia
- Kubernetes deploys the application into multiple Pods

_Why this is necessary_ : ArgoCD pulls from DockerHub, but only once for each tag. This is a Kubernetes default config: the image is pulled IfNotPresent.
Consequently, you need to modify the image tag in `deployment.yaml` to pull an updated Docker image.
Therefore there is a small uncertainty in the exact version that is deployed. We use the SHA digest to fix the exact version of the Docker image. However, identifying the corresponding Github commit or tag is not trivial (this is on v2.0 todo list).

Notes and alternatives. 
1) If the image tag is not specified or is set to latest, the pull policiy defaults to Always. 
2) In DockerHub, tags can be set to be immutable to ensure consistency.
3) An ArgoCD image updater resource (an additional pod)can detect updates to DockerHub. It does so by making commits to the deployment repository

How to test changes to the deployment repo without committing to main ?
- solution 1: test in a different namespace
- solution 2: use another ArgoCD application to track a `dev` branch

