FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim 
# Debian image with python and uv installed, less things to add

# Install Python
# RUN apt-get -y update && apt-get install -y python3-pip curl

# Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/root/.local/bin/:$PATH"

# Install project dependencies
COPY pyproject.toml .
RUN uv sync

COPY src ./src
COPY app ./app
COPY config ./config

CMD ["bash", "-c", "./app/api/run.sh"]
