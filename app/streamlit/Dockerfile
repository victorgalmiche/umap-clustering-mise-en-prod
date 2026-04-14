FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim 

COPY pyproject.toml .
RUN uv sync

COPY src ./src
COPY app ./app
COPY config ./config

CMD ["bash", "-c", "./app/streamlit/run.sh"]
