"""
pytest fixture
"""

import numpy as np
import pytest
import subprocess
import time
import socket

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

np.random.seed(42)


@pytest.fixture(scope="session")
def iris():
    return load_iris()


@pytest.fixture(scope="session")
def iris_split():
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def wait_for_api(host="127.0.0.1", port=8000, timeout=10):
    """ wait for the api to be ready """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    raise RuntimeError("The API has not started")


@pytest.fixture(scope="session", autouse=True)
def start_api():
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.api.api:app", "--port", "8000"],
        env={**__import__("os").environ, "APP_ENV": "test"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    try:
        wait_for_api(timeout=60)
    except RuntimeError:
        print(proc.stderr.read().decode())
        proc.terminate()
        raise
    yield
    proc.terminate()
    proc.wait()