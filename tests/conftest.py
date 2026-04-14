"""
pytest fixture
"""

import numpy as np
import pytest
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
