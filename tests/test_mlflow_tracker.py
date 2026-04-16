import os

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import mlflow
import tempfile

from adapter.mlflow_tracker import ExperimentTracker, UmapStorage
from umap_algo.umap_class import umap_mapping


@pytest.fixture(scope="function")
def mlflow_tmp_dir():
    """Create a temporary MLflow tracking URI"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.set_tracking_uri(f"file://{tmp_dir}")
        yield tmp_dir


@pytest.fixture
def trained_umap():
    X = load_iris().data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = umap_mapping(n_components=2)
    Y = model.fit_transform(X_scaled)

    model.X_train_ = np.array([])
    model.Y_train_ = np.array([])

    return model, X_scaled, Y


def test_full_mlflow_pipeline(mlflow_tmp_dir, trained_umap):
    model, X_scaled, Y = trained_umap

    tracker = ExperimentTracker(
        experiment_name="pytest_mlflow_tracker",
        run_name="test_full_mlflow_pipeline",
        run_tags={"user": os.getenv("GIT_USER_NAME"), "dataset": "iris"},
    )

    with tracker.run():
        tracker.log_pyfunc_model(
            pyfunc_model=UmapStorage(model),
            artifact_path="umap_model",
            registered_model_name=None,
            X_train=X_scaled,
            Y_train=Y,
        )

        run_id = tracker.current_run_id

    # Reload model
    model_uri = f"runs:/{run_id}/umap_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Test predict
    X_new = X_scaled[:10]
    preds = loaded_model.predict(X_new)

    assert preds.shape == (10, 2)


def test_load_context_restores_state(mlflow_tmp_dir, trained_umap):
    model, X_scaled, Y = trained_umap

    tracker = ExperimentTracker(
        experiment_name="pytest_mlflow_tracker",
        run_name="test_load_context_restores_state",
        run_tags={"user": os.getenv("GIT_USER_NAME"), "dataset": "iris"},
    )

    with tracker.run():
        tracker.log_pyfunc_model(
            pyfunc_model=UmapStorage(model),
            artifact_path="umap_model",
            registered_model_name=None,
            X_train=Y,
            Y_train=X_scaled,
        )
        run_id = tracker.current_run_id

    model_uri = f"runs:/{run_id}/umap_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    internal = loaded_model._model_impl.python_model.umap_model

    assert internal.X_train_.size > 0
    assert internal.Y_train_.size > 0


def test_model_is_broken_without_artifacts(trained_umap):
    model, _, _ = trained_umap

    assert model.X_train_.size == 0
    assert model.Y_train_.size == 0
