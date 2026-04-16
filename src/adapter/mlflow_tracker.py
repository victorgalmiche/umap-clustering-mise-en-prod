import logging
import os
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

import mlflow
import mlflow.models
import mlflow.pyfunc
import numpy as np
import tempfile

logger = logging.getLogger(Path(__file__).stem)
load_dotenv(override=True)


class ExperimentTracker:
    def __init__(
        self, experiment_name: str, run_name: str | None = None, run_tags: dict[str, str] | None = None
    ) -> None:

        # URI recovery
        mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_server:
            raise ValueError("MLFLOW_TRACKING_URI n'est pas définie dans l'environnement.")

        self.tracking_uri = mlflow_server
        mlflow.set_tracking_uri(self.tracking_uri)

        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Experiment
        self.experiment_name = experiment_name

        try:
            exp = mlflow.set_experiment(self.experiment_name)
            self.experiment_id = exp.experiment_id
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à MLflow: {e}")
            raise

        self.run_tags = run_tags
        self.run_name = run_name
        self.current_run_id = None

        logger.info(f"Connecté au serveur MLflow distant: {self.tracking_uri}")

    @contextmanager
    def run(self):
        run = mlflow.start_run(run_name=self.run_name)
        self.current_run_id = run.info.run_id
        mlflow.set_tags(self.run_tags)
        try:
            yield run
        finally:
            mlflow.end_run()

    def log_metrics(
        self,
        metrics: dict[str, int | float | None],
    ) -> None:
        logger.info(f"Logging metrics {metrics}")
        mlflow.log_metrics(metrics=metrics)

    def log_params(self, params: dict[str, str | int | float | None]) -> None:
        """Log multiple parameters at once"""
        logger.info(f"Logging params: {params}")
        mlflow.log_params(params)

    def log_pyfunc_model(
        self,
        pyfunc_model,
        artifact_path: str,
        registered_model_name: str,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> None:

        logger.info(f"Logging PyFunc model to artifact path: {artifact_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_X = os.path.join(tmp_dir, "X_train.npy")
            path_Y = os.path.join(tmp_dir, "Y_train.npy")

            np.save(path_X, X_train)
            np.save(path_Y, Y_train)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=pyfunc_model,
                artifacts={
                    "X_train": path_X,
                    "Y_train": path_Y,
                },
                registered_model_name=registered_model_name,
            )

        logger.info("Model successfully logged.")


class UmapStorage(mlflow.pyfunc.PythonModel):
    def __init__(self, umap_model):
        self.umap_model = umap_model

    def load_context(self, context):
        """
        Load data from MLflow artifacts
        """

        path_X = context.artifacts["X_train"]
        path_Y = context.artifacts["Y_train"]

        X_train = np.load(path_X)
        Y_train = np.load(path_Y)

        self.umap_model.X_train_ = X_train
        self.umap_model.Y_train_ = Y_train

    def predict(self, context, model_input):
        """
        model_input : np.ndarray ou pandas.DataFrame
        """
        if hasattr(model_input, "values"):
            X_new = model_input.values
        else:
            X_new = np.array(model_input)

        return self.umap_model.transform(X_new)
