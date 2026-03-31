import os
import mlflow
from pathlib import Path
import mlflow.models
import mlflow.pyfunc
from typing import Dict, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(Path(__file__).stem)


class ExperimentTracker:
    def __init__(
        self, experiment_name: str, run_name: str | None = None, run_tags: dict[str, str] | None = None
    ) -> None:

        if mlflow.active_run() is not None:
            mlflow.end_run()

        logger.info("MLFLOW")
        logger.info(mlflow.active_run())

        self.experiment_name = f"/experiments/{experiment_name}"
        self.experiment_id = mlflow.set_experiment(experiment_name).experiment_id
        self.run_tags = run_tags
        self.run_name = run_name
        mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
        self.tracking_uri = mlflow_server

        mlflow.set_tracking_uri(mlflow_server)

        logger.info(
            f"Set up MLflow with experiment_name: {self.experiment_name}, "
            f"experiment_id: {self.experiment_id}, "
            f"saving experiment in {self.tracking_uri}"
        )

    @contextmanager
    def run(self):
        logger.info("MLFLOW2")
        logger.info(mlflow.active_run())
        run = mlflow.start_run(run_name=self.run_name)
        mlflow.set_tags(self.run_tags)
        try:
            yield run
        finally:
            mlflow.end_run()

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float, None]],
    ) -> None:
        logger.info(f"Logging metrics {metrics}")
        mlflow.log_metrics(metrics=metrics)

    def log_params(self, params: Dict[str, Union[str, int, float, None]]) -> None:
        """Log multiple parameters at once"""
        logger.info(f"Logging params: {params}")
        mlflow.log_params(params)
