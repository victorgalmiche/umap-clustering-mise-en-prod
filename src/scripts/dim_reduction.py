import logging
import os
from pathlib import Path

import hydra
from sklearn.datasets import load_iris
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.adapter.mlflow_tracker import ExperimentTracker, UmapStorage
from src.umap_algo.umap_class import umap_mapping

logger = logging.getLogger(Path(__file__).stem)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def job(cfg):

    # dataset = pd.read_parquet(config.path_dataset)
    logger.info("Loading Data...")
    dataset = load_iris().data

    scaler = StandardScaler()
    hyperparameters = cfg.umap
    model = umap_mapping(
        n_neighbors=hyperparameters.n_neighbors,
        n_components=hyperparameters.n_components,
        min_dist=hyperparameters.min_dist,
        KNN_metric=hyperparameters.KNN_metric,
        KNN_method=hyperparameters.KNN_method,
    )

    logger.info("Fitting model...")
    dataset_standardized = scaler.fit_transform(dataset)
    dataset_transformed = model.fit_transform(dataset_standardized)

    trust = trustworthiness(
        X=dataset, X_embedded=dataset_transformed, n_neighbors=cfg.metrics.n_neighbors_trustworthiness
    )

    metrics = {"trustworthiness": trust}

    experiment_tracker = ExperimentTracker(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.mlflow.run_name,
        run_tags={"user": os.getenv("GIT_USER_NAME"), "dataset": cfg.mlflow.dataset_name},
    )

    logger.info("Preparing model registry...")
    model.X_train_, model.Y_train_ = np.array([]), np.array([])
    pyfunc_model = UmapStorage(model)

    logger.info("Logging metrics, params, registering model...")
    with experiment_tracker.run():
        experiment_tracker.log_metrics(metrics)
        experiment_tracker.log_params(hyperparameters)
        experiment_tracker.log_pyfunc_model(
            pyfunc_model=pyfunc_model,
            artifact_path=cfg.mlflow.artifact_path,
            registered_model_name=cfg.mlflow.registered_model_name,
            X_train=dataset_standardized,
            Y_train=dataset_transformed,
        )

    # use model
    # import mlflow
    # run_id = experiment_tracker.current_run_id
    # artifact_path = cfg.mlflow.artifact_path
    # model_uri = f"runs:/{run_id}/{artifact_path}"
    # loaded_model = mlflow.pyfunc.load_model(experiment_tracker.model_uri)

    logger.info("End of the job...")


if __name__ == "__main__":
    job()
