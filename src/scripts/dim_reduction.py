""" """

import hydra
import os
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.manifold import trustworthiness
import logging
from pathlib import Path

from src.umap_algo.umap_class import umap_mapping
from src.adapter.mlflow_tracker import ExperimentTracker

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
        run_tags={"user": os.getenv("USER"), "dataset": cfg.mlflow.dataset_name},
    )

    logger.info("Logging metrics, params...")
    with experiment_tracker.run():
        experiment_tracker.log_metrics(metrics)
        experiment_tracker.log_params(hyperparameters)
    logger.info("End of the job...")


if __name__ == "__main__":
    job()
