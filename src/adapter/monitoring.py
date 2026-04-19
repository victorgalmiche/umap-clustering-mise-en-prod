"""
Monitoring module for API metrics tracking via MLflow
"""

import logging
import os
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import mlflow

logger = logging.getLogger(Path(__file__).stem)


class ApplicationMonitor:
    """
    Tracks application-level metrics: latency, request sizes, errors.
    Uses MLflow for centralized monitoring.
    """

    def __init__(self, experiment_name: str = "/monitoring/app-metrics"):
        """
        Initialize the application monitor.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment path for monitoring metrics
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Monitoring initialized with experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")

    @contextmanager
    def track_request(self, endpoint: str, method: str):
        """
        Context manager to track request metrics (latency, success/failure).

        Parameters
        ----------
        endpoint : str
            API endpoint name (e.g., '/train', '/transform')
        method : str
            HTTP method (GET, POST, etc.)

        Yields
        ------
        dict
            Metrics dictionary to update with custom values
        """
        start_time = time.time()
        metrics = {
            "latency_ms": 0,
            "success": 0,
            "error": 0,
        }

        try:
            yield metrics
            metrics["success"] = 1
        except Exception as e:
            metrics["error"] = 1
            logger.error(f"Request error on {method} {endpoint}: {str(e)}")
            raise
        finally:
            latency = (time.time() - start_time) * 1000  # Convert to ms
            metrics["latency_ms"] = latency

            # Log to MLflow
            run_name = f"{method}-{endpoint.lstrip('/')}-tracking"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("endpoint", endpoint)
                mlflow.set_tag("method", method)
                mlflow.log_metrics(metrics)

    def log_input_size(self, endpoint: str, file_size_bytes: int, n_samples: int, n_features: int):
        """
        Log input data size metrics.

        Parameters
        ----------
        endpoint : str
            API endpoint name
        file_size_bytes : int
            Uploaded file size in bytes
        n_samples : int
            Number of rows in CSV
        n_features : int
            Number of features/columns
        """
        run_name = f"input_size-{endpoint.lstrip('/')}-{n_samples}samples-{n_features}features"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("metric_type", "input_size")
            mlflow.log_metrics(
                {
                    "file_size_kb": file_size_bytes / 1024,
                    "n_samples": n_samples,
                    "n_features": n_features,
                }
            )

    def log_error(self, endpoint: str, error_type: str, is_critical: bool = False):
        """
        Log error metrics.

        Parameters
        ----------
        endpoint : str
            API endpoint where error occurred
        error_type : str
            Type of error (e.g., 'invalid_csv', 'invalid_access_key', 'computation_error')
        is_critical : bool
            Whether the error is critical (service-impacting)
        """
        run_name = f"error-{endpoint.lstrip('/')}-{error_type}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("error_type", error_type)
            mlflow.set_tag("critical", str(is_critical))
            mlflow.log_metric("error_count", 1)
            logger.warning(f"Error on {endpoint}: {error_type} (critical={is_critical})")

    def log_cache_status(self, cache_size: int, max_models: int = 100):
        """
        Log model cache status.

        Parameters
        ----------
        cache_size : int
            Current number of cached models
        max_models : int
            Maximum models allowed in cache
        """
        run_name = f"cache_status-{cache_size}_{max_models}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("metric_type", "cache_status")
            mlflow.log_metrics(
                {
                    "cached_models": cache_size,
                    "cache_utilization_pct": (cache_size / max_models) * 100,
                }
            )

    def log_request(self, endpoint: str, method: str, status_code: int, latency_ms: float) -> None:
        """
        Log a successful HTTP request with latency and status code.

        This method records request-level metrics in MLflow, including latency,
        HTTP status code, and a success indicator.

        Parameters
        ----------
        endpoint : str
            API endpoint path (e.g., '/train', '/transform')
        method : str
            HTTP method used for the request (e.g., 'GET', 'POST')
        status_code : int
            HTTP response status code (e.g., 200, 404, 500)
        latency_ms : float
            Request processing time in milliseconds

        Returns
        -------
        None
            This method does not return anything. Metrics are logged to MLflow.

        Notes
        -----
        - A request is considered successful if status_code < 400
        - Metrics logged:
            - latency_ms
            - success (1 or 0)
        - Tags:
            - endpoint
            - method
            - status_code
        """
        run_name = f"{method}-{endpoint.lstrip('/')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("method", method)
            mlflow.set_tag("status_code", status_code)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("success", 1 if status_code < 400 else 0)

    def log_request_error(self, endpoint: str, method: str) -> None:
        """
        Log a failed HTTP request.

        This method is typically used in exception handlers or middleware when
        a request raises an unhandled error.

        Parameters
        ----------
        endpoint : str
            API endpoint path where the error occurred (e.g., '/train')
        method : str
            HTTP method used for the request (e.g., 'GET', 'POST')

        Returns
        -------
        None
            This method does not return anything. Error metrics are logged to MLflow.

        Notes
        -----
        - This method logs a single metric:
            - error = 1
        - Tags:
            - endpoint
            - method
        - Does not include latency (should be handled separately if needed)
        """
        run_name = f"{method}-{endpoint.lstrip('/')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("endpoint", endpoint)
            mlflow.set_tag("method", method)
            mlflow.log_metric("error", 1)


# Singleton instance
_monitor: Optional[ApplicationMonitor] = None


def get_monitor() -> ApplicationMonitor:
    """Get or create the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ApplicationMonitor()
    return _monitor
