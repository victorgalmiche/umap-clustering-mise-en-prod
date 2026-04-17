# API Monitoring

This document describes the application monitoring system for the UMAP API.

## Overview

The API includes built-in monitoring that automatically tracks:
- **Request latency** (response time in milliseconds)
- **Input data sizes** (file size, rows, columns)
- **Error rates** by endpoint and error type
- **Model cache status** (number of cached models)

All metrics are logged to **MLflow** under the `/monitoring/app-metrics` experiment.

## Monitoring Architecture

### Middleware-Based Tracking

A FastAPI middleware (`monitoring_middleware` in `app/api/api.py`) automatically tracks every request:
- Measures latency for all endpoints
- Logs HTTP status codes
- Records success/failure status
- Tags metrics with endpoint and HTTP method

```python
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    # Automatically tracks: latency_ms, success, status_code
```

### Metrics Collection

#### 1. Request-Level Metrics

Every request logs:
- `latency_ms`: Response time in milliseconds
- `success`: 1 (success) or 0 (failure)
- `status_code`: HTTP status code
- `endpoint`: API endpoint (e.g., `/train`, `/transform`)
- `method`: HTTP method (`GET`, `POST`, etc.)

**Example MLflow tags:**
```
endpoint: /train
method: POST
status_code: 200
```

#### 2. Input Size Metrics

For file upload endpoints (`/train`, `/transform`, `/umap`):
- `file_size_kb`: Upload file size in kilobytes
- `n_samples`: Number of rows in the CSV
- `n_features`: Number of features/columns

#### 3. Error Metrics

When errors occur, they are logged with:
- `error_type`: Type of error
  - `invalid_csv_format`: CSV format validation failed
  - `invalid_access_key`: Invalid model access key
  - `computation_error`: UMAP computation failed
- `critical`: Boolean indicating if error is service-impacting
- `error_count`: Counter (always 1 per log)

#### 4. Cache Status Metrics

After model training, the cache status is logged:
- `cached_models`: Current number of models in memory cache
- `cache_utilization_pct`: Percentage of cache capacity used (assuming max 100 models)

## Endpoints

### Health Check

```bash
GET /health
```

Returns service health status and monitoring information:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "cached_models": 5,
  "environment": "prod"
}
```

**Use case:** Kubernetes health probes, load balancer checks, monitoring dashboards

### Training

```bash
POST /train
```

Metrics logged:
- Input size (file, rows, columns)
- Training latency
- Cache size after training
- Success/failure status

### Transform

```bash
POST /transform
```

Metrics logged:
- Input size (file, rows, columns)
- Transform latency
- Error tracking (invalid key, computation error)
- Success/failure status

### Legacy Endpoint

```bash
POST /umap
```

Metrics logged:
- Input size
- Latency
- Success/failure status


## Viewing Metrics

Currently the metrics are logged to MLflow, but the existing dashboard graphs are not very useful.




