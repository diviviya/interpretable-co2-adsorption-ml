# Deployment Guide (Render)

Deploy the application as one FastAPI web service. FastAPI serves both the API and the static frontend, so a separate frontend host is not required.

## Render service

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`

## Required environment variables

Set these in the Render service dashboard. Do not commit real values.

- `ENVIRONMENT=production`
- `JWT_SECRET`: a randomly generated secret of at least 32 characters
- `JWT_EXPIRATION_HOURS=24`
- `ADMIN_USERNAME`: the initial administrator username
- `ADMIN_PASSWORD`: a strong unique administrator password
- `CO2_DATA_DIR=/var/data`

Attach a persistent disk to the service and mount it at `/var/data`. Account records, prediction history, model registry, training-job records, uploaded datasets, and admin-trained model bundles are stored under this directory. Without a persistent disk, runtime records can be lost on redeploy or restart.

If the frontend is served by this same FastAPI service, leave `CORS_ALLOWED_ORIGINS` empty. If a separate frontend origin is introduced later, set it to a comma-separated list of exact trusted origins, including scheme and hostname.

## Model files

The initial ensemble remains in the repository at `artifacts/deployment/co2_adsorption_ensemble.joblib`. Admin-trained candidates are written to the persistent disk and their absolute paths are stored in the model registry. On startup, the backend restores the registered active model from that path. Keep the Python and ML package versions in `requirements.txt` aligned with those used to create the serialized model artifacts.

## Operating limits

- JSON-file storage is intended for a small academic demonstration and a single service instance. Do not scale horizontally; multiple instances would not share transactional writes.
- Admin training runs in an in-process background thread. A service restart interrupts an active training job.
- A persistent disk is required to retain account/history data, uploaded datasets, and trained model versions.
- After deployment, verify `/api/health`, register a normal user, sign in as admin, run a prediction, inspect history, and test a CSV upload/download.
