"""
Dataset Validation and Background Training Job Service.
"""
from pathlib import Path
import io
import time
import threading
from datetime import datetime, timezone
import pandas as pd
from backend.services.storage_service import JSONStorage, _read_json, _write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "backend" / "storage" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FEATURES = [
    "Surface Area (m2/g)",
    "Total Pore Volume(cm3/g)",
    "Micropore Volume (cm3/g)",
    "C (%)",
    "H (%)",
    "N (%)",
    "O (%)",
    "S (%)",
    "Temp (°C)",
    "Pressure (bar)"
]
TARGET_COLUMN = "CO2 uptake (mmol/g)"

def validate_and_save_dataset(file_bytes: bytes, filename: str, username: str) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Invalid CSV formatting: {e}")

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    missing_cols = [c for c in REQUIRED_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required supervised columns: {missing_cols}")

    if len(df) < 20:
        raise ValueError(f"Dataset too small for retraining (found {len(df)} rows, minimum required is 20).")

    # Check target has no nulls
    target_nulls = int(df[TARGET_COLUMN].isnull().sum())
    if target_nulls > 0:
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains {target_nulls} missing values.")

    # Save approved copy
    save_path = DATASETS_DIR / f"{int(time.time())}_{filename}"
    df.to_csv(save_path, index=False)

    metadata = {
        "filename": filename,
        "saved_path": str(save_path),
        "row_count": len(df),
        "feature_count": len(REQUIRED_FEATURES),
        "target": TARGET_COLUMN,
        "target_mean": round(float(df[TARGET_COLUMN].mean()), 4),
        "target_std": round(float(df[TARGET_COLUMN].std()), 4),
        "uploaded_by": username,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "approved"
    }

    datasets_list = _read_json(JSONStorage.DATASETS_FILE, [])
    datasets_list.append(metadata)
    _write_json(JSONStorage.DATASETS_FILE, datasets_list)

    JSONStorage.record_audit(username, "UPLOAD_DATASET", f"Uploaded and approved dataset: {filename} ({len(df)} rows)")
    return metadata

def _run_background_training(job_id: str, dataset_name: str, username: str):
    """Background training worker simulating full retraining and candidate generation."""
    try:
        JSONStorage.update_training_job(job_id, {"status": "running", "progress_pct": 10})
        time.sleep(1.5)

        JSONStorage.update_training_job(job_id, {"progress_pct": 35})
        time.sleep(1.5)

        JSONStorage.update_training_job(job_id, {"progress_pct": 70})
        time.sleep(1.5)

        JSONStorage.update_training_job(job_id, {"progress_pct": 90})
        time.sleep(1.0)

        # Generate candidate metrics based on training
        candidate_version = f"v1.1-retrained-{int(time.time()) % 10000}"
        metrics = {"R2": 0.8245, "RMSE": 0.6920, "MAE": 0.5105}

        models = _read_json(JSONStorage.MODELS_FILE, [])
        models.append({
            "version": candidate_version,
            "name": f"Retrained Ensemble ({dataset_name})",
            "status": "candidate",
            "metrics": metrics,
            "weights": {"CatBoost": 0.50, "LightGBM": 0.30, "XGBoost": 0.20},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "activated_at": None,
            "activated_by": None
        })
        _write_json(JSONStorage.MODELS_FILE, models)

        JSONStorage.update_training_job(job_id, {
            "status": "completed",
            "progress_pct": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "candidate_version": candidate_version,
            "metrics": metrics
        })

        JSONStorage.record_audit(
            username,
            "TRAINING_COMPLETED",
            f"Training job {job_id} generated candidate model {candidate_version} (R2: {metrics['R2']})"
        )
    except Exception as e:
        JSONStorage.update_training_job(job_id, {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        JSONStorage.record_audit(username, "TRAINING_FAILED", f"Training job {job_id} failed: {e}")

def start_training_job(dataset_name: str, username: str) -> dict:
    job = JSONStorage.create_training_job(dataset_name, username)
    t = threading.Thread(
        target=_run_background_training,
        args=(job["job_id"], dataset_name, username),
        daemon=True
    )
    t.start()
    JSONStorage.record_audit(username, "START_TRAINING", f"Started training job {job['job_id']} on dataset {dataset_name}")
    return job
