"""
Dataset Validation and Background Training Job Service.
"""
from pathlib import Path
import io
import time
import os
import re
import threading
from datetime import datetime, timezone
import pandas as pd
from backend.services.storage_service import JSONStorage, STORAGE_DIR, _read_json, _write_json
from ml.admin_training import train_uploaded_dataset

DATASETS_DIR = STORAGE_DIR / "datasets"
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
    """Validate a labelled CSV and save it under a server-generated safe name."""
    if len(file_bytes) > 15 * 1024 * 1024:
        raise ValueError("Dataset exceeds the 15 MB upload limit.")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid CSV formatting: {exc}") from exc
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    missing_cols = [c for c in REQUIRED_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required supervised columns: {missing_cols}")
    if len(df) < 20:
        raise ValueError(f"Dataset too small for retraining (found {len(df)} rows, minimum required is 20).")
    for column in REQUIRED_FEATURES + [TARGET_COLUMN]:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Column '{column}' contains non-numeric values.")
        if column == TARGET_COLUMN and converted.isna().any():
            raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing or invalid values.")
        if not converted.dropna().map(lambda value: pd.notna(value) and abs(float(value)) != float('inf')).all():
            raise ValueError(f"Column '{column}' contains non-finite values.")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem).strip("._") or "dataset"
    stored_name = f"{int(time.time())}_{safe_stem}.csv"
    save_path = DATASETS_DIR / stored_name
    df.to_csv(save_path, index=False)
    metadata = {
        "filename": filename,
        "stored_filename": stored_name,
        "saved_path": str(save_path),
        "row_count": len(df),
        "feature_count": len(REQUIRED_FEATURES),
        "target": TARGET_COLUMN,
        "target_mean": round(float(pd.to_numeric(df[TARGET_COLUMN]).mean()), 4),
        "target_std": round(float(pd.to_numeric(df[TARGET_COLUMN]).std()), 4),
        "uploaded_by": username,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "approved",
    }
    with threading.Lock():
        datasets_list = _read_json(JSONStorage.DATASETS_FILE, [])
        datasets_list.append(metadata)
        _write_json(JSONStorage.DATASETS_FILE, datasets_list)
    JSONStorage.record_audit(username, "UPLOAD_DATASET", f"Uploaded and validated dataset: {stored_name} ({len(df)} rows)")
    return metadata

def _run_background_training(job_id: str, dataset_name: str, username: str):
    """Run real model fitting in a background thread and register a candidate."""
    try:
        jobs = JSONStorage.get_training_job(job_id) or {}
        dataset_path = jobs.get("dataset_path")
        if not dataset_path or not Path(dataset_path).is_file():
            raise FileNotFoundError("Uploaded dataset could not be found. Upload it again.")
        JSONStorage.update_training_job(job_id, {"status": "running", "progress_pct": 5, "stage": "validating dataset"})
        result = train_uploaded_dataset(dataset_path)
        metadata = result["metadata"]
        JSONStorage.update_training_job(job_id, {"progress_pct": 95, "stage": "registering candidate"})
        models = _read_json(JSONStorage.MODELS_FILE, [])
        models.append(metadata)
        _write_json(JSONStorage.MODELS_FILE, models)
        JSONStorage.update_training_job(job_id, {
            "status": "completed", "progress_pct": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "candidate_version": metadata["version"],
            "metrics": metadata["metrics"],
            "weights": metadata["weights"],
            "validation_report": result["validation_report"],
            "stage": "completed",
        })
        JSONStorage.record_audit(username, "TRAINING_COMPLETED", f"Created candidate {metadata['version']} from {dataset_name}; validation R2={metadata['metrics'].get('R2')}")
    except Exception as exc:
        JSONStorage.update_training_job(job_id, {
            "status": "failed", "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "stage": "failed",
        })
        JSONStorage.record_audit(username, "TRAINING_FAILED", f"Training job {job_id} failed: {exc}")

def start_training_job(dataset_name: str, username: str) -> dict:
    datasets = _read_json(JSONStorage.DATASETS_FILE, [])
    selected = next((item for item in reversed(datasets) if item.get("stored_filename") == dataset_name or item.get("filename") == dataset_name), None)
    if not selected:
        raise ValueError("Uploaded dataset was not found. Upload and validate a dataset first.")
    job = JSONStorage.create_training_job(dataset_name, username)
    JSONStorage.update_training_job(job["job_id"], {"dataset_path": selected["saved_path"]})
    t = threading.Thread(
        target=_run_background_training,
        args=(job["job_id"], dataset_name, username),
        daemon=True
    )
    t.start()
    JSONStorage.record_audit(username, "START_TRAINING", f"Started training job {job['job_id']} on dataset {dataset_name}")
    return job
