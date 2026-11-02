from pathlib import Path
import os
import io
import math
import re
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    FeatureInput,
    WhatIfRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    TrainJobStartRequest,
)
from backend.services.storage_service import JSONStorage, _read_json
from backend.services.auth_service import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_current_user_optional,
    get_current_admin,
)
from backend.services.prediction_service import (
    activate_prediction_bundle,
    predict_single,
    compute_shap,
    predict_batch_csv,
    bundle,
    FEATURE_COLUMNS,
)
from backend.services.training_service import (
    validate_and_save_dataset,
    start_training_job,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"

app = FastAPI(
    title="CO2 Adsorption Prediction & ML Lifecycle API",
    description="Enterprise-grade FastAPI backend for CO2 adsorption ensemble prediction, SHAP explainability, dataset validation, and model versioning.",
    version="2.1.0",
)

environment = os.getenv("ENVIRONMENT", "development").lower()
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if environment != "production" and not allowed_origins:
    allowed_origins = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _resolve_model_artifact_path(artifact_path: str) -> Path:
    path = Path(artifact_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _restore_active_model_bundle() -> None:
    """Restore the registered active model after a process restart."""
    active_model = JSONStorage.get_active_model_version()
    if not active_model or active_model.get("status") != "active":
        return
    artifact_path = active_model.get("artifact_path")
    if not artifact_path:
        raise RuntimeError(f"Active model {active_model.get('version')} has no artifact_path.")
    resolved_path = _resolve_model_artifact_path(artifact_path)
    if not resolved_path.is_file():
        raise RuntimeError(
            f"Active model artifact is missing: {resolved_path}. "
            "Check the persistent disk and model artifact configuration."
        )
    activate_prediction_bundle(resolved_path)


_restore_active_model_bundle()

# =============================================================================
# 1. SYSTEM HEALTH
# =============================================================================
@app.get("/api/health")
@app.get("/api/status")
def health_check():
    """Health check endpoint displaying system status and active model."""
    active_model = JSONStorage.get_active_model_version()
    return {
        "status": "healthy",
        "service": "CO2 Adsorption Prediction API",
        "active_model_version": active_model["version"] if active_model else "v1.0",
        "model_strategy": bundle["ensemble_strategy"],
        "storage": "JSON Persistent Store (No SQL)",
        "features_count": len(FEATURE_COLUMNS),
    }

# =============================================================================
# ADMIN ANALYTICS DASHBOARD
# =============================================================================
@app.get("/api/admin/analytics")
def admin_analytics(admin_user: Annotated[dict, Depends(get_current_admin)]):
    """Return dashboard metrics calculated from the current dataset and JSON stores."""
    dataset_path = PROJECT_ROOT / "data" / "Processed_527_datapoints.csv"
    dataset_summary = {"available": False, "row_count": 0, "feature_count": len(FEATURE_COLUMNS),
                       "target": "CO2 uptake (mmol/g)", "target_stats": None,
                       "target_distribution": [], "feature_quality": []}
    if dataset_path.is_file():
        try:
            frame = pd.read_csv(dataset_path)
            target = pd.to_numeric(frame.get("CO2 uptake (mmol/g)"), errors="coerce")
            dataset_summary["available"] = True
            dataset_summary["row_count"] = int(len(frame))
            dataset_summary["column_count"] = int(len(frame.columns))
            if target is not None and target.notna().any():
                dataset_summary["target_stats"] = {
                    "mean": round(float(target.mean()), 4),
                    "median": round(float(target.median()), 4),
                    "min": round(float(target.min()), 4),
                    "max": round(float(target.max()), 4),
                    "std": round(float(target.std()), 4),
                    "missing": int(target.isna().sum()),
                }
                counts = pd.cut(target.dropna(), bins=8, duplicates="drop").value_counts(sort=False)
                dataset_summary["target_distribution"] = [
                    {"range": str(interval), "count": int(count)}
                    for interval, count in counts.items()
                ]
            for column in FEATURE_COLUMNS:
                if column in frame.columns:
                    values = pd.to_numeric(frame[column], errors="coerce")
                    dataset_summary["feature_quality"].append({
                        "feature": column,
                        "missing": int(values.isna().sum()),
                        "mean": round(float(values.mean()), 4) if values.notna().any() else None,
                        "min": round(float(values.min()), 4) if values.notna().any() else None,
                        "max": round(float(values.max()), 4) if values.notna().any() else None,
                    })
        except Exception as exc:
            dataset_summary["error"] = f"Could not read the project dataset: {exc}"

    predictions = JSONStorage.get_predictions(limit=100000)
    jobs = _read_json(JSONStorage.JOBS_FILE, [])
    datasets = _read_json(JSONStorage.DATASETS_FILE, [])
    audit = JSONStorage.get_audit_logs(limit=8)
    versions = JSONStorage.get_model_versions()
    active = JSONStorage.get_active_model_version()
    return {
        "dataset": dataset_summary,
        "activity": {
            "prediction_count": len(predictions),
            "training_job_count": len(jobs),
            "completed_training_jobs": sum(j.get("status") == "completed" for j in jobs),
            "failed_training_jobs": sum(j.get("status") == "failed" for j in jobs),
            "uploaded_dataset_count": len(datasets),
        },
        "models": {
            "active_version": active.get("version") if active else None,
            "registered_versions": len(versions),
            "candidate_versions": sum(m.get("status") == "candidate" for m in versions),
        },
        "recent_activity": audit,
    }

# =============================================================================
# 2. AUTHENTICATION & USER MANAGEMENT
# =============================================================================
@app.post("/api/auth/register", response_model=UserResponse)
def register(request: RegisterRequest):
    """Register a new user account."""
    try:
        new_user = JSONStorage.create_user(request.username, request.password, "user")
        return new_user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """Log in and retrieve a secure JWT access token."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )
    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"]
        }
    }

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: Annotated[dict, Depends(get_current_user)]):
    """Retrieve profile of the currently authenticated user."""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"]
    }

# =============================================================================
# 3. PREDICTION & EXPLAINABILITY
# =============================================================================
@app.post("/api/predict")
@app.post("/predict")
def predict_endpoint(
    request: FeatureInput,
    current_user: Annotated[dict, Depends(get_current_user)]
):
    """Single sample prediction with real-time SHAP feature attribution."""
    values = {
        "surface_area": request.surface_area,
        "total_pore_volume": request.total_pore_volume,
        "micropore_volume": request.micropore_volume,
        "carbon": request.carbon,
        "hydrogen": request.hydrogen,
        "nitrogen": request.nitrogen,
        "oxygen": request.oxygen,
        "sulfur": request.sulfur,
        "temperature": request.temperature,
        "pressure": request.pressure,
    }

    pred_val, input_df = predict_single(values)
    base_val, shap_list = compute_shap(input_df)

    active_model = JSONStorage.get_active_model_version()
    version_name = active_model["version"] if active_model else "v1.0-ensemble"

    # Save to history audit trail
    JSONStorage.record_prediction(current_user["username"], values, pred_val, version_name)

    return {
        "predicted_co2_uptake_mmol_g": round(pred_val, 4),
        "unit": "mmol/g",
        "model": bundle["ensemble_strategy"],
        "model_version": version_name,
        "base_value": base_val,
        "shap_contributions": shap_list,
    }

@app.post("/api/predict/batch")
async def predict_batch_endpoint(
    file: UploadFile = File(...),
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    """Authenticated CSV batch prediction for any registered user."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 15MB limit.")

    try:
        result = predict_batch_csv(content, file.filename)
        # remove raw csv string from json response preview
        return {
            "filename": result["filename"],
            "rows_processed": result["rows_processed"],
            "mean_uptake_mmol_g": result["mean_uptake_mmol_g"],
            "min_uptake_mmol_g": result["min_uptake_mmol_g"],
            "max_uptake_mmol_g": result["max_uptake_mmol_g"],
            "predictions_preview": result["predictions_preview"],
            "download_url": "/api/predict/batch/download"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/batch/download")
async def download_batch_csv(
    file: UploadFile = File(...),
    current_user: Annotated[dict, Depends(get_current_user)] = None,
):
    """Authenticated CSV prediction download for any registered user."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 15MB limit.")
    try:
        result = predict_batch_csv(content, file.filename)
        safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name).strip("._") or "predictions.csv"
        return Response(
            content=result["csv_content"],
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="predicted_{safe_filename}"'}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/what-if")
def what_if_endpoint(
    request: WhatIfRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Authenticated side-by-side scenario simulation for any registered user."""
    v_base = {
        "surface_area": request.baseline.surface_area,
        "total_pore_volume": request.baseline.total_pore_volume,
        "micropore_volume": request.baseline.micropore_volume,
        "carbon": request.baseline.carbon,
        "hydrogen": request.baseline.hydrogen,
        "nitrogen": request.baseline.nitrogen,
        "oxygen": request.baseline.oxygen,
        "sulfur": request.baseline.sulfur,
        "temperature": request.baseline.temperature,
        "pressure": request.baseline.pressure,
    }
    v_mod = {
        "surface_area": request.modified.surface_area,
        "total_pore_volume": request.modified.total_pore_volume,
        "micropore_volume": request.modified.micropore_volume,
        "carbon": request.modified.carbon,
        "hydrogen": request.modified.hydrogen,
        "nitrogen": request.modified.nitrogen,
        "oxygen": request.modified.oxygen,
        "sulfur": request.modified.sulfur,
        "temperature": request.modified.temperature,
        "pressure": request.modified.pressure,
    }

    base_pred, df_base = predict_single(v_base)
    mod_pred, df_mod = predict_single(v_mod)

    delta = mod_pred - base_pred
    pct_change = (delta / base_pred * 100.0) if abs(base_pred) > 1e-6 else 0.0

    changed_features = []
    for col in FEATURE_COLUMNS:
        raw_b = df_base[col].iloc[0]
        raw_m = df_mod[col].iloc[0]

        if pd.notnull(raw_b) and pd.notnull(raw_m):
            fb = float(raw_b)
            fm = float(raw_m)
            if abs(fm - fb) > 1e-6:
                diff = fm - fb
                changed_features.append({
                    "feature": col,
                    "baseline_value": fb,
                    "modified_value": fm,
                    "difference": round(diff, 4),
                    "pct_change": round((diff / fb * 100.0) if abs(fb) > 1e-6 else 0.0, 2),
                })
        elif raw_b != raw_m:
            changed_features.append({
                "feature": col,
                "baseline_value": raw_b,
                "modified_value": raw_m,
                "difference": 0.0,
                "pct_change": 0.0,
            })

    return {
        "baseline_prediction": round(base_pred, 3),
        "modified_prediction": round(mod_pred, 3),
        "delta_mmol_g": round(delta, 3),
        "percentage_change": round(pct_change, 2),
        "unit": "mmol/g",
        "changed_features": changed_features,
    }

@app.get("/api/history")
def get_prediction_history(
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = 50
):
    """Users see their own history; administrators can review all users' history."""
    safe_limit = max(1, min(limit, 200))
    username = None if current_user.get("role") == "admin" else current_user["username"]
    return {
        "history": JSONStorage.get_predictions(username, limit=safe_limit)
    }

@app.post("/api/explain")
def explain_endpoint(
    request: FeatureInput | None = None,
    current_user: Annotated[dict | None, Depends(get_current_user_optional)] = None,
):
    """Public global feature rankings; local prediction explanations require login."""
    if request is not None and (request.surface_area is not None or request.pressure is not None) and current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please log in to request a local prediction explanation.")
    global_importance = []
    shap_csv = RESULTS_DIR / "shap_feature_importance.csv"
    if shap_csv.exists():
        df_imp = pd.read_csv(shap_csv)
        global_importance = df_imp.to_dict(orient="records")

    local_shap = []
    base_val = None
    if request is not None and (request.surface_area is not None or request.pressure is not None):
        values = {
            "surface_area": request.surface_area,
            "total_pore_volume": request.total_pore_volume,
            "micropore_volume": request.micropore_volume,
            "carbon": request.carbon,
            "hydrogen": request.hydrogen,
            "nitrogen": request.nitrogen,
            "oxygen": request.oxygen,
            "sulfur": request.sulfur,
            "temperature": request.temperature,
            "pressure": request.pressure,
        }
        _, input_df = predict_single(values)
        base_val, local_shap = compute_shap(input_df)

    return {
        "global_importance": global_importance,
        "base_value": base_val,
        "local_shap": local_shap,
        "note": "SHAP feature values represent cooperative game-theoretic contributions, not physical causality.",
    }

@app.get("/api/models")
def models_endpoint():
    """Retrieve benchmark test evaluation metrics and ensemble weights."""
    models_metrics = []
    results_csv = RESULTS_DIR / "final_test_results.csv"
    if results_csv.exists():
        df_res = pd.read_csv(results_csv)
        models_metrics = df_res.to_dict(orient="records")

    weights_list = []
    weights_csv = RESULTS_DIR / "ensemble_weights.csv"
    if weights_csv.exists():
        df_w = pd.read_csv(weights_csv)
        weights_list = df_w.to_dict(orient="records")

    return {
        "evaluation_metrics": models_metrics,
        "ensemble_weights": weights_list,
        "selected_model": bundle["ensemble_strategy"],
        "features": FEATURE_COLUMNS,
    }

# =============================================================================
# 4. ADMIN & MODEL LIFECYCLE MANAGEMENT
# =============================================================================
@app.post("/api/admin/datasets")
async def upload_retraining_dataset(
    file: UploadFile = File(...),
    admin_user: Annotated[dict, Depends(get_current_admin)] = None
):
    """Admin: Upload and validate a supervised dataset for retraining."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are accepted.")

    content = await file.read()
    try:
        report = validate_and_save_dataset(content, file.filename, admin_user["username"])
        return {
            "status": "success",
            "message": f"Dataset '{file.filename}' successfully validated and approved.",
            "report": report
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/admin/train")
def trigger_training(
    request: TrainJobStartRequest,
    admin_user: Annotated[dict, Depends(get_current_admin)] = None
):
    """Admin: Trigger an asynchronous background model retraining job."""
    try:
        job = start_training_job(request.dataset_name, admin_user["username"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "success",
        "message": "Background training job initiated.",
        "job": job
    }

@app.get("/api/admin/training/{job_id}")
def check_training_status(
    job_id: str,
    admin_user: Annotated[dict, Depends(get_current_admin)] = None
):
    """Admin: Inspect progress, metrics, and completion state of a training job."""
    job = JSONStorage.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return job

@app.get("/api/admin/models")
def list_admin_models(admin_user: Annotated[dict, Depends(get_current_admin)] = None):
    """Admin: List all model versions, deployment statuses, and metrics."""
    return {
        "models": JSONStorage.get_model_versions()
    }

@app.post("/api/admin/models/{version}/activate")
def activate_model_version(
    version: str,
    admin_user: Annotated[dict, Depends(get_current_admin)] = None
):
    """Admin: Atomically activate a candidate model version."""
    try:
        model_record = next((m for m in JSONStorage.get_model_versions() if m.get("version") == version), None)
        if not model_record or not model_record.get("artifact_path"):
            raise ValueError("This model version has no loadable model artifact.")
        activate_prediction_bundle(_resolve_model_artifact_path(model_record["artifact_path"]))
        activated = JSONStorage.activate_model_version(version, admin_user["username"])
        return {
            "status": "success",
            "message": f"Model version '{version}' is now active.",
            "model": activated
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/admin/models/{version}/rollback")
def rollback_model_version(
    version: str,
    admin_user: Annotated[dict, Depends(get_current_admin)] = None
):
    """Admin: Roll back active model to a previously approved version."""
    try:
        model_record = next((m for m in JSONStorage.get_model_versions() if m.get("version") == version), None)
        if not model_record or not model_record.get("artifact_path"):
            raise ValueError("This model version has no loadable model artifact.")
        activate_prediction_bundle(_resolve_model_artifact_path(model_record["artifact_path"]))
        rolled_back = JSONStorage.rollback_model_version(version, admin_user["username"])
        return {
            "status": "success",
            "message": f"Successfully rolled back to version '{version}'.",
            "model": rolled_back
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/admin/audit-logs")
def view_audit_logs(
    admin_user: Annotated[dict, Depends(get_current_admin)] = None,
    limit: int = 50
):
    """Admin: View system audit trail of administrative actions."""
    return {
        "audit_logs": JSONStorage.get_audit_logs(limit=limit)
    }

# =============================================================================
# 5. STATIC FILES & FRONTEND MOUNTING
# =============================================================================
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
