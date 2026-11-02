"""Training workflow for datasets uploaded through the admin application.

This is intentionally separate from the frozen research pipeline in ml/train.py:
admin uploads receive a fresh train/validation split and never modify the
original train/validation/test assignments or reported final-test results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import uuid

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from ml.data_validation import FEATURE_COLUMNS, TARGET_COLUMN, validate_training_dataset
from ml.evaluation import evaluate_regression
from ml.train import build_model_pipelines
from backend.services.storage_service import STORAGE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADMIN_CANDIDATE_DIR = STORAGE_DIR / "admin_candidates"
UPLOAD_DATASETS_DIR = STORAGE_DIR / "datasets"


def train_uploaded_dataset(dataset_path: str | Path, random_state: int = 42) -> dict:
    """Fit six regressors, optimize nonnegative validation weights, save candidate."""
    dataset_path = Path(dataset_path).resolve()
    try:
        dataset_path.relative_to(UPLOAD_DATASETS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Training data must be an uploaded dataset in backend storage.") from exc

    df = pd.read_csv(dataset_path)
    cleaned, report = validate_training_dataset(df, remove_duplicates=True)
    if not report.valid:
        raise ValueError("Dataset validation failed: " + "; ".join(report.errors))
    if len(cleaned) < 20:
        raise ValueError("At least 20 unique rows are required for admin retraining.")
    if cleaned[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The target must contain at least two distinct values.")

    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN].astype(float)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipelines = build_model_pipelines(random_state=random_state)
    fitted_models = {}
    validation_predictions = []
    individual_metrics = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        imputed_validation = pipeline.named_steps["imputer"].transform(X_val)
        predictions = np.asarray(
            pipeline.named_steps["model"].predict(imputed_validation), dtype=float
        )
        if not np.isfinite(predictions).all():
            raise ValueError(f"{name} produced non-finite validation predictions.")
        fitted_models[name] = pipeline
        validation_predictions.append(predictions)
        individual_metrics[name] = evaluate_regression(y_val, predictions)

    prediction_matrix = np.column_stack(validation_predictions)
    n_models = prediction_matrix.shape[1]
    equal_weights = np.full(n_models, 1.0 / n_models)

    def objective(weights):
        residual = prediction_matrix @ weights - y_val.to_numpy()
        return float(np.sqrt(np.mean(residual ** 2))) + 0.01 * float(
            np.sum((weights - equal_weights) ** 2)
        )

    optimized = minimize(
        objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1500, "ftol": 1e-10},
    )
    if optimized.success:
        weights = np.clip(optimized.x, 0.0, 1.0)
        weights /= weights.sum()
    else:
        # A valid equal-weight ensemble is preferable to inventing weights.
        weights = equal_weights

    model_order = list(pipelines.keys())
    weight_map = {name: float(weight) for name, weight in zip(model_order, weights)}
    ensemble_predictions = prediction_matrix @ weights
    ensemble_metrics = evaluate_regression(y_val, ensemble_predictions)

    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    candidate_dir = ADMIN_CANDIDATE_DIR / run_id
    candidate_dir.mkdir(parents=True, exist_ok=False)

    bundle = {
        "project_name": "Interpretable Ensemble Learning for CO2 Adsorption on Biomass-Derived Porous Carbons",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "model_order": model_order,
        "models": fitted_models,
        "ensemble_weights": weight_map,
        "ensemble_strategy": "admin_validation_optimized_weighted_ensemble",
        "prediction_unit": "mmol/g",
        "preprocessing": "Median imputation inside each fitted model pipeline",
        "random_state": random_state,
    }
    artifact_path = candidate_dir / "co2_adsorption_ensemble.joblib"
    joblib.dump(bundle, artifact_path)

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = {
        "version": "candidate-" + run_id,
        "name": "Admin-retrained ensemble",
        "status": "candidate",
        "created_at": created.isoformat(),
        "artifact_path": str(artifact_path.resolve()),
        "dataset_filename": dataset_path.name,
        "dataset_sha256": dataset_hash,
        "rows_received": int(report.rows_received),
        "rows_after_duplicate_removal": int(len(cleaned)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_set_evaluated": False,
        "metrics": ensemble_metrics,
        "weights": weight_map,
        "individual_validation_metrics": individual_metrics,
        "validation_method": "Random 80/20 train/validation split; random_state=42",
        "note": "Candidate metrics are validation-only. This run does not evaluate the frozen research test set and must not be presented as an independent test result.",
    }
    (candidate_dir / "candidate_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"metadata": metadata, "validation_report": report.to_dict()}
