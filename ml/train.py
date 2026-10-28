
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)
from ml.evaluation import evaluate_regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "candidates"
SPLIT_FILE = PROJECT_ROOT / "artifacts" / "splits" / "split_assignments.csv"


def build_model_pipelines(random_state=42):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    estimators = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
    }

    return {
        name: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
        for name, estimator in estimators.items()
    }


def train_candidates(dataset):
    # Validate and remove exact duplicate rows, preserving the dataset's row order.
    cleaned, report = validate_training_dataset(
        dataset, remove_duplicates=True
    )

    if not report.valid:
        raise ValueError(f"Dataset validation failed: {report.errors}")

    if len(cleaned) < 10:
        raise ValueError("At least 10 unique rows are required.")

    # Load the project's original split assignments.
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Original split file not found: {SPLIT_FILE}"
        )

    splits = pd.read_csv(SPLIT_FILE)

    if list(splits.columns) != ["row_id", "split"]:
        raise ValueError(
            "Unexpected split file format. Expected columns: row_id, split."
        )

    if len(splits) != len(cleaned):
        raise ValueError(
            f"Split file has {len(splits)} rows, but the deduplicated "
            f"dataset has {len(cleaned)} rows."
        )

    if splits["row_id"].tolist() != list(range(len(cleaned))):
        raise ValueError(
            "Split row IDs must be consecutive and match dataset row order."
        )

    if splits["split"].isna().any():
        raise ValueError("Split assignments contain missing labels.")

    allowed_splits = {"train", "validation", "test"}
    if set(splits["split"].unique()) != allowed_splits:
        raise ValueError(
            f"Expected split labels {allowed_splits}; "
            f"found {set(splits['split'].unique())}."
        )

    # Select development rows only. The test rows are never fitted or scored here.
    train_mask = splits["split"].eq("train").to_numpy()
    validation_mask = splits["split"].eq("validation").to_numpy()

    X = cleaned[FEATURE_COLUMNS]
    y = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise").astype(float)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation splits must both be non-empty.")

    if y_train.nunique() < 2:
        raise ValueError("Training targets must contain at least two distinct values.")

    models = build_model_pipelines(random_state=42)
    fitted_models = {}
    model_results = []

    for name, pipeline in models.items():
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)

        predictions = np.asarray(fitted.predict(X_val), dtype=float)

        if not np.isfinite(predictions).all():
            raise ValueError(f"{name} produced non-finite validation predictions.")

        metrics = evaluate_regression(y_val, predictions)

        fitted_models[name] = fitted
        model_results.append({
            "model": name,
            **metrics,
        })

    model_results.sort(key=lambda item: item["RMSE"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_path = CANDIDATE_DIR / run_id
    candidate_path.mkdir(parents=True, exist_ok=False)

    for name, model in fitted_models.items():
        safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        joblib.dump(model, candidate_path / f"{safe_name}.joblib")

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "activated": False,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "rows_received": report.rows_received,
        "rows_after_duplicate_removal": len(cleaned),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows_reserved": int(splits["split"].eq("test").sum()),
        "random_state": 42,
        "selection_metric": "RMSE",
        "models": model_results,
        "best_validation_model": model_results[0]["model"],
        "test_set_evaluated": False,
        "note": (
            "Baseline comparison using the original train/validation split. "
            "The original test set was not used for fitting or model selection. "
            "Validation metrics must not be compared directly with test metrics."
        ),
    }

    with (candidate_path / "candidate_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return {
        "candidate_directory": str(candidate_path),
        "metadata": metadata,
        "validation_report": report.to_dict(),
    }
