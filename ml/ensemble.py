
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)
from ml.evaluation import evaluate_regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "candidates"
SPLIT_FILE = PROJECT_ROOT / "artifacts" / "splits" / "split_assignments.csv"

MODEL_NAMES = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def _safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_candidate_models(candidate_dir):
    """Load the six saved models from one candidate directory."""
    models = {}

    for name in MODEL_NAMES:
        path = candidate_dir / f"{_safe_name(name)}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    if set(models) != set(MODEL_NAMES):
        missing = sorted(set(MODEL_NAMES) - set(models))
        raise ValueError(
            f"Candidate {candidate_dir.name} is missing models: {missing}"
        )

    return models


def _find_latest_candidate(expected_status):
    matches = []

    for path in CANDIDATE_DIR.iterdir():
        if not path.is_dir():
            continue

        metadata_path = path / "candidate_metadata.json"
        if not metadata_path.exists():
            continue

        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        # The tuned candidate uses status "tuned_candidate".
        if metadata.get("status") == expected_status:
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No candidate with status {expected_status!r} was found."
        )

    return sorted(matches, key=lambda path: path.name)[-1]



def _validation_data(dataset):
    cleaned, report = validate_training_dataset(
        dataset, remove_duplicates=True
    )

    if not report.valid:
        raise ValueError(f"Dataset validation failed: {report.errors}")

    splits = pd.read_csv(SPLIT_FILE)

    if (
        list(splits.columns) != ["row_id", "split"]
        or len(splits) != len(cleaned)
        or splits["row_id"].tolist() != list(range(len(cleaned)))
    ):
        raise ValueError(
            "Split assignments do not match the deduplicated dataset."
        )

    if splits["split"].isna().any():
        raise ValueError("Split assignments contain missing labels.")

    if set(splits["split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError("Unexpected split labels.")

    mask = splits["split"].eq("validation").to_numpy()

    X_val = cleaned.loc[mask, FEATURE_COLUMNS]
    y_val = pd.to_numeric(
        cleaned.loc[mask, TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=float)

    if len(X_val) == 0:
        raise ValueError("Validation split is empty.")

    return X_val, y_val, splits, report


def _metrics_for_predictions(y_true, predictions):
    return evaluate_regression(y_true, predictions)


def construct_ensemble(dataset):
    """
    Compare baseline and tuned individual models, then compare:
    - equal weights across the tuned models
    - validation-RMSE-optimized weights across the tuned models

    The original test rows are never used here.
    """
    X_val, y_val, splits, report = _validation_data(dataset)

    baseline_dir = _find_latest_candidate("candidate")
    tuned_dir = _find_latest_candidate("tuned_candidate")

    baseline_models = _load_candidate_models(baseline_dir)
    tuned_models = _load_candidate_models(tuned_dir)

    # Compare baseline and tuned variants using identical validation rows.
    individual_results = []
    tuned_predictions = {}

    for name in MODEL_NAMES:
        baseline_pred = np.asarray(
            baseline_models[name].predict(X_val), dtype=float
        )
        tuned_pred = np.asarray(
            tuned_models[name].predict(X_val), dtype=float
        )

        if (
            not np.isfinite(baseline_pred).all()
            or not np.isfinite(tuned_pred).all()
        ):
            raise ValueError(f"{name} produced non-finite predictions.")

        baseline_metrics = _metrics_for_predictions(y_val, baseline_pred)
        tuned_metrics = _metrics_for_predictions(y_val, tuned_pred)

        individual_results.append({
            "model": name,
            "baseline_validation": baseline_metrics,
            "tuned_validation": tuned_metrics,
        })

        tuned_predictions[name] = tuned_pred

    prediction_matrix = np.column_stack([
        tuned_predictions[name] for name in MODEL_NAMES
    ])

    # Strategy 1: equal weights across all six tuned models.
    equal_weights = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES))
    equal_predictions = prediction_matrix @ equal_weights
    equal_metrics = _metrics_for_predictions(y_val, equal_predictions)

    # Strategy 2: optimize non-negative weights summing to one.
    # A small regularization term discourages extreme weights.
    def objective(weights):
        residual = prediction_matrix @ weights - y_val
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        regularization = 0.01 * float(
            np.sum((weights - equal_weights) ** 2)
        )
        return rmse + regularization

    result = minimize(
        objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(MODEL_NAMES),
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1.0,
            }
        ],
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Weight optimization failed: {result.message}"
        )

    optimized_weights = np.clip(result.x, 0.0, 1.0)
    optimized_weights /= optimized_weights.sum()

    optimized_predictions = prediction_matrix @ optimized_weights
    optimized_metrics = _metrics_for_predictions(
        y_val, optimized_predictions
    )

    # Choose the simpler strategy if optimized weights do not improve RMSE.
    if optimized_metrics["RMSE"] < equal_metrics["RMSE"]:
        selected_strategy = "optimized_nonnegative_weights"
        selected_weights = optimized_weights
        selected_metrics = optimized_metrics
    else:
        selected_strategy = "equal_weights"
        selected_weights = equal_weights
        selected_metrics = equal_metrics

    weights_dict = {
        name: float(weight)
        for name, weight in zip(MODEL_NAMES, selected_weights)
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = CANDIDATE_DIR / f"{run_id}_ensemble"
    output_dir.mkdir(parents=True, exist_ok=False)

    # Save a deployment-compatible candidate bundle.
    bundle = {
        "project_name": (
            "Interpretable Ensemble Learning for CO2 Adsorption "
            "on Biomass-Derived Porous Carbons"
        ),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "model_order": MODEL_NAMES,
        "models": tuned_models,
        "ensemble_weights": weights_dict,
        "ensemble_strategy": selected_strategy,
        "prediction_unit": "mmol/g",
        "preprocessing": (
            "Median imputation inside each fitted model pipeline"
        ),
        "random_state": 42,
    }

    joblib.dump(
        bundle,
        output_dir / "co2_adsorption_ensemble.joblib",
    )

    comparison = {
        "baseline_candidate": baseline_dir.relative_to(PROJECT_ROOT).as_posix(),
        "tuned_candidate": tuned_dir.relative_to(PROJECT_ROOT).as_posix(),
        "individual_models": individual_results,
        "equal_weight_ensemble": {
            "weights": {
                name: float(weight)
                for name, weight in zip(MODEL_NAMES, equal_weights)
            },
            **equal_metrics,
        },
        "optimized_weight_ensemble": {
            "weights": {
                name: float(weight)
                for name, weight in zip(MODEL_NAMES, optimized_weights)
            },
            **optimized_metrics,
        },
        "selected_strategy": selected_strategy,
        "selected_weights": weights_dict,
        "selected_validation_metrics": selected_metrics,
    }

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ensemble_candidate",
        "activated": False,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(splits["split"].eq("train").sum()),
        "validation_rows": int(splits["split"].eq("validation").sum()),
        "test_rows_reserved": int(splits["split"].eq("test").sum()),
        "test_set_evaluated": False,
        "selected_strategy": selected_strategy,
        "selected_weights": weights_dict,
        "selected_validation_metrics": selected_metrics,
        "comparison": comparison,
        "note": (
            "Candidate only. Weights were selected using validation data. "
            "The original test set was not used. Independent test evaluation "
            "is required before any deployment decision."
        ),
    }

    with (output_dir / "candidate_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return {
        "candidate_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "metadata": metadata,
        "validation_report": report.to_dict(),
    }
