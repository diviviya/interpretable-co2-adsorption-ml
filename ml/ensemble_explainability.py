
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_ROOT / "artifacts" / "splits" / "split_assignments.csv"

DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "candidates"
    / "20261028T094742Z_ensemble"
)


def _get_shap_array(values):
    """Normalize regression SHAP output to (rows, features)."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected a single regression output.")
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

    return values


def explain_ensemble(
    dataset,
    candidate_directory=DEFAULT_CANDIDATE,
    max_background_rows=100,
):
    candidate_directory = Path(candidate_directory)
    bundle_path = candidate_directory / "co2_adsorption_ensemble.joblib"

    if not bundle_path.exists():
        raise FileNotFoundError(f"Candidate bundle not found: {bundle_path}")

    if max_background_rows < 1:
        raise ValueError("max_background_rows must be positive.")

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
        raise ValueError("Split assignments do not match the dataset.")

    if splits["split"].isna().any():
        raise ValueError("Split assignments contain missing labels.")

    train_mask = splits["split"].eq("train").to_numpy()
    validation_mask = splits["split"].eq("validation").to_numpy()

    X_train = cleaned.loc[train_mask, FEATURE_COLUMNS]
    X_val = cleaned.loc[validation_mask, FEATURE_COLUMNS]

    if X_train.empty or X_val.empty:
        raise ValueError("Training and validation splits must be non-empty.")

    bundle = joblib.load(bundle_path)

    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Candidate feature columns do not match the dataset.")

    models = bundle["models"]
    weights = bundle["ensemble_weights"]
    model_order = bundle["model_order"]

    if set(models) != set(weights) or set(models) != set(model_order):
        raise ValueError("Bundle models, weights and model order do not match.")

    weight_values = np.array([weights[name] for name in model_order], dtype=float)

    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0).any()
        or not np.isclose(weight_values.sum(), 1.0)
    ):
        raise ValueError("Invalid ensemble weights.")

    # Retained for reproducible metadata and compatibility.
    # Tree-path-dependent SHAP does not use an explicit background dataset.
    background = X_train.sample(
        n=min(max_background_rows, len(X_train)),
        random_state=42,
    )

    n_rows = len(X_val)
    n_features = len(FEATURE_COLUMNS)

    ensemble_shap = np.zeros((n_rows, n_features), dtype=float)
    ensemble_expected_value = 0.0
    ensemble_prediction = np.zeros(n_rows, dtype=float)
    per_model_checks = {}

    for name in model_order:
        pipeline = models[name]

        if not hasattr(pipeline, "named_steps"):
            raise TypeError(f"{name} is not a fitted preprocessing pipeline.")

        imputer = pipeline.named_steps.get("imputer")
        estimator = pipeline.named_steps.get("model")

        if imputer is None or estimator is None:
            raise ValueError(
                f"{name} pipeline must contain 'imputer' and 'model' steps."
            )

        # Explain the estimator on the exact imputed inputs it receives.
        validation_transformed = imputer.transform(X_val)

        explainer = shap.TreeExplainer(
            estimator,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        values = _get_shap_array(
            explainer.shap_values(validation_transformed)
        )

        if values.shape != (n_rows, n_features):
            raise ValueError(
                f"{name} SHAP shape {values.shape} does not match "
                f"expected {(n_rows, n_features)}."
            )

        expected = np.asarray(
            explainer.expected_value, dtype=float
        ).reshape(-1)

        if expected.size != 1:
            raise ValueError(f"{name} has an unexpected SHAP output shape.")

        expected = float(expected[0])
        prediction = np.asarray(pipeline.predict(X_val), dtype=float)

        if not np.isfinite(values).all() or not np.isfinite(prediction).all():
            raise ValueError(
                f"{name} produced non-finite explanations or predictions."
            )

        # Check local additivity for each component model.
        reconstructed = expected + values.sum(axis=1)
        max_error = float(np.max(np.abs(reconstructed - prediction)))

        if max_error > 1e-3:
            raise ValueError(
                f"{name} SHAP additivity check failed: "
                f"maximum absolute error = {max_error:.6g}"
            )

        weight = float(weights[name])
        ensemble_shap += weight * values
        ensemble_expected_value += weight * expected
        ensemble_prediction += weight * prediction

        per_model_checks[name] = {
            "weight": weight,
            "expected_value": expected,
            "max_absolute_additivity_error": max_error,
        }

    # Verify that the weighted SHAP values explain the actual ensemble.
    reconstructed_ensemble = (
        ensemble_expected_value + ensemble_shap.sum(axis=1)
    )
    ensemble_errors = np.abs(
        reconstructed_ensemble - ensemble_prediction
    )
    max_ensemble_error = float(ensemble_errors.max())

    if max_ensemble_error > 1e-3:
        raise ValueError(
            "Ensemble SHAP additivity check failed: "
            f"maximum absolute error = {max_ensemble_error:.6g}"
        )

    output_dir = candidate_directory / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One row per validation observation, one column per input feature.
    shap_df = pd.DataFrame(
        ensemble_shap,
        columns=FEATURE_COLUMNS,
        index=X_val.index,
    )
    shap_df.index.name = "dataset_row_id"
    shap_df.to_csv(output_dir / "ensemble_shap_values.csv")

    # Save the explained inputs and corresponding ensemble predictions.
    predictions_df = X_val.copy()
    predictions_df["ensemble_prediction"] = ensemble_prediction
    predictions_df["shap_expected_value"] = ensemble_expected_value
    predictions_df["shap_reconstructed_prediction"] = reconstructed_ensemble
    predictions_df["absolute_additivity_error"] = ensemble_errors
    predictions_df.index.name = "dataset_row_id"
    predictions_df.to_csv(output_dir / "ensemble_shap_predictions.csv")

    mean_abs_shap = np.mean(np.abs(ensemble_shap), axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_absolute_shap": mean_abs_shap,
    }).sort_values("mean_absolute_shap", ascending=False)

    importance_df.to_csv(
        output_dir / "ensemble_shap_feature_importance.csv",
        index=False,
    )

    metadata = {
        "candidate_directory": str(candidate_directory.resolve()),
        "target_column": TARGET_COLUMN,
        "prediction_unit": bundle["prediction_unit"],
        "explained_split": "validation",
        "explained_rows": int(len(X_val)),
        "background_split": "train",
        "background_rows": int(len(background)),
        "model_order": model_order,
        "ensemble_weights": weights,
        "ensemble_expected_value": float(ensemble_expected_value),
        "max_absolute_ensemble_additivity_error": max_ensemble_error,
        "per_model_checks": per_model_checks,
        "shap_method": (
            "Weighted sum of model-specific TreeSHAP values using "
            "tree_path_dependent explanations. No explicit background "
            "dataset is passed to TreeExplainer."
        ),
        "note": (
            "These are ensemble-level explanations for the saved candidate. "
            "The original test set was not used for SHAP background selection."
        ),
    }

    with (output_dir / "ensemble_shap_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return {
        "output_directory": str(output_dir),
        "metadata": metadata,
        "feature_importance": importance_df,
    }
