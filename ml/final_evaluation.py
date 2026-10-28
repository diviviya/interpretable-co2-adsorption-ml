
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)
from ml.evaluation import evaluate_regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_ROOT / "artifacts" / "splits" / "split_assignments.csv"


def evaluate_candidate_on_test(dataset, candidate_directory):
    candidate_directory = Path(candidate_directory)

    bundle_path = candidate_directory / "co2_adsorption_ensemble.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Candidate bundle not found: {bundle_path}")

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

    test_mask = splits["split"].eq("test").to_numpy()
    if not test_mask.any():
        raise ValueError("The original test split is empty.")

    X_test = cleaned.loc[test_mask, FEATURE_COLUMNS]
    y_test = pd.to_numeric(
        cleaned.loc[test_mask, TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=float)

    bundle = joblib.load(bundle_path)

    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Candidate feature columns do not match the dataset.")

    models = bundle["models"]
    weights = bundle["ensemble_weights"]

    if set(models) != set(weights):
        raise ValueError("Model names and ensemble weight names do not match.")

    weight_values = np.asarray(list(weights.values()), dtype=float)
    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0).any()
        or not np.isclose(weight_values.sum(), 1.0)
    ):
        raise ValueError("Ensemble weights are invalid.")

    predictions_by_model = {
        name: np.asarray(model.predict(X_test), dtype=float)
        for name, model in models.items()
    }

    if any(
        not np.isfinite(predictions).all()
        for predictions in predictions_by_model.values()
    ):
        raise ValueError("A model produced non-finite test predictions.")

    y_pred = sum(
        weights[name] * predictions_by_model[name]
        for name in models
    )

    metrics = evaluate_regression(y_test, y_pred)

    result = {
        "candidate_directory": str(candidate_directory.resolve()),
        "target_column": TARGET_COLUMN,
        "prediction_unit": bundle["prediction_unit"],
        "test_rows": int(test_mask.sum()),
        "test_set_evaluated": True,
        "ensemble_strategy": bundle["ensemble_strategy"],
        "ensemble_weights": weights,
        "metrics": metrics,
        "note": (
            "Final evaluation of the selected candidate on the original "
            "held-out test split. Do not use these results to tune weights "
            "or select models."
        ),
    }

    output_path = candidate_directory / "final_test_evaluation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return result
