
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)
from ml.evaluation import evaluate_regression
from ml.train import build_model_pipelines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_ROOT / "artifacts" / "splits" / "split_assignments.csv"
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "candidates"

N_TRIALS = 30
CV_FOLDS = 5
RANDOM_STATE = 42


def _suggest_params(trial, model_name):
    """Return model-specific hyperparameters for an Optuna trial."""

    if model_name == "Random Forest":
        return {
            "model__n_estimators": trial.suggest_int(
                "n_estimators", 200, 600, step=100
            ),
            "model__max_depth": trial.suggest_categorical(
                "max_depth", [None, 5, 10, 15, 20]
            ),
            "model__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            "model__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "model__max_features": trial.suggest_categorical(
                "max_features", [1.0, "sqrt", 0.7]
            ),
        }

    if model_name == "Extra Trees":
        return {
            "model__n_estimators": trial.suggest_int(
                "n_estimators", 200, 600, step=100
            ),
            "model__max_depth": trial.suggest_categorical(
                "max_depth", [None, 5, 10, 15, 20]
            ),
            "model__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            "model__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "model__max_features": trial.suggest_categorical(
                "max_features", [1.0, "sqrt", 0.7]
            ),
        }

    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": trial.suggest_int(
                "n_estimators", 100, 500, step=50
            ),
            "model__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "model__max_depth": trial.suggest_int("max_depth", 2, 6),
            "model__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "model__subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int(
                "n_estimators", 150, 600, step=50
            ),
            "model__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "model__max_depth": trial.suggest_int("max_depth", 2, 8),
            "model__min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 10
            ),
            "model__subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "model__colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "model__reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
        }

    if model_name == "LightGBM":
        return {
            "model__n_estimators": trial.suggest_int(
                "n_estimators", 150, 600, step=50
            ),
            "model__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "model__num_leaves": trial.suggest_int("num_leaves", 8, 40),
            "model__max_depth": trial.suggest_int("max_depth", 3, 10),
            "model__min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 30
            ),
            "model__subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "model__colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "model__reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
        }

    if model_name == "CatBoost":
        return {
            "model__iterations": trial.suggest_int(
                "iterations", 200, 600, step=50
            ),
            "model__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "model__depth": trial.suggest_int("depth", 4, 8),
            "model__l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0
            ),
        }

    raise ValueError(f"Unsupported model: {model_name}")


def tune_models(dataset, n_trials=N_TRIALS, cv_folds=CV_FOLDS):
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

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

    train_mask = splits["split"].eq("train").to_numpy()
    validation_mask = splits["split"].eq("validation").to_numpy()

    X = cleaned[FEATURE_COLUMNS]
    y = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise").astype(float)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]

    if len(X_train) < cv_folds:
        raise ValueError("Not enough training rows for the requested CV folds.")

    if y_train.nunique() < 2:
        raise ValueError("Training targets must contain at least two distinct values.")

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    baseline_pipelines = build_model_pipelines(
        random_state=RANDOM_STATE
    )

    # Sequential trials keep compute use predictable.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_path = CANDIDATE_DIR / f"{run_id}_tuned"
    candidate_path.mkdir(parents=True, exist_ok=False)

    tuned_models = {}
    model_results = []
    study_summaries = {}

    for model_name, base_pipeline in baseline_pipelines.items():
        print(f"\nTuning {model_name} ({n_trials} trials)...")

        def objective(trial):
            params = _suggest_params(trial, model_name)
            pipeline = clone(base_pipeline).set_params(**params)

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )

            return float(-np.mean(scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE
            ),
        )
        study.optimize(objective, n_trials=n_trials)

        best_pipeline = clone(base_pipeline).set_params(
            **{
                f"model__{key}": value
                for key, value in study.best_params.items()
            }
        )
        best_pipeline.fit(X_train, y_train)

        validation_predictions = np.asarray(
            best_pipeline.predict(X_val), dtype=float
        )

        if not np.isfinite(validation_predictions).all():
            raise ValueError(
                f"{model_name} produced non-finite validation predictions."
            )

        metrics = evaluate_regression(y_val, validation_predictions)
        tuned_models[model_name] = best_pipeline

        model_results.append({
            "model": model_name,
            "best_cv_rmse": float(study.best_value),
            **metrics,
            "best_params": study.best_params,
        })

        study_summaries[model_name] = {
            "best_cv_rmse": float(study.best_value),
            "best_params": study.best_params,
            "trials_completed": len(study.trials),
        }

        safe_name = re.sub(
            r"[^a-z0-9]+", "_", model_name.lower()
        ).strip("_")

        joblib.dump(
            best_pipeline,
            candidate_path / f"{safe_name}.joblib",
        )

        study.trials_dataframe().to_csv(
            candidate_path / f"{safe_name}_trials.csv",
            index=False,
        )

        print(
            f"  Best CV RMSE: {study.best_value:.4f} | "
            f"Validation RMSE: {metrics['RMSE']:.4f}"
        )

    model_results.sort(key=lambda item: item["RMSE"])

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "tuned_candidate",
        "activated": False,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows_reserved": int(splits["split"].eq("test").sum()),
        "cv_folds": cv_folds,
        "n_trials_per_model": n_trials,
        "selection_metric": "validation RMSE",
        "models": model_results,
        "studies": study_summaries,
        "best_validation_model": model_results[0]["model"],
        "test_set_evaluated": False,
        "note": (
            "Optuna tuning used cross-validation on training rows only. "
            "Validation rows were used for model comparison. "
            "The original test rows were not used."
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
