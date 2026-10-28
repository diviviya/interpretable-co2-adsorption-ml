
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Repository root: the directory containing this file's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "co2_adsorption_ensemble.joblib"
)


class PredictionError(ValueError):
    """Raised when prediction inputs are invalid."""


class CO2AdsorptionPredictor:
    """Load and run the saved CO2 adsorption ensemble."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model bundle not found: {model_path}"
            )

        self.bundle = joblib.load(model_path)

        required_keys = {
            "feature_columns",
            "model_order",
            "models",
            "ensemble_weights",
            "prediction_unit",
        }

        missing_keys = required_keys - self.bundle.keys()
        if missing_keys:
            raise ValueError(
                f"Model bundle is missing keys: {sorted(missing_keys)}"
            )

        self.feature_columns = self.bundle["feature_columns"]
        self.model_order = self.bundle["model_order"]
        self.models = self.bundle["models"]
        self.weights = self.bundle["ensemble_weights"]
        self.prediction_unit = self.bundle["prediction_unit"]

        if set(self.model_order) != set(self.models):
            raise ValueError(
                "Model order does not match the models in the bundle."
            )

        if set(self.model_order) != set(self.weights):
            raise ValueError(
                "Model names do not match the ensemble weight keys."
            )

        weight_values = np.array(
            [self.weights[name] for name in self.model_order],
            dtype=float,
        )

        if (
            not np.isfinite(weight_values).all()
            or np.any(weight_values < 0)
            or not np.isclose(weight_values.sum(), 1.0)
        ):
            raise ValueError(
                "Ensemble weights must be finite, non-negative, "
                "and sum to 1."
            )


    def _prepare_input(self, values: dict) -> pd.DataFrame:
        """Validate features while allowing missing values for imputation."""

        if not isinstance(values, dict):
            raise PredictionError(
                "Input must be a dictionary of feature names and values."
            )

        extra = [
            feature
            for feature in values
            if feature not in self.feature_columns
        ]

        if extra:
            raise PredictionError(
                f"Unexpected features: {extra}"
            )

        row = {}

        for feature in self.feature_columns:
            value = values.get(feature)

            if value is None:
                row[feature] = np.nan
                continue

            if isinstance(value, bool):
                raise PredictionError(
                    f"'{feature}' must be numeric, not boolean."
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise PredictionError(
                    f"'{feature}' must be numeric or null."
                ) from None

            if not np.isfinite(numeric_value):
                raise PredictionError(
                    f"'{feature}' must be finite or null."
                )

            row[feature] = numeric_value

        return pd.DataFrame(
            [row],
            columns=self.feature_columns,
        )

    def predict(self, values: dict) -> dict:
        """Return individual model predictions and the ensemble result."""

        X = self._prepare_input(values)

        individual_predictions = {}

        for model_name in self.model_order:
            prediction = np.asarray(
                self.models[model_name].predict(X),
                dtype=float,
            ).reshape(-1)

            if prediction.size != 1 or not np.isfinite(prediction[0]):
                raise RuntimeError(
                    f"{model_name} returned an invalid prediction."
                )

            individual_predictions[model_name] = float(prediction[0])

        ensemble_prediction = sum(
            self.weights[name] * individual_predictions[name]
            for name in self.model_order
        )

        return {
            "prediction": float(ensemble_prediction),
            "unit": self.prediction_unit,
            "individual_predictions": individual_predictions,
            "ensemble_weights": {
                name: float(self.weights[name])
                for name in self.model_order
            },
        }


# Convenient reusable instance for the backend.
predictor = CO2AdsorptionPredictor()
