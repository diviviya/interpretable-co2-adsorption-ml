from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "portable_ensemble"
    / "co2_adsorption_ensemble.joblib"
)


class PredictionError(ValueError):
    """Raised when prediction inputs are invalid."""


class _PortableXGBoostPipeline:
    """Run the saved imputer followed by a JSON-loaded XGBoost model."""

    def __init__(self, imputer, model):
        self.imputer = imputer
        self.model = model

    @property
    def named_steps(self):
        return {"imputer": self.imputer, "model": self.model}

    def predict(self, X):
        return self.model.predict(self.imputer.transform(X))


class CO2AdsorptionPredictor:
    """Load and run the saved CO2 adsorption ensemble."""

    def __init__(self, model_path: Path = MODEL_PATH):
        model_path = Path(model_path)

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

        if self.bundle.get("portable_format") == 1:
            required_files = {
                "xgboost_model_file",
                "xgboost_imputer_file",
            }
            missing_files = required_files - self.bundle.keys()
            if missing_files:
                raise ValueError(
                    "Portable bundle is missing file references: "
                    f"{sorted(missing_files)}"
                )

            bundle_dir = model_path.parent
            xgb_path = bundle_dir / self.bundle["xgboost_model_file"]
            imputer_path = bundle_dir / self.bundle["xgboost_imputer_file"]

            if not xgb_path.is_file():
                raise FileNotFoundError(
                    f"XGBoost JSON model not found: {xgb_path}"
                )
            if not imputer_path.is_file():
                raise FileNotFoundError(
                    f"XGBoost imputer not found: {imputer_path}"
                )

            imputer = joblib.load(imputer_path)
            xgb_model = XGBRegressor()
            xgb_model.load_model(str(xgb_path))

            xgb_name = "XGBoost"
            if xgb_name not in self.models:
                raise ValueError(
                    "Portable bundle has no XGBoost model entry."
                )

            self.models[xgb_name] = _PortableXGBoostPipeline(
                imputer, xgb_model
            )

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
            feature for feature in values
            if feature not in self.feature_columns
        ]
        if extra:
            raise PredictionError(f"Unexpected features: {extra}")

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

        return pd.DataFrame([row], columns=self.feature_columns)

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


predictor = CO2AdsorptionPredictor()
