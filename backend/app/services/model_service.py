"""
model_service.py: everything that touches the trained model lives here.

The routers (the URL handlers) never open joblib files themselves. They call
the functions below. That keeps the ML part in one place, so if the model
changes later, only this file changes.
"""
import json
import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap

from ..config import ARTIFACTS, RESULTS, settings


class ModelService:
    def __init__(self):
        self.bundle = None
        self.features: list[str] = []
        self.feature_info: dict = {}
        self.active_model: dict = {}
        self.explainers: dict = {}
        self.base_value: float = 0.0
        self.importance: list[dict] = []

    # ------------------------------------------------------------------
    # Loading (runs once, when the server starts)
    # ------------------------------------------------------------------
    def load(self):
        # model_bundle.joblib was saved by Notebook 09:
        # {"models": {name: Pipeline(impute -> model)}, "weights": {name: w},
        #  "features": [...10 column names...], "target": "CO2 uptake (mmol/g)"}
        bundle_path = ARTIFACTS / "model_bundle.joblib"
        if not bundle_path.exists():
            raise RuntimeError(
                f"{bundle_path} not found. Run notebooks 07 and 09 first to create it."
            )
        self.bundle = joblib.load(bundle_path)
        self.features = list(self.bundle["features"])

        with open(ARTIFACTS / "feature_info.json") as f:
            self.feature_info = json.load(f)

        self.active_model = self._describe_active_model()
        self._build_explainers()
        self.importance = self._load_importance()

    def _build_explainers(self):
        # Same method as Notebook 08: one TreeExplainer per model (on the model
        # step, after the imputer), then combine them with the ensemble weights.
        models, weights = self.bundle["models"], self.bundle["weights"]
        self.explainers = {
            name: shap.TreeExplainer(models[name].named_steps["model"]) for name in weights
        }
        # base_value = the ensemble's average prediction before seeing any input.
        # Each explainer's expected_value is that model's average; weight them.
        def model_base(name):
            if name == "CatBoost":
                # shap reports 0 for CatBoost: it misses the starting value
                # CatBoost adds to every prediction. CatBoost stores it itself.
                return float(models[name].named_steps["model"].get_scale_and_bias()[1])
            return float(np.ravel(self.explainers[name].expected_value)[0])

        self.base_value = float(sum(w * model_base(name) for name, w in weights.items()))

    def _load_importance(self) -> list[dict]:
        # Global importance saved by Notebook 08 (mean |SHAP| over all rows).
        imp = pd.read_csv(RESULTS / "08_shap_importance.csv", index_col=0).iloc[:, 0]
        return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in imp.items()]

    def _describe_active_model(self) -> dict:
        # Test metrics come from Notebook 07's saved table, not typed by hand.
        test = pd.read_csv(RESULTS / "07_test_results.csv", index_col=0)
        row = test.loc["Ensemble (entropy-TOPSIS)"]

        clean = pd.read_csv(ARTIFACTS / "clean_data.csv")
        n_test = int((clean["split"] == "test").sum())

        # When the bundle file was saved = when this model version was made.
        saved = datetime.fromtimestamp((ARTIFACTS / "model_bundle.joblib").stat().st_mtime)

        return {
            "version": settings.model_version,
            "created_at": saved.astimezone().isoformat(timespec="seconds"),
            "model_type": "Entropy-TOPSIS weighted ensemble of 6 tree models",
            "status": "active",
            "dataset": f"Yuan et al. (2021) biomass carbons, {len(clean)} rows",
            "metrics": {k: round(float(row[k]), 3) for k in ("rmse", "mae", "r2")},
            "metrics_split": f"test ({n_test} rows)",
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def to_frame(self, rows: list[dict]) -> pd.DataFrame:
        """Turn input dicts into a DataFrame with the exact training column order.
        A blank (None) becomes NaN, and each model's SimpleImputer fills it with
        the training median, the same way it did during training."""
        X = pd.DataFrame(rows, columns=self.features)
        return X.astype(float)

    def predict_many(self, rows: list[dict]) -> np.ndarray:
        X = self.to_frame(rows)
        models, weights = self.bundle["models"], self.bundle["weights"]
        # Weighted sum of the 6 models' predictions (entropy-TOPSIS weights).
        return sum(w * models[name].predict(X) for name, w in weights.items())

    def predict_one(self, features: dict) -> float:
        return float(self.predict_many([features])[0])

    # ------------------------------------------------------------------
    # Explanation (SHAP) for one prediction
    # ------------------------------------------------------------------
    def explain_one(self, features: dict) -> list[dict]:
        """How much each input pushed this prediction above or below base_value.
        base_value + sum(contributions) = the prediction (SHAP values add up)."""
        models, weights = self.bundle["models"], self.bundle["weights"]
        X = self.to_frame([features])

        # The explainers work on the model step, so fill blanks first, using
        # the same median imputer as the pipelines (all six learned the same
        # medians from the same training rows; Notebook 08 does the same).
        imputer = models["CatBoost"].named_steps["impute"]
        X_imp = pd.DataFrame(imputer.transform(X), columns=self.features)

        contrib = sum(
            w * np.asarray(self.explainers[name].shap_values(X_imp))[0]
            for name, w in weights.items()
        )
        return [
            {"feature": f, "value": features.get(f), "contribution": round(float(c), 4)}
            for f, c in zip(self.features, contrib)
        ]


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# One shared instance for the whole app.
model_service = ModelService()
