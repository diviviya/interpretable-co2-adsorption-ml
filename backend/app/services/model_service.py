"""
model_service.py: everything that touches the trained model lives here.

The routers (the URL handlers) never open joblib files themselves. They call
the functions below. That keeps the ML part in one place, so if the model
changes later, only this file changes.

Which model is loaded is decided by registry_service (the "active" entry).
load() can be called again while the server runs, to switch models.
"""
import json
import math

import joblib
import numpy as np
import pandas as pd
import shap

from ..config import ARTIFACTS, REPO_ROOT, RESULTS


class ModelService:
    def __init__(self):
        self.bundle = None
        self.version: str = ""
        self.features: list[str] = []
        self.feature_info: dict = {}
        self.active_model: dict = {}
        self.explainers: dict = {}
        self.base_value: float = 0.0
        self.importance: list[dict] = []

    # ------------------------------------------------------------------
    # Loading (at startup, and again when the admin activates a model)
    # ------------------------------------------------------------------
    def load(self, entry: dict):
        # A bundle is a dict saved with joblib (by Notebook 09, or by training_service):
        # {"models": {name: Pipeline(impute -> model)}, "weights": {name: w},
        #  "features": [...10 column names...], "target": "CO2 uptake (mmol/g)",
        #  "importance": [...]  <- only in bundles made by the admin training}
        bundle_path = REPO_ROOT / entry["path"]
        if not bundle_path.exists():
            raise RuntimeError(f"{bundle_path} not found.")
        bundle = joblib.load(bundle_path)

        # Build everything for the new model first...
        explainers, base_value = build_explainers(bundle)
        importance = self._load_importance(bundle)
        if not self.feature_info:
            # Input ranges for the form; these come from the original data.
            with open(ARTIFACTS / "feature_info.json") as f:
                self.feature_info = json.load(f)

        # ...then swap it all in at once, so a prediction arriving during the
        # switch never mixes the old model with the new explainers.
        (self.bundle, self.explainers, self.base_value, self.importance,
         self.features, self.version, self.active_model) = (
            bundle, explainers, base_value, importance,
            list(bundle["features"]), entry["version"],
            {**{k: v for k, v in entry.items() if k != "path"}, "status": "active"},
        )

    def _load_importance(self, bundle) -> list[dict]:
        # Bundles trained from the admin page carry their own importance.
        if "importance" in bundle:
            return bundle["importance"]
        # The original model: global importance saved by Notebook 08.
        imp = pd.read_csv(RESULTS / "08_shap_importance.csv", index_col=0).iloc[:, 0]
        return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in imp.items()]

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


def build_explainers(bundle) -> tuple[dict, float]:
    """Same method as Notebook 08: one TreeExplainer per model (on the model
    step, after the imputer), combined with the ensemble weights.
    Also used by training_service to compute importance for a new model."""
    models, weights = bundle["models"], bundle["weights"]
    explainers = {
        name: shap.TreeExplainer(models[name].named_steps["model"]) for name in weights
    }

    # base_value = the ensemble's average prediction before seeing any input.
    # Each explainer's expected_value is that model's average; weight them.
    def model_base(name):
        if name == "CatBoost":
            # shap reports 0 for CatBoost: it misses the starting value
            # CatBoost adds to every prediction. CatBoost stores it itself.
            return float(models[name].named_steps["model"].get_scale_and_bias()[1])
        return float(np.ravel(explainers[name].expected_value)[0])

    base_value = float(sum(w * model_base(name) for name, w in weights.items()))
    return explainers, base_value


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# One shared instance for the whole app.
model_service = ModelService()
