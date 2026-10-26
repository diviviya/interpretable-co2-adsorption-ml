from pathlib import Path
import math
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "co2_adsorption_ensemble.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model bundle not found: {MODEL_PATH}"
    )

print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())
print("Model size:", MODEL_PATH.stat().st_size if MODEL_PATH.exists() else "Missing")

bundle = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = bundle["feature_columns"]

FEATURE_MAP = {
    "surface_area": "Surface Area (m2/g)",
    "total_pore_volume": "Total Pore Volume(cm3/g)",
    "micropore_volume": "Micropore Volume (cm3/g)",
    "carbon": "C (%)",
    "hydrogen": "H (%)",
    "nitrogen": "N (%)",
    "oxygen": "O (%)",
    "sulfur": "S (%)",
    "temperature": "Temp (°C)",
    "pressure": "Pressure (bar)",
}

assert set(FEATURE_MAP.values()) == set(FEATURE_COLUMNS)

app = FastAPI(
    title="CO2 Adsorption Prediction API",
    description="Interpretable Ensemble Learning for CO2 Adsorption on Biomass-Derived Porous Carbons",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    surface_area: float | None = None
    total_pore_volume: float | None = None
    micropore_volume: float | None = None
    carbon: float | None = None
    carbon_pct: float | None = None
    hydrogen: float | None = None
    hydrogen_pct: float | None = None
    nitrogen: float | None = None
    nitrogen_pct: float | None = None
    oxygen: float | None = None
    oxygen_pct: float | None = None
    sulfur: float | None = None
    sulfur_pct: float | None = None
    temperature: float | None = None
    temperature_c: float | None = None
    pressure: float | None = None
    pressure_bar: float | None = None

    @model_validator(mode="after")
    def normalize_fields(self):
        if self.carbon is None and self.carbon_pct is not None:
            self.carbon = self.carbon_pct
        if self.hydrogen is None and self.hydrogen_pct is not None:
            self.hydrogen = self.hydrogen_pct
        if self.nitrogen is None and self.nitrogen_pct is not None:
            self.nitrogen = self.nitrogen_pct
        if self.oxygen is None and self.oxygen_pct is not None:
            self.oxygen = self.oxygen_pct
        if self.sulfur is None and self.sulfur_pct is not None:
            self.sulfur = self.sulfur_pct
        if self.temperature is None and self.temperature_c is not None:
            self.temperature = self.temperature_c
        if self.pressure is None and self.pressure_bar is not None:
            self.pressure = self.pressure_bar

        for name, value in self:
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number or null")
        return self


class WhatIfRequest(BaseModel):
    baseline: FeatureInput
    modified: FeatureInput


def make_single_prediction(data: FeatureInput) -> tuple[float, pd.DataFrame]:
    values = {
        "surface_area": data.surface_area,
        "total_pore_volume": data.total_pore_volume,
        "micropore_volume": data.micropore_volume,
        "carbon": data.carbon,
        "hydrogen": data.hydrogen,
        "nitrogen": data.nitrogen,
        "oxygen": data.oxygen,
        "sulfur": data.sulfur,
        "temperature": data.temperature,
        "pressure": data.pressure,
    }

    input_data = {FEATURE_MAP[name]: value for name, value in values.items()}
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    predictions = []
    for name in bundle["model_order"]:
        pipeline = bundle["models"][name]
        pred = pipeline.predict(input_df)
        predictions.append(pred)

    prediction_matrix = np.column_stack(predictions)
    weights = np.array([bundle["ensemble_weights"][name] for name in bundle["model_order"]])
    res = float((prediction_matrix @ weights)[0])
    return res, input_df


def compute_shap_attribution(input_df: pd.DataFrame):
    catboost_pipeline = bundle["models"].get("CatBoost")
    shap_contributions = []
    base_value = None

    if catboost_pipeline is not None:
        try:
            from catboost import Pool
            imputer = catboost_pipeline.named_steps["imputer"]
            cb_model = catboost_pipeline.named_steps["model"]

            X_imp = imputer.transform(input_df)
            pool = Pool(X_imp, feature_names=FEATURE_COLUMNS)
            shap_raw = cb_model.get_feature_importance(pool, type="ShapValues")[0]
            base_value = round(float(shap_raw[-1]), 3)

            for col_name, sv in zip(FEATURE_COLUMNS, shap_raw[:-1]):
                val_float = float(input_df[col_name].iloc[0]) if pd.notnull(input_df[col_name].iloc[0]) else 0.0
                sv_float = float(sv)
                shap_contributions.append({
                    "feature": col_name,
                    "input_value": val_float,
                    "shap_value": round(sv_float, 4),
                    "impact": "positive" if sv_float >= 0 else "negative",
                    "formatted_shap": f"{'+' if sv_float >= 0 else ''}{sv_float:.3f}",
                })

            shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        except Exception as e:
            print("SHAP calculation exception:", e)

    return base_value, shap_contributions


@app.get("/api/health")
@app.get("/api/status")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CO2 Adsorption Prediction API",
        "model_strategy": bundle["ensemble_strategy"],
        "models_count": len(bundle["model_order"]),
        "features_count": len(FEATURE_COLUMNS),
    }


@app.post("/api/predict")
@app.post("/predict")
def predict_endpoint(request: FeatureInput):
    """Predict CO2 uptake from 10 experimental features."""
    pred_val, input_df = make_single_prediction(request)
    base_val, shap_list = compute_shap_attribution(input_df)

    return {
        "predicted_co2_uptake_mmol_g": pred_val,
        "unit": "mmol/g",
        "model": bundle["ensemble_strategy"],
        "base_value": base_val,
        "shap_contributions": shap_list,
    }


@app.post("/api/explain")
def explain_endpoint(request: FeatureInput | None = None):
    """Return local SHAP attribution and global feature importance from results."""
    # Global feature importance from results CSV
    global_importance = []
    shap_csv = RESULTS_DIR / "shap_feature_importance.csv"
    if shap_csv.exists():
        df_imp = pd.read_csv(shap_csv)
        global_importance = df_imp.to_dict(orient="records")

    local_shap = []
    base_val = None
    if request is not None and (request.surface_area is not None or request.pressure is not None):
        _, input_df = make_single_prediction(request)
        base_val, local_shap = compute_shap_attribution(input_df)

    return {
        "global_importance": global_importance,
        "base_value": base_val,
        "local_shap": local_shap,
        "note": "SHAP feature values represent cooperative game-theoretic contributions, not physical causality.",
    }


@app.post("/api/what-if")
def what_if_endpoint(request: WhatIfRequest):
    """Compare baseline vs modified predictions side-by-side."""
    base_pred, df_base = make_single_prediction(request.baseline)
    mod_pred, df_mod = make_single_prediction(request.modified)

    delta = mod_pred - base_pred
    pct_change = (delta / base_pred * 100.0) if abs(base_pred) > 1e-6 else 0.0

    # Find which features changed
    changed_features = []
    for col in FEATURE_COLUMNS:
        raw_base = df_base[col].iloc[0]
        raw_mod = df_mod[col].iloc[0]

        if pd.notnull(raw_base) and pd.notnull(raw_mod):
            v_base = float(raw_base)
            v_mod = float(raw_mod)
            if abs(v_mod - v_base) > 1e-6:
                diff = v_mod - v_base
                changed_features.append({
                    "feature": col,
                    "baseline_value": v_base,
                    "modified_value": v_mod,
                    "difference": round(diff, 4),
                    "pct_change": round((diff / v_base * 100.0) if abs(v_base) > 1e-6 else 0.0, 2),
                })
        elif raw_base != raw_mod:
            changed_features.append({
                "feature": col,
                "baseline_value": raw_base,
                "modified_value": raw_mod,
                "difference": 0.0,
                "pct_change": 0.0,
            })

    return {
        "baseline_prediction": round(base_pred, 3),
        "modified_prediction": round(mod_pred, 3),
        "delta_mmol_g": round(delta, 3),
        "percentage_change": round(pct_change, 2),
        "unit": "mmol/g",
        "changed_features": changed_features,
    }


@app.get("/api/models")
def models_endpoint():
    """Return model evaluation metrics and ensemble weights."""
    models_metrics = []
    results_csv = RESULTS_DIR / "final_test_results.csv"
    if results_csv.exists():
        df_res = pd.read_csv(results_csv)
        models_metrics = df_res.to_dict(orient="records")

    weights_list = []
    weights_csv = RESULTS_DIR / "ensemble_weights.csv"
    if weights_csv.exists():
        df_w = pd.read_csv(weights_csv)
        weights_list = df_w.to_dict(orient="records")

    return {
        "evaluation_metrics": models_metrics,
        "ensemble_weights": weights_list,
        "selected_model": bundle["ensemble_strategy"],
        "features": FEATURE_COLUMNS,
    }


# Mount frontend static files and root page
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
