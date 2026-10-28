"""
Prediction Service for Single and Batch CSV Predictions.
"""
from pathlib import Path
import io
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "deployment" / "co2_adsorption_ensemble.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Active model bundle not found: {MODEL_PATH}")

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

def predict_single(input_dict: dict) -> tuple[float, pd.DataFrame]:
    input_data = {FEATURE_MAP[name]: input_dict.get(name) for name in FEATURE_MAP}
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

def compute_shap(input_df: pd.DataFrame):
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

def predict_batch_csv(file_bytes: bytes, filename: str) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    # Match columns flexibly (either technical names or exact dataset names)
    col_mapping = {}
    for col in df.columns:
        col_clean = col.strip().lower()
        for k, v in FEATURE_MAP.items():
            if col_clean in (k.lower(), v.lower(), k.replace("_", " ").lower()):
                col_mapping[col] = v
                break

    df_renamed = df.rename(columns=col_mapping)
    missing = [c for c in FEATURE_COLUMNS if c not in df_renamed.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_features = df_renamed[FEATURE_COLUMNS].copy()

    # Predict row by row or in batch using model pipelines
    predictions = []
    for name in bundle["model_order"]:
        pipeline = bundle["models"][name]
        pred = pipeline.predict(X_features)
        predictions.append(pred)

    pred_matrix = np.column_stack(predictions)
    weights = np.array([bundle["ensemble_weights"][name] for name in bundle["model_order"]])
    final_preds = (pred_matrix @ weights).round(4)

    df_out = df.copy()
    df_out["predicted_co2_uptake_mmol_g"] = final_preds

    # Summary metrics
    return {
        "filename": filename,
        "rows_processed": len(df),
        "mean_uptake_mmol_g": round(float(np.mean(final_preds)), 4),
        "min_uptake_mmol_g": round(float(np.min(final_preds)), 4),
        "max_uptake_mmol_g": round(float(np.max(final_preds)), 4),
        "predictions_preview": df_out.head(10).to_dict(orient="records"),
        "csv_content": df_out.to_csv(index=False)
    }
