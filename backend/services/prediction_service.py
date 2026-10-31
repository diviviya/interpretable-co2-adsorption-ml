"""
Prediction Service for Single and Batch CSV Predictions.
Uses the official CO2AdsorptionPredictor from ml.predictor.
"""
from pathlib import Path
import io
import pandas as pd
import numpy as np

from ml.predictor import predictor, PredictionError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_COLUMNS = predictor.feature_columns

# Map Pydantic request field names (snake_case) to exact model feature columns
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

def format_sample_dict(input_dict: dict) -> dict:
    """Map flexible dictionary keys to the exact model feature names."""
    sample = {}
    for k, v in input_dict.items():
        if k in FEATURE_COLUMNS:
            sample[k] = v
        elif k in FEATURE_MAP:
            sample[FEATURE_MAP[k]] = v
    # Ensure all feature columns are present (with None if absent for median imputation)
    for col in FEATURE_COLUMNS:
        if col not in sample:
            sample[col] = None
    return sample

def predict_single(input_dict: dict) -> tuple[dict, pd.DataFrame]:
    """
    Run prediction through the official CO2AdsorptionPredictor.
    Returns:
        result: dict containing 'prediction', 'unit', 'individual_predictions', 'ensemble_weights'
        input_df: DataFrame used for SHAP or feature inspection
    """
    sample = format_sample_dict(input_dict)
    
    # Run prediction using official predictor
    result = predictor.predict(sample)
    
    # Create single-row DataFrame for SHAP explanation
    input_df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
    
    return result, input_df

def compute_shap(input_df: pd.DataFrame):
    """
    Compute local Tree-SHAP explanations using the CatBoost model step.
    Falls back gracefully if CatBoost or SHAP is unavailable.
    """
    catboost_pipeline = predictor.models.get("CatBoost")
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
                val_raw = input_df[col_name].iloc[0]
                val_float = float(val_raw) if pd.notnull(val_raw) else 0.0
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
    """
    Accept raw CSV file bytes, parse, align headers, and predict batch rows.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    # Match columns flexibly (case-insensitive and technical names)
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

    # Predict row by row using the official predictor logic
    predictions = []
    for idx, row in df_renamed[FEATURE_COLUMNS].iterrows():
        sample = {col: (None if pd.isna(row[col]) else float(row[col])) for col in FEATURE_COLUMNS}
        res = predictor.predict(sample)
        predictions.append(res["prediction"])

    final_preds = np.array(predictions).round(4)

    df_out = df.copy()
    df_out["predicted_co2_uptake_mmol_g"] = final_preds

    return {
        "filename": filename,
        "rows_processed": len(df),
        "mean_uptake_mmol_g": round(float(np.mean(final_preds)), 4),
        "min_uptake_mmol_g": round(float(np.min(final_preds)), 4),
        "max_uptake_mmol_g": round(float(np.max(final_preds)), 4),
        "predictions_preview": df_out.head(10).to_dict(orient="records"),
        "csv_content": df_out.to_csv(index=False)
    }
