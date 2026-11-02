"""
Prediction Service for Single and Batch CSV Predictions.
"""
from pathlib import Path
import io
import pandas as pd
import numpy as np
import joblib
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "deployment" / "co2_adsorption_ensemble.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Active model bundle not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
_bundle_lock = threading.RLock()
FEATURE_COLUMNS = bundle["feature_columns"]

def activate_prediction_bundle(artifact_path: str | Path) -> dict:
    """Load and validate a candidate before atomically swapping inference bundle."""
    candidate = joblib.load(Path(artifact_path))
    required = {"feature_columns", "model_order", "models", "ensemble_weights", "ensemble_strategy"}
    if not isinstance(candidate, dict) or not required.issubset(candidate):
        raise ValueError("Model artifact has an invalid bundle structure.")
    if list(candidate["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("Candidate feature schema does not match the application schema.")
    if set(candidate["model_order"]) != set(candidate["models"]):
        raise ValueError("Candidate model order does not match its model collection.")
    if set(candidate["model_order"]) != set(candidate["ensemble_weights"]):
        raise ValueError("Candidate ensemble weights do not match its models.")
    weights = np.asarray([candidate["ensemble_weights"][name] for name in candidate["model_order"]], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Candidate ensemble weights must be finite, nonnegative and sum to one.")
    # Smoke-test model output before changing the live bundle.
    sample = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}])
    for name in candidate["model_order"]:
        model_pipeline = candidate["models"][name]
        imputed = model_pipeline.named_steps["imputer"].transform(sample)
        prediction = np.asarray(model_pipeline.named_steps["model"].predict(imputed), dtype=float)
        if prediction.size != 1 or not np.isfinite(prediction).all():
            raise ValueError(f"Candidate model {name} failed its load smoke test.")
    with _bundle_lock:
        bundle.clear()
        bundle.update(candidate)
    return bundle

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

def _predict_pipeline(pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Predict through fitted pipeline steps without sklearn tag introspection."""
    imputed = pipeline.named_steps["imputer"].transform(frame)
    return np.asarray(pipeline.named_steps["model"].predict(imputed), dtype=float)


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Reject physically invalid or non-numeric prediction inputs.

    Missing feature values remain allowed because the fitted pipelines use
    median imputation. This function is used for both single and batch inputs.
    """
    if frame.empty:
        raise ValueError("At least one prediction row is required.")

    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_text = frame[column].notna() & values.isna()
        if invalid_text.any():
            rows = (invalid_text[invalid_text].index + 1).tolist()[:5]
            raise ValueError(
                f"Column '{column}' contains non-numeric value(s) at row(s): {rows}."
            )
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"Column '{column}' contains infinite values.")

    constraints = {
        "Surface Area (m2/g)": (0, None, "must be zero or greater"),
        "Total Pore Volume(cm3/g)": (0, None, "must be zero or greater"),
        "Micropore Volume (cm3/g)": (0, None, "must be zero or greater"),
        "C (%)": (0, 100, "must be between 0 and 100"),
        "H (%)": (0, 100, "must be between 0 and 100"),
        "N (%)": (0, 100, "must be between 0 and 100"),
        "O (%)": (0, 100, "must be between 0 and 100"),
        "S (%)": (0, 100, "must be between 0 and 100"),
        "Temp (°C)": (-273.15, None, "must be above -273.15 °C"),
        "Pressure (bar)": (0, None, "must be greater than 0"),
    }
    for column, (minimum, maximum, message) in constraints.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (values < minimum)
        if column in ("Temp (°C)", "Pressure (bar)"):
            invalid = values.notna() & (values <= minimum)
        if maximum is not None:
            invalid = invalid | (values.notna() & (values > maximum))
        if invalid.any():
            rows = (invalid[invalid].index + 1).tolist()[:5]
            raise ValueError(
                f"Invalid value in '{column}' at row(s) {rows}: {message}."
            )


def predict_single(input_dict: dict) -> tuple[float, pd.DataFrame]:
    input_data = {FEATURE_MAP[name]: input_dict.get(name) for name in FEATURE_MAP}
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    _validate_prediction_frame(input_df)

    predictions = []
    for name in bundle["model_order"]:
        pipeline = bundle["models"][name]
        pred = _predict_pipeline(pipeline, input_df)
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
    _validate_prediction_frame(X_features)

    # Predict row by row or in batch using model pipelines
    predictions = []
    for name in bundle["model_order"]:
        pipeline = bundle["models"][name]
        pred = _predict_pipeline(pipeline, X_features)
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
