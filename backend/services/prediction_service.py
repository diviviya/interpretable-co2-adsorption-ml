"""Prediction services for single, batch and ensemble-level SHAP inference."""
from pathlib import Path
import io
import threading

import numpy as np
import pandas as pd
from ml.predictor import CO2AdsorptionPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "deployment" / "portable_ensemble"
    / "co2_adsorption_ensemble.joblib"
)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Active model bundle not found: {MODEL_PATH}")

bundle = CO2AdsorptionPredictor(MODEL_PATH).bundle
_bundle_lock = threading.RLock()
FEATURE_COLUMNS = list(bundle["feature_columns"])

FEATURE_MAP = {
    "surface_area": "Surface Area (m2/g)",
    "total_pore_volume": "Total Pore Volume(cm3/g)",
    "micropore_volume": "Micropore Volume (cm3/g)",
    "carbon": "C (%)", "hydrogen": "H (%)", "nitrogen": "N (%)",
    "oxygen": "O (%)", "sulfur": "S (%)", "temperature": "Temp (°C)",
    "pressure": "Pressure (bar)",
}


def get_bundle_snapshot() -> dict:
    """Return a shallow snapshot so a request uses one consistent active version."""
    with _bundle_lock:
        return dict(bundle)


def restore_bundle(snapshot: dict) -> None:
    """Restore a previously active bundle if registry persistence fails."""
    with _bundle_lock:
        bundle.clear()
        bundle.update(snapshot)


def activate_prediction_bundle(artifact_path: str | Path) -> dict:
    """Validate candidate artifact and smoke-test it before swapping inference."""
    from ml.predictor import CO2AdsorptionPredictor

    restored = CO2AdsorptionPredictor(Path(artifact_path)).bundle
    required = {"feature_columns", "model_order", "models", "ensemble_weights", "ensemble_strategy"}
    if not required.issubset(restored):
        raise ValueError("Model artifact has an invalid bundle structure.")
    if list(restored["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Candidate feature schema does not match the application schema.")
    if set(restored["model_order"]) != set(restored["models"]):
        raise ValueError("Candidate model order does not match its model collection.")
    if set(restored["model_order"]) != set(restored["ensemble_weights"]):
        raise ValueError("Candidate ensemble weights do not match its models.")
    weights = np.asarray([restored["ensemble_weights"][n] for n in restored["model_order"]], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Candidate ensemble weights must be finite, nonnegative and sum to one.")

    sample = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    for name in restored["model_order"]:
        pipeline = restored["models"][name]
        imputed = pipeline.named_steps["imputer"].transform(sample)
        prediction = np.asarray(pipeline.named_steps["model"].predict(imputed), dtype=float).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction).all():
            raise ValueError(f"Candidate model {name} failed its load smoke test.")

    with _bundle_lock:
        bundle.clear()
        bundle.update(restored)
    return bundle


def _predict_pipeline(pipeline, frame: pd.DataFrame) -> np.ndarray:
    imputed = pipeline.named_steps["imputer"].transform(frame)
    return np.asarray(pipeline.named_steps["model"].predict(imputed), dtype=float).reshape(-1)


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("At least one prediction row is required.")
    if frame[FEATURE_COLUMNS].isna().all(axis=1).any():
        rows = (frame.index[frame[FEATURE_COLUMNS].isna().all(axis=1)] + 1).tolist()[:5]
        raise ValueError(f"Prediction row(s) {rows} have no feature values. Provide at least one feature per row.")
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_text = frame[column].notna() & values.isna()
        if invalid_text.any():
            rows = (invalid_text[invalid_text].index + 1).tolist()[:5]
            raise ValueError(f"Column '{column}' contains non-numeric value(s) at row(s): {rows}.")
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"Column '{column}' contains infinite values.")

    constraints = {
        "Surface Area (m2/g)": (0, None),
        "Total Pore Volume(cm3/g)": (0, None),
        "Micropore Volume (cm3/g)": (0, None),
        "C (%)": (0, 100), "H (%)": (0, 100), "N (%)": (0, 100),
        "O (%)": (0, 100), "S (%)": (0, 100),
        "Temp (°C)": (-273.15, None), "Pressure (bar)": (0, None),
    }
    for column, (minimum, maximum) in constraints.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (values <= minimum if column in ("Temp (°C)", "Pressure (bar)") else values < minimum)
        if maximum is not None:
            invalid |= values.notna() & (values > maximum)
        if invalid.any():
            rows = (invalid[invalid].index + 1).tolist()[:5]
            rule = f"between {minimum} and {maximum}" if maximum is not None else f"greater than {minimum}"
            raise ValueError(f"Invalid value in '{column}' at row(s) {rows}: must be {rule}.")


def predict_single(input_dict: dict, active_bundle: dict | None = None) -> tuple[float, pd.DataFrame]:
    input_data = {FEATURE_MAP[name]: input_dict.get(name) for name in FEATURE_MAP}
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    _validate_prediction_frame(input_df)
    active = active_bundle or get_bundle_snapshot()
    predictions = [_predict_pipeline(active["models"][name], input_df) for name in active["model_order"]]
    prediction_matrix = np.column_stack(predictions)
    weights = np.array([active["ensemble_weights"][name] for name in active["model_order"]], dtype=float)
    return float((prediction_matrix @ weights)[0]), input_df


def _normalize_shap_values(values, expected_shape):
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected one regression output from TreeSHAP.")
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != expected_shape:
        raise ValueError(f"Unexpected SHAP values shape {values.shape}; expected {expected_shape}.")
    return values


def compute_shap(input_df: pd.DataFrame, active_bundle: dict | None = None):
    """Return exact weighted TreeSHAP contributions for the active ensemble.

    Each component model is explained on the imputed matrix it actually receives.
    SHAP values and expected values are combined using the active ensemble weights.
    A local additivity check prevents returning misleading explanations.
    """
    import shap

    active = active_bundle or get_bundle_snapshot()
    n_rows, n_features = input_df.shape
    combined_values = np.zeros((n_rows, n_features), dtype=float)
    combined_base = 0.0
    combined_prediction = np.zeros(n_rows, dtype=float)
    component_checks = {}

    for name in active["model_order"]:
        weight = float(active["ensemble_weights"][name])
        if abs(weight) < 1e-12:
            continue
        pipeline = active["models"][name]
        imputed = pipeline.named_steps["imputer"].transform(input_df)
        estimator = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(
            estimator,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        values = _normalize_shap_values(explainer.shap_values(imputed), (n_rows, n_features))
        expected_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if expected_values.size != 1:
            raise ValueError(f"{name} returned an unsupported SHAP expected-value shape.")
        expected = float(expected_values[0])
        component_prediction = np.asarray(estimator.predict(imputed), dtype=float).reshape(-1)
        if not np.isfinite(values).all() or not np.isfinite(component_prediction).all():
            raise ValueError(f"{name} produced non-finite SHAP values or predictions.")
        component_error = float(np.max(np.abs(expected + values.sum(axis=1) - component_prediction)))
        if component_error > 1e-3:
            raise ValueError(f"{name} SHAP additivity check failed (max error {component_error:.6g}).")
        combined_values += weight * values
        combined_base += weight * expected
        combined_prediction += weight * component_prediction
        component_checks[name] = {"weight": weight, "max_additivity_error": component_error}

    total_error = np.abs(combined_base + combined_values.sum(axis=1) - combined_prediction)
    max_error = float(total_error.max())
    if max_error > 1e-3:
        raise ValueError(f"Ensemble SHAP additivity check failed (max error {max_error:.6g}).")

    contributions = []
    row = input_df.iloc[0]
    for index, column in enumerate(FEATURE_COLUMNS):
        value = row[column]
        shap_value = float(combined_values[0, index])
        contributions.append({
            "feature": column,
            "input_value": float(value) if pd.notna(value) else None,
            "shap_value": round(shap_value, 6),
            "impact": "positive" if shap_value >= 0 else "negative",
            "formatted_shap": f"{shap_value:+.4f}",
        })
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return float(combined_base), contributions


def predict_batch_csv(file_bytes: bytes, filename: str) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file: {exc}") from exc
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    col_mapping = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        for key, value in FEATURE_MAP.items():
            if clean in (key.lower(), value.lower(), key.replace("_", " ").lower()):
                col_mapping[col] = value
                break
    renamed = df.rename(columns=col_mapping)
    missing = [column for column in FEATURE_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    features = renamed[FEATURE_COLUMNS].copy()
    _validate_prediction_frame(features)
    active = get_bundle_snapshot()
    columns = [_predict_pipeline(active["models"][name], features) for name in active["model_order"]]
    matrix = np.column_stack(columns)
    weights = np.array([active["ensemble_weights"][name] for name in active["model_order"]], dtype=float)
    predictions = (matrix @ weights).round(4)
    output = df.copy()
    output["predicted_co2_uptake_mmol_g"] = predictions
    return {
        "filename": filename,
        "rows_processed": int(len(df)),
        "mean_uptake_mmol_g": round(float(np.mean(predictions)), 4),
        "min_uptake_mmol_g": round(float(np.min(predictions)), 4),
        "max_uptake_mmol_g": round(float(np.max(predictions)), 4),
        "predictions_preview": output.head(10).to_dict(orient="records"),
        "csv_content": output.to_csv(index=False),
    }
