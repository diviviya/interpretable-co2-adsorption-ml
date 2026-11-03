import numpy as np
import pandas as pd

from ml.predictor import predictor
from backend.services.prediction_service import compute_shap


def test_runtime_shap_explains_weighted_active_ensemble():
    sample = {
        "Surface Area (m2/g)": 1000,
        "Total Pore Volume(cm3/g)": 0.6,
        "Micropore Volume (cm3/g)": 0.3,
        "C (%)": 75,
        "H (%)": 2,
        "N (%)": 1,
        "O (%)": 20,
        "S (%)": 0.2,
        "Temp (°C)": 25,
        "Pressure (bar)": 1,
    }
    frame = pd.DataFrame([sample], columns=predictor.feature_columns)
    prediction = predictor.predict(sample)["prediction"]
    base_value, contributions = compute_shap(frame)

    assert len(contributions) == len(predictor.feature_columns)
    assert all(np.isfinite(item["shap_value"]) for item in contributions)
    assert np.isclose(
        base_value + sum(item["shap_value"] for item in contributions),
        prediction,
        atol=1e-3,
    )
