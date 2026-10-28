import numpy as np
import pandas as pd

from ml.ensemble_explainability import explain_ensemble
from ml.data_validation import FEATURE_COLUMNS


def test_ensemble_explanations_reconstruct_predictions():
    dataset = pd.read_csv("data/Processed_527_datapoints.csv")
    result = explain_ensemble(dataset)

    output_dir = result["output_directory"]
    predictions = pd.read_csv(
        f"{output_dir}/ensemble_shap_predictions.csv"
    )
    shap_values = pd.read_csv(
        f"{output_dir}/ensemble_shap_values.csv"
    )
    importance = pd.read_csv(
        f"{output_dir}/ensemble_shap_feature_importance.csv"
    )

    assert len(predictions) == 84
    assert list(shap_values.columns[1:]) == FEATURE_COLUMNS
    assert np.isfinite(shap_values[FEATURE_COLUMNS].to_numpy()).all()

    assert np.allclose(
        predictions["ensemble_prediction"],
        predictions["shap_reconstructed_prediction"],
        atol=1e-3,
    )

    assert (
        predictions["absolute_additivity_error"].max() < 1e-3
    )

    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert (importance["mean_absolute_shap"] >= 0).all()

    assert (
        result["metadata"]["max_absolute_ensemble_additivity_error"]
        < 1e-3
    )
