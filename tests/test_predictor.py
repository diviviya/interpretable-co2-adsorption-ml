
import numpy as np
import pytest

from ml.predictor import (
    CO2AdsorptionPredictor,
    PredictionError,
    predictor,
)


SAMPLE = {
    "Surface Area (m2/g)": 1000,
    "Total Pore Volume(cm3/g)": 0.5,
    "Micropore Volume (cm3/g)": 0.3,
    "C (%)": 80,
    "H (%)": 2,
    "N (%)": 1,
    "O (%)": 15,
    "S (%)": 0.5,
    "Temp (°C)": 25,
    "Pressure (bar)": 1,
}


def test_prediction_contains_all_models():
    result = predictor.predict(SAMPLE)

    assert set(result["individual_predictions"]) == set(
        predictor.model_order
    )
    assert np.isfinite(result["prediction"])
    assert result["unit"] == "mmol/g"


def test_ensemble_matches_weighted_model_predictions():
    result = predictor.predict(SAMPLE)

    expected = sum(
        predictor.weights[name]
        * result["individual_predictions"][name]
        for name in predictor.model_order
    )

    assert result["prediction"] == pytest.approx(
        expected, abs=1e-10
    )


def test_predictions_match_saved_bundle_directly():
    X = predictor._prepare_input(SAMPLE)

    expected = sum(
        predictor.weights[name]
        * float(predictor.models[name].predict(X)[0])
        for name in predictor.model_order
    )

    actual = predictor.predict(SAMPLE)["prediction"]

    assert actual == pytest.approx(expected, abs=1e-10)


def test_missing_value_uses_pipeline_imputation():
    sample = SAMPLE.copy()
    sample["Surface Area (m2/g)"] = None

    result = predictor.predict(sample)

    assert np.isfinite(result["prediction"])


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -float("inf"), True, "invalid"],
)
def test_invalid_values_are_rejected(bad_value):
    sample = SAMPLE.copy()
    sample["Pressure (bar)"] = bad_value

    with pytest.raises(PredictionError):
        predictor.predict(sample)


def test_unexpected_feature_is_rejected():
    sample = SAMPLE.copy()
    sample["unexpected_feature"] = 10

    with pytest.raises(PredictionError):
        predictor.predict(sample)


def test_custom_model_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        CO2AdsorptionPredictor(
            tmp_path / "missing_model.joblib"
        )
