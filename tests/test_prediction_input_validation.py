"""Regression tests for physically invalid single and batch prediction inputs."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.schemas import FeatureInput
from backend.services.prediction_service import (
    FEATURE_COLUMNS,
    _validate_prediction_frame,
)


VALID_SINGLE_INPUT = {
    "surface_area": 1000,
    "total_pore_volume": 0.6,
    "micropore_volume": 0.3,
    "carbon": 75,
    "hydrogen": 2,
    "nitrogen": 1,
    "oxygen": 20,
    "sulfur": 0.2,
    "temperature": 25,
    "pressure": 1,
}


def valid_batch_frame(rows=2):
    values = {
        "Surface Area (m2/g)": 1000.0,
        "Total Pore Volume(cm3/g)": 0.6,
        "Micropore Volume (cm3/g)": 0.3,
        "C (%)": 75.0,
        "H (%)": 2.0,
        "N (%)": 1.0,
        "O (%)": 20.0,
        "S (%)": 0.2,
        "Temp (°C)": 25.0,
        "Pressure (bar)": 1.0,
    }
    return pd.DataFrame([values.copy() for _ in range(rows)], columns=FEATURE_COLUMNS)


def test_valid_single_input_is_accepted():
    parsed = FeatureInput(**VALID_SINGLE_INPUT)
    assert parsed.surface_area == 1000
    assert parsed.pressure == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("surface_area", -100, "Surface area cannot be negative"),
        ("total_pore_volume", -0.1, "Total pore volume cannot be negative"),
        ("micropore_volume", -0.1, "Micropore volume cannot be negative"),
        ("carbon", -1, "Carbon percentage must be between 0 and 100"),
        ("carbon", 101, "Carbon percentage must be between 0 and 100"),
        ("hydrogen", 101, "Hydrogen percentage must be between 0 and 100"),
        ("nitrogen", 101, "Nitrogen percentage must be between 0 and 100"),
        ("oxygen", 101, "Oxygen percentage must be between 0 and 100"),
        ("sulfur", 101, "Sulfur percentage must be between 0 and 100"),
        ("temperature", -273.15, "Temperature must be above absolute zero"),
        ("pressure", 0, "Pressure must be greater than 0 bar"),
        ("pressure", -1, "Pressure must be greater than 0 bar"),
    ],
)
def test_invalid_single_input_is_rejected(field, value, message):
    payload = {**VALID_SINGLE_INPUT, field: value}
    with pytest.raises(ValidationError) as exc_info:
        FeatureInput(**payload)
    assert message in str(exc_info.value)


def test_single_input_aliases_are_also_validated():
    payload = {key: value for key, value in VALID_SINGLE_INPUT.items() if key != "carbon"}
    payload["carbon_pct"] = 101
    with pytest.raises(ValidationError, match="Carbon percentage"):
        FeatureInput(**payload)


def test_missing_single_features_remain_allowed_for_imputation():
    payload = {**VALID_SINGLE_INPUT, "micropore_volume": None}
    parsed = FeatureInput(**payload)
    assert parsed.micropore_volume is None


def test_valid_batch_frame_is_accepted():
    _validate_prediction_frame(valid_batch_frame())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Surface Area (m2/g)", -100, "Surface Area (m2/g)"),
        ("Total Pore Volume(cm3/g)", -0.1, "Total Pore Volume(cm3/g)"),
        ("Micropore Volume (cm3/g)", -0.1, "Micropore Volume (cm3/g)"),
        ("C (%)", 101, "C (%)"),
        ("H (%)", 101, "H (%)"),
        ("N (%)", 101, "N (%)"),
        ("O (%)", 101, "O (%)"),
        ("S (%)", 101, "S (%)"),
        ("Temp (°C)", -273.15, "Temp (°C)"),
        ("Pressure (bar)", 0, "Pressure (bar)"),
    ],
)
def test_invalid_batch_values_are_rejected_with_column_and_row(column, value, message):
    frame = valid_batch_frame(rows=2)
    frame.loc[1, column] = value
    with pytest.raises(ValueError) as exc_info:
        _validate_prediction_frame(frame)
    error = str(exc_info.value)
    assert message in error
    assert "row(s) [2]" in error


def test_batch_non_numeric_and_infinite_values_are_rejected():
    non_numeric = valid_batch_frame()
    non_numeric["Surface Area (m2/g)"] = non_numeric["Surface Area (m2/g)"].astype(object)
    non_numeric.loc[0, "Surface Area (m2/g)"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        _validate_prediction_frame(non_numeric)

    infinite = valid_batch_frame()
    infinite.loc[0, "Pressure (bar)"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _validate_prediction_frame(infinite)


def test_missing_batch_features_remain_allowed_for_imputation():
    frame = valid_batch_frame()
    frame.loc[0, "Micropore Volume (cm3/g)"] = np.nan
    _validate_prediction_frame(frame)


def test_prediction_row_with_no_feature_values_is_rejected():
    frame = valid_batch_frame(rows=1)
    frame.loc[:, :] = np.nan
    with pytest.raises(ValueError, match="no feature values"):
        _validate_prediction_frame(frame)
