
import pytest

from ml.evaluation import evaluate_regression


def test_regression_metrics():
    result = evaluate_regression(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    )

    assert result["MAE"] == pytest.approx(1 / 3)
    assert result["RMSE"] == pytest.approx((1 / 3) ** 0.5)
    assert result["R2"] == pytest.approx(0.5)


def test_empty_inputs_fail():
    with pytest.raises(ValueError):
        evaluate_regression([], [])


def test_mismatched_shapes_fail():
    with pytest.raises(ValueError):
        evaluate_regression([1.0, 2.0], [1.0])


def test_non_finite_predictions_fail():
    with pytest.raises(ValueError):
        evaluate_regression([1.0, 2.0], [1.0, float("nan")])
