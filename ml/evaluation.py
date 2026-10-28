
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict:
    """Calculate standard regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("Actual and predicted values must have matching shapes.")

    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual and predicted values must be finite.")

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
