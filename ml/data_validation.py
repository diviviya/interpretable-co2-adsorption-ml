
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.predictor import predictor


FEATURE_COLUMNS = predictor.feature_columns
TARGET_COLUMN = predictor.bundle["target_column"]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


@dataclass
class ValidationReport:
    valid: bool
    rows_received: int
    rows_after_duplicate_removal: int
    missing_values: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "rows_received": self.rows_received,
            "rows_after_duplicate_removal": (
                self.rows_after_duplicate_removal
            ),
            "missing_values": self.missing_values,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_training_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate a labelled dataset for model retraining.

    The returned DataFrame is a copy. The input DataFrame is
    never modified. Missing feature values are permitted because
    the fitted model pipelines use median imputation, but target
    values must be present.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    rows_received = len(df)
    errors = []
    warnings = []

    if df.empty:
        report = ValidationReport(
            valid=False,
            rows_received=0,
            rows_after_duplicate_removal=0,
            errors=["Dataset is empty."],
        )
        return df.copy(), report

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[
            df.columns.duplicated()
        ].tolist()

        report = ValidationReport(
            valid=False,
            rows_received=rows_received,
            rows_after_duplicate_removal=rows_received,
            errors=[
                f"Duplicate column names: {duplicate_columns}"
            ],
        )
        return df.copy(), report

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        report = ValidationReport(
            valid=False,
            rows_received=rows_received,
            rows_after_duplicate_removal=rows_received,
            errors=[
                f"Missing required columns: {missing_columns}"
            ],
        )
        return df.copy(), report

    result = df[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        converted = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        invalid_text = result[column].notna() & converted.isna()

        if invalid_text.any():
            errors.append(
                f"Column '{column}' contains "
                f"{int(invalid_text.sum())} non-numeric value(s)."
            )

        result[column] = converted

        values = result[column].dropna().to_numpy(dtype=float)

        if not np.isfinite(values).all():
            errors.append(
                f"Column '{column}' contains infinite values."
            )

    # Features may be missing because the training pipeline imputes them.
    feature_missing = result[FEATURE_COLUMNS].isna().sum()
    missing_values = {
        column: int(count)
        for column, count in feature_missing.items()
        if count > 0
    }

    if missing_values:
        warnings.append(
            "Missing feature values will require imputation "
            "during model training."
        )

    if result[TARGET_COLUMN].isna().any():
        errors.append(
            f"Target column '{TARGET_COLUMN}' contains missing "
            "or invalid values."
        )

    duplicate_count = int(result.duplicated().sum())

    if duplicate_count:
        if remove_duplicates:
            result = result.drop_duplicates().reset_index(drop=True)
            warnings.append(
                f"Removed {duplicate_count} duplicate row(s)."
            )
        else:
            warnings.append(
                f"Found {duplicate_count} duplicate row(s). "
                "Review them before training."
            )

    if len(result) < 10:
        warnings.append(
            "Very few rows are available. Model evaluation "
            "may be unreliable."
        )

    report = ValidationReport(
        valid=len(errors) == 0,
        rows_received=rows_received,
        rows_after_duplicate_removal=len(result),
        missing_values=missing_values,
        errors=errors,
        warnings=warnings,
    )

    return result, report
