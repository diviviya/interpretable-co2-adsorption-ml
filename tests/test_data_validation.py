
import pandas as pd

from ml.data_validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_dataset,
)


def make_valid_dataset(rows=20):
    data = {
        column: [1.0] * rows
        for column in FEATURE_COLUMNS
    }
    data[TARGET_COLUMN] = [2.0] * rows
    return pd.DataFrame(data)


def test_valid_dataset_passes():
    df = make_valid_dataset()

    cleaned, report = validate_training_dataset(df)

    assert report.valid
    assert len(cleaned) == len(df)
    assert not report.errors


def test_missing_required_column_fails():
    df = make_valid_dataset().drop(
        columns=[FEATURE_COLUMNS[0]]
    )

    _, report = validate_training_dataset(df)

    assert not report.valid
    assert any(
        "Missing required columns" in error
        for error in report.errors
    )


def test_missing_features_are_allowed():
    df = make_valid_dataset()
    df.loc[0, FEATURE_COLUMNS[0]] = None

    _, report = validate_training_dataset(df)

    assert report.valid
    assert FEATURE_COLUMNS[0] in report.missing_values


def test_missing_target_fails():
    df = make_valid_dataset()
    df.loc[0, TARGET_COLUMN] = None

    _, report = validate_training_dataset(df)

    assert not report.valid



def test_duplicate_removal_is_optional():
    df = make_valid_dataset()

    # Make each original row unique.
    df[FEATURE_COLUMNS[0]] = range(1, len(df) + 1)

    # Append one exact duplicate.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    unchanged, report = validate_training_dataset(df)
    cleaned, cleaned_report = validate_training_dataset(
        df,
        remove_duplicates=True,
    )

    assert len(unchanged) == 21
    assert len(cleaned) == 20
    assert report.valid
    assert cleaned_report.valid
    assert any("duplicate" in warning.lower() for warning in report.warnings)



def test_non_numeric_feature_fails():
    df = make_valid_dataset()

    column = FEATURE_COLUMNS[0]
    df[column] = df[column].astype(object)
    df.loc[0, column] = "not-a-number"

    _, report = validate_training_dataset(df)

    assert not report.valid
    assert any(
        "non-numeric" in error
        for error in report.errors
    )
