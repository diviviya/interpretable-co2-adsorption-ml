"""
batch_service.py: turns an uploaded CSV into rows the model can predict,
and collects a clear message for every row that can't be used.
"""
import io

import pandas as pd
from pydantic import ValidationError

from ..schemas import ALL_FEATURES, PredictRequest

MAX_ROWS = 5000


def read_csv(raw: bytes) -> pd.DataFrame:
    # Excel on Windows often saves CSVs as cp1252 instead of UTF-8, and the
    # "°" in "Temp (°C)" is stored differently in each. Try both.
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Couldn't read the file. Save it as 'CSV UTF-8' and try again.")

    # dtype=str + keep_default_na=False: read every cell as plain text, so we
    # can give our own clear message for blanks and non-numbers.
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def check_columns(df: pd.DataFrame):
    # Problems with the whole file: stop before looking at any row.
    missing = sorted(ALL_FEATURES - set(df.columns))
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    if df.empty:
        raise ValueError("The file has a header but no data rows.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Too many rows ({len(df)}). The limit is {MAX_ROWS}.")


def parse_rows(df: pd.DataFrame):
    """Returns (good, errors): good = [(row_number, features)], errors = [{row, message}]."""
    good, errors = [], []
    for i, record in enumerate(df.to_dict("records")):
        row_no = i + 2   # line number in the CSV file: line 1 is the header
        try:
            features = {}
            for col in ALL_FEATURES:
                text = record[col].strip()
                try:
                    features[col] = None if text == "" else float(text)
                except ValueError:
                    raise ValueError(f'"{col}" is not a number ("{text}")')
            # Same rules as single predictions: reuse the schema.
            features = PredictRequest(features=features).features
        except ValidationError as e:
            errors.append({"row": row_no, "message": e.errors()[0]["msg"].removeprefix("Value error, ")})
            continue
        except ValueError as e:
            errors.append({"row": row_no, "message": str(e)})
            continue
        good.append((row_no, features))
    return good, errors