"""
dataset_service.py: checks an uploaded training CSV and saves it if usable.

Training needs more than prediction does: the target column must be there
and filled in, and there must be enough rows to split into train/val/test.
"""
import secrets
from datetime import datetime

import pandas as pd

from ..config import REPO_ROOT
from ..schemas import ALL_FEATURES
from .batch_service import read_csv

TARGET = "CO2 uptake (mmol/g)"
REQUIRED_COLUMNS = sorted(ALL_FEATURES) + [TARGET]
MIN_ROWS = 100          # fewer than this and the 3-way split is too small to trust

# backend/data/ is already in .gitignore, so uploads never reach GitHub.
DATASETS_DIR = REPO_ROOT / "backend" / "data" / "datasets"


def validate(raw: bytes, filename: str) -> dict:
    errors = []
    df = read_csv(raw)     # same utf-8 / cp1252 handling as batch prediction

    # 1. Whole-file problems: if columns are missing, nothing else is checkable.
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append("Missing required column(s): " + ", ".join(missing_cols))
        return _report(None, filename, df, {}, 0, errors)

    df = df[REQUIRED_COLUMNS]          # ignore any extra columns
    if len(df) < MIN_ROWS:
        errors.append(f"Only {len(df)} rows. Training needs at least {MIN_ROWS}.")

    # 2. Turn text into numbers. errors="coerce" makes anything that isn't
    #    a number become NaN, so we can count blanks and bad cells separately.
    blank = df.apply(lambda col: col.str.strip() == "")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    not_a_number = numeric.isna() & ~blank

    for col in REQUIRED_COLUMNS:
        bad = int(not_a_number[col].sum())
        if bad:
            first_line = int(not_a_number[col].idxmax()) + 2   # +2: header is line 1
            errors.append(f'"{col}": {bad} cell(s) are not numbers (first at line {first_line}).')

    # 3. The target can't be imputed: a row without an answer can't teach anything.
    target_blanks = int(blank[TARGET].sum())
    if target_blanks:
        errors.append(f'"{TARGET}" is empty in {target_blanks} row(s). Every row needs a value.')

    # Feature blanks are fine: the pipeline fills them with the training median.
    missing_values = {c: int(blank[c].sum()) for c in REQUIRED_COLUMNS}
    duplicate_rows = int(numeric.duplicated().sum())

    # 4. Save only if usable, under a fresh ID the training step will use.
    dataset_id = None
    if not errors:
        dataset_id = datetime.now().strftime("ds-%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        numeric.to_csv(DATASETS_DIR / f"{dataset_id}.csv", index=False)

    return _report(dataset_id, filename, df, missing_values, duplicate_rows, errors)


def _report(dataset_id, filename, df, missing_values, duplicate_rows, errors) -> dict:
    # The exact shape admin.js reads.
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "errors": errors,
    }


def path_for(dataset_id: str):
    # Used by the training step. Only accept IDs that look like ours, so a
    # request can't point at "../../something" on disk.
    if not dataset_id.startswith("ds-") or not dataset_id.replace("-", "").isalnum():
        return None
    path = DATASETS_DIR / f"{dataset_id}.csv"
    return path if path.exists() else None