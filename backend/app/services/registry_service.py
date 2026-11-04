"""
registry_service.py: the list of model versions and which one is active.

The list lives in one small JSON file, backend/data/models/registry.json.
Each entry describes one saved model bundle:
  {version, created_at, model_type, status, dataset, metrics, metrics_split, path}
status is one of:
  "active"    - the model every prediction uses right now (exactly one)
  "candidate" - newly trained, not used yet; the admin can activate it
  "previous"  - was active before; the admin can roll back to it

The first time the server starts, the file doesn't exist yet, so we create
it with one entry: the original model from the notebooks (v1).
"""
import json
import threading
from datetime import datetime

import pandas as pd

from ..config import ARTIFACTS, REPO_ROOT, RESULTS, settings

# backend/data/ is in .gitignore, so trained models never reach GitHub.
MODELS_DIR = REPO_ROOT / "backend" / "data" / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Training runs in a background thread, and requests run in other threads.
# The lock stops two of them editing registry.json at the same moment.
_lock = threading.Lock()


def _seed_entry() -> dict:
    """Describe the original notebook model (v1). Numbers come from the
    notebooks' saved files, not typed by hand."""
    test = pd.read_csv(RESULTS / "07_test_results.csv", index_col=0)
    row = test.loc["Ensemble (entropy-TOPSIS)"]
    clean = pd.read_csv(ARTIFACTS / "clean_data.csv")
    n_test = int((clean["split"] == "test").sum())
    bundle_path = ARTIFACTS / "model_bundle.joblib"
    saved = datetime.fromtimestamp(bundle_path.stat().st_mtime)
    return {
        "version": settings.model_version,
        "created_at": saved.astimezone().isoformat(timespec="seconds"),
        "model_type": "Entropy-TOPSIS weighted ensemble of 6 tree models",
        "status": "active",
        "dataset": f"Yuan et al. (2021) biomass carbons, {len(clean)} rows",
        "metrics": {k: round(float(row[k]), 3) for k in ("rmse", "mae", "r2")},
        "metrics_split": f"test ({n_test} rows)",
        # Paths are stored relative to the repo root, with "/" so they work
        # on Windows and Linux alike.
        "path": bundle_path.relative_to(REPO_ROOT).as_posix(),
    }


def _read() -> list[dict]:
    if not REGISTRY_PATH.exists():
        entries = [_seed_entry()]
        _write(entries)
        return entries
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write(entries: list[dict]) -> None:
    # Write to a temporary file first, then swap it in. If the server stops
    # halfway through writing, the old registry.json is still intact.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def public(entry: dict) -> dict:
    # The file path is for the server only; the browser doesn't need it.
    return {k: v for k, v in entry.items() if k != "path"}


def list_models() -> list[dict]:
    with _lock:
        return [public(e) for e in reversed(_read())]   # newest first


def active_entry() -> dict:
    with _lock:
        return next(e for e in _read() if e["status"] == "active")


def next_version() -> str:
    # v1-ensemble, v2-ensemble, ... (only one training job runs at a time,
    # so two jobs can't pick the same number)
    with _lock:
        return f"v{len(_read()) + 1}-ensemble"


def add_candidate(entry: dict) -> None:
    with _lock:
        entries = _read()
        entries.append({**entry, "status": "candidate"})
        _write(entries)


def switch_to(version: str, allowed_status: str, load_model) -> list[dict]:
    """Make `version` the active model.
    allowed_status: "candidate" for Activate, "previous" for Roll back.
    load_model: a function that loads the entry's bundle into the app.

    The model is loaded FIRST. If loading fails, the registry is not
    changed, so the old model keeps serving and the file stays truthful."""
    with _lock:
        entries = _read()
        target = next((e for e in entries if e["version"] == version), None)
        if target is None:
            raise LookupError(f"No model version called {version}.")
        if target["status"] != allowed_status:
            raise ValueError(f"{version} is '{target['status']}', not '{allowed_status}'.")

        load_model(target)

        for e in entries:
            if e["status"] == "active":
                e["status"] = "previous"
        target["status"] = "active"
        _write(entries)
        return [public(e) for e in reversed(entries)]
