"""
training_service.py: trains a new candidate model in a background thread.

The steps are the same as the notebooks, so a model trained here on the
original data comes out like the original model:
  1. drop duplicate rows                          (Notebook 02)
  2. split 64% train / 16% validation / 20% test, seed 42  (Notebook 02)
  3. fit the 6 models on train with the saved best params, score them on
     validation, and turn those scores into entropy-TOPSIS weights
  4. refit the 6 models on train + validation, test the ensemble  (Notebook 07)
  5. save the bundle + add it to the registry as a "candidate"

No Optuna here: tuning takes far too long for a web request, so we reuse
artifacts/best_params.json from the notebooks.

The candidate is NOT used for predictions until the admin activates it.
"""
import json
import threading
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ..config import ARTIFACTS, REPO_ROOT
from . import dataset_service, registry_service
from .model_service import build_explainers, model_service

SEED = 42
TARGET = dataset_service.TARGET

# Same classes and fixed settings as Notebook 07.
MODEL_CLASSES = {
    "Random Forest":     (RandomForestRegressor,     {"random_state": SEED, "n_jobs": -1}),
    "Extra Trees":       (ExtraTreesRegressor,       {"random_state": SEED, "n_jobs": -1}),
    "Gradient Boosting": (GradientBoostingRegressor, {"random_state": SEED}),
    "XGBoost":           (XGBRegressor,              {"random_state": SEED, "n_jobs": -1}),
    "LightGBM":          (LGBMRegressor,             {"random_state": SEED, "n_jobs": -1, "verbose": -1}),
    "CatBoost":          (CatBoostRegressor,         {"random_state": SEED, "verbose": 0}),
}


# ======================================================================
# Job bookkeeping (who is running, progress, logs)
# ======================================================================
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _new_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"job-{stamp}-{uuid.uuid4().hex[:4]}"


def _update(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)


def _log(job_id: str, message: str, progress: float | None = None) -> None:
    with _lock:
        job = _jobs[job_id]
        job["logs"].append(message)
        if progress is not None:
            job["progress"] = round(progress, 3)


def get_job(job_id: str) -> dict | None:
    # Return a copy: the training thread may append to logs while
    # FastAPI is turning this dict into JSON.
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {**job, "logs": list(job["logs"])}


def start_job(dataset_id: str) -> str:
    # path_for returns None for an unknown or badly formed ID.
    csv_path = dataset_service.path_for(dataset_id)
    if csv_path is None:
        raise FileNotFoundError(dataset_id)

    with _lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise RuntimeError("A training job is already running. Wait for it to finish.")
        job_id = _new_job_id()
        _jobs[job_id] = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.0,
            "logs": [],
            "candidate": None,
            "error": None,
        }

    # daemon=True: this thread won't keep the server alive on shutdown.
    threading.Thread(target=_run, args=(job_id, csv_path, dataset_id), daemon=True).start()
    return job_id


# ======================================================================
# The ML part
# ======================================================================
def _make_model(name: str, best_params: dict) -> Pipeline:
    cls, fixed = MODEL_CLASSES[name]
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", cls(**best_params[name], **fixed)),
    ])


def _metrics(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def entropy_topsis_weights(scores: dict[str, dict]) -> tuple[dict, dict]:
    """scores: {model name: {"rmse", "mae", "r2"}} on the validation set.
    Returns (model weights that sum to 1, criterion weights).

    Entropy step: decides how much each METRIC counts. A metric where all
    models score almost the same can't tell them apart, so it gets little say.
    TOPSIS step: ranks each MODEL by how close it is to an imaginary perfect
    model (best on every metric) and how far from a worst one. That
    closeness, scaled to sum to 1, becomes the model's weight."""
    names = list(scores)
    crit = ["rmse", "mae", "r2"]
    benefit = np.array([False, False, True])        # lower RMSE/MAE is better, higher R² is better
    M = np.array([[scores[n][c] for c in crit] for n in names], dtype=float)

    # --- Entropy weights for the 3 metrics ---
    lo, hi = M.min(axis=0), M.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    R = np.where(benefit, (M - lo) / span, (hi - M) / span) + 1e-12   # 0..1, 1 = best; tiny value avoids log(0)
    P = R / R.sum(axis=0)
    E = -(P * np.log(P)).sum(axis=0) / np.log(len(names))
    D = 1 - E
    crit_w = D / D.sum() if D.sum() > 0 else np.full(len(crit), 1 / len(crit))

    # --- TOPSIS closeness for each model ---
    V = M / np.sqrt((M ** 2).sum(axis=0)) * crit_w
    best = np.where(benefit, V.max(axis=0), V.min(axis=0))
    worst = np.where(benefit, V.min(axis=0), V.max(axis=0))
    d_best = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(axis=1))
    total = d_best + d_worst
    C = np.where(total > 0, d_worst / np.where(total > 0, total, 1), 0.5)

    w = C / C.sum()
    return dict(zip(names, (float(x) for x in w))), dict(zip(crit, (float(x) for x in crit_w)))


def _importance(models: dict, weights: dict, X: pd.DataFrame, features: list[str]) -> list[dict]:
    """Global importance for the About page: mean |SHAP| over (up to) 100
    rows, combined with the ensemble weights. Same idea as Notebook 08.
    100 rows is enough for a stable ranking and keeps this step short
    (SHAP keeps the server busy while it runs)."""
    sample = X.sample(min(100, len(X)), random_state=SEED)
    imputer = models["CatBoost"].named_steps["impute"]
    X_imp = pd.DataFrame(imputer.transform(sample), columns=features)
    explainers, _ = build_explainers({"models": models, "weights": weights})
    contrib = sum(w * np.asarray(explainers[n].shap_values(X_imp)) for n, w in weights.items())
    mean_abs = np.abs(contrib).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": features[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order]


def _run(job_id: str, csv_path, dataset_id: str) -> None:
    # Everything is inside try/except: if anything breaks, the job is marked
    # "failed" with the reason, instead of the thread dying silently and the
    # page showing "Training…" forever.
    try:
        # Same column order as the active model, so a candidate bundle is
        # interchangeable with it.
        features = list(model_service.features)
        best_params = json.loads((ARTIFACTS / "best_params.json").read_text())

        # 1. Load + drop duplicates
        _log(job_id, "Loading dataset", 0.02)
        df = pd.read_csv(csv_path)
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        _log(job_id, f"{before} rows, {before - len(df)} duplicate(s) removed -> {len(df)} rows")

        # 2. Split exactly like Notebook 02: 20% test, then 20% of the rest as validation
        ids = np.arange(len(df))
        dev_ids, test_ids = train_test_split(ids, test_size=0.20, random_state=SEED)
        train_ids, val_ids = train_test_split(dev_ids, test_size=0.20, random_state=SEED)
        X, y = df[features], df[TARGET]
        X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
        X_val, y_val = X.iloc[val_ids], y.iloc[val_ids]
        X_dev, y_dev = X.iloc[dev_ids], y.iloc[dev_ids]
        X_test, y_test = X.iloc[test_ids], y.iloc[test_ids]
        _log(job_id, f"Split: {len(train_ids)} train / {len(val_ids)} validation / {len(test_ids)} test", 0.05)

        # 3. Fit on train, score on validation -> weights
        _log(job_id, "Refitting models with saved best params (validation round)")
        val_scores = {}
        for i, name in enumerate(MODEL_CLASSES):
            model = _make_model(name, best_params).fit(X_train, y_train)
            val_scores[name] = _metrics(y_val, model.predict(X_val))
            s = val_scores[name]
            _log(job_id, f"  {name}: val RMSE {s['rmse']:.3f}, R² {s['r2']:.3f}", 0.05 + 0.35 * (i + 1) / 6)

        weights, crit_w = entropy_topsis_weights(val_scores)
        _log(job_id, "Metric weights (entropy): " + ", ".join(f"{k} {v:.2f}" for k, v in crit_w.items()))
        _log(job_id, "Model weights (TOPSIS): " + ", ".join(f"{k} {v:.3f}" for k, v in weights.items()), 0.42)

        # 4. Refit on train + validation, test once
        _log(job_id, "Refitting models on train + validation (final round)")
        final_models, test_preds = {}, {}
        for i, name in enumerate(MODEL_CLASSES):
            final_models[name] = _make_model(name, best_params).fit(X_dev, y_dev)
            test_preds[name] = final_models[name].predict(X_test)
            _log(job_id, f"  {name}: done", 0.42 + 0.38 * (i + 1) / 6)

        ensemble = sum(weights[n] * test_preds[n] for n in weights)
        test_metrics = _metrics(y_test, ensemble)
        cat = _metrics(y_test, test_preds["CatBoost"])
        _log(job_id, f"Test: ensemble RMSE {test_metrics['rmse']:.3f}, MAE {test_metrics['mae']:.3f}, "
                     f"R² {test_metrics['r2']:.3f}  (CatBoost alone R² {cat['r2']:.3f})", 0.82)

        # 5. Importance, save, register
        _log(job_id, "Computing SHAP importance")
        importance = _importance(final_models, weights, X_dev, features)

        _log(job_id, "Saving candidate model", 0.95)
        version = registry_service.next_version()
        registry_service.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = registry_service.MODELS_DIR / f"{version}.joblib"
        joblib.dump({"models": final_models, "weights": weights, "features": features,
                     "target": TARGET, "importance": importance}, path)

        entry = {
            "version": version,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_type": "Entropy-TOPSIS weighted ensemble of 6 tree models",
            "dataset": f"{dataset_id}, {len(df)} rows",
            "metrics": {k: round(v, 3) for k, v in test_metrics.items()},
            "metrics_split": f"test ({len(test_ids)} rows)",
            "path": path.relative_to(REPO_ROOT).as_posix(),
        }
        registry_service.add_candidate(entry)

        _log(job_id, f"Done: {version} saved as a candidate")
        _update(job_id, status="succeeded", progress=1.0,
                candidate={**registry_service.public(entry), "status": "candidate"})
    except Exception as e:
        _log(job_id, f"Failed: {type(e).__name__}: {e}")
        _update(job_id, status="failed", error=f"Training failed: {e}")
