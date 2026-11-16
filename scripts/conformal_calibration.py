"""
conformal_calibration.py: ENHANCEMENT over the base paper (Fan et al.).

The paper's ensemble gives one number per prediction. This script adds a
90% PREDICTION INTERVAL around that number using split-conformal
prediction, and checks honestly that the interval really covers ~90% of
unseen test samples.

How it works (plain words):
  1. Take the development data (train + validation = 420 rows).
  2. 5-fold cross-validation: for each fold, fit the 6 models on the other
     4 folds and predict the held-out fold. Every row now has a prediction
     made by models that never saw that row (an "out-of-fold" prediction).
  3. residual = |actual - ensemble prediction| for each row.
  4. The 90% interval half-width q is the (finite-sample corrected) 90th
     percentile of those residuals: prediction +/- q.
  5. Check on the 106 TEST rows, which were never used here, how often
     the true value really falls inside prediction +/- q.

Run from the repo root (the folder that contains backend/ and artifacts/):
    python scripts/conformal_calibration.py
Outputs:
    artifacts/conformal.json              (read by the backend)
    results/10_conformal_results.csv      (numbers for the report)
"""
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reuse the exact model definitions the app's training uses (same classes,
# same fixed settings, same tuned parameters as Notebooks 05 and 07).
from backend.app.services.training_service import _make_model, MODEL_CLASSES  # noqa: E402

ARTIFACTS, RESULTS = REPO / "artifacts", REPO / "results"
TARGET = "CO2 uptake (mmol/g)"
LEVELS = [0.80, 0.90, 0.95]          # coverage levels we store
HEADLINE = 0.90                       # the one the app shows

df = pd.read_csv(ARTIFACTS / "clean_data.csv")
FEATURES = [c for c in df.columns if c not in ("row_id", "split", TARGET)]
weights = json.loads((ARTIFACTS / "ensemble_weights.json").read_text())
best_params = json.loads((ARTIFACTS / "best_params.json").read_text())

dev = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
X_dev, y_dev = dev[FEATURES], dev[TARGET].to_numpy()
X_test, y_test = test[FEATURES], test[TARGET].to_numpy()

# ---- Step 2: out-of-fold ensemble predictions on the development data ----
oof = np.zeros(len(dev))
for k, (tr, ho) in enumerate(KFold(5, shuffle=True, random_state=42).split(X_dev), 1):
    fold_pred = 0.0
    for name, w in weights.items():
        m = _make_model(name, best_params).fit(X_dev.iloc[tr], y_dev[tr])
        fold_pred = fold_pred + w * m.predict(X_dev.iloc[ho])
    oof[ho] = fold_pred
    print(f"fold {k}/5 done")

# ---- Steps 3-4: residuals and conformal quantiles ----
res = np.abs(y_dev - oof)
n = len(res)


def conformal_q(level: float) -> float:
    # Finite-sample correction: use the ceil((n+1)*level)/n quantile.
    k = min(math.ceil((n + 1) * level), n)
    return float(np.sort(res)[k - 1])


q = {f"{lv:.2f}": round(conformal_q(lv), 4) for lv in LEVELS}

# ---- Step 5: honest check on the untouched test rows ----
bundle = joblib.load(ARTIFACTS / "model_bundle.joblib")
test_pred = sum(w * bundle["models"][nm].predict(X_test) for nm, w in bundle["weights"].items())
test_res = np.abs(y_test - test_pred)
rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))

rows = []
for lv in LEVELS:
    half = q[f"{lv:.2f}"]
    lo = np.maximum(test_pred - half, 0.0)          # uptake can't be negative
    hi = test_pred + half
    covered = float(np.mean((y_test >= lo) & (y_test <= hi)))
    # Baseline for comparison: the usual "+/- z * RMSE" (assumes normal errors).
    z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}[lv]
    g_cov = float(np.mean(test_res <= z * rmse))
    rows.append({"level": lv, "conformal_half_width": half,
                 "conformal_test_coverage": round(covered, 3),
                 "gaussian_half_width": round(z * rmse, 4),
                 "gaussian_test_coverage": round(g_cov, 3)})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
RESULTS.mkdir(exist_ok=True)
table.to_csv(RESULTS / "10_conformal_results.csv", index=False)

out = {
    "method": "split-conformal prediction on 5-fold out-of-fold ensemble residuals",
    "n_calibration": int(n),
    "headline_level": f"{HEADLINE:.2f}",
    "half_width": q,
    "test_coverage": {f"{r['level']:.2f}": r["conformal_test_coverage"] for r in rows},
}
(ARTIFACTS / "conformal.json").write_text(json.dumps(out, indent=2))
print("saved artifacts/conformal.json and results/10_conformal_results.csv")
