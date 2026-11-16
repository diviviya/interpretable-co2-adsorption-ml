"""
upload_to_hf.py: publishes the CarbonLab model to your Hugging Face account.

Uploads the trained model + its small supporting files + a model card
(README.md) to a PUBLIC model repo, so anyone (and your deployed app)
can download it.

Before running:
  1. Create a WRITE token at huggingface.co -> Settings -> Access Tokens.
  2. In the same Command Prompt, set it (it stays only in that window):
         set HF_TOKEN=hf_xxxxxxxxxxxxxxxx
  3. Run from the repo root:
         python scripts\\upload_to_hf.py
NEVER paste the token into a file, into git, or send it to anyone.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

USERNAME = "adhithyasharma"
REPO_ID = f"{USERNAME}/carbonlab-co2-ensemble"
GITHUB = "https://github.com/madhithyasharma-del/co2-adsorption-ml"

REPO = Path(__file__).resolve().parents[1]
A, R = REPO / "artifacts", REPO / "results"

FILES = [
    (A / "model_bundle.joblib", "model_bundle.joblib"),
    (A / "ensemble_weights.json", "ensemble_weights.json"),
    (A / "feature_info.json", "feature_info.json"),
    (A / "best_params.json", "best_params.json"),
    (A / "conformal.json", "conformal.json"),
    (R / "07_test_results.csv", "results/07_test_results.csv"),
    (R / "10_conformal_results.csv", "results/10_conformal_results.csv"),
    (R / "08_shap_importance.png", "results/08_shap_importance.png"),
    (R / "07_pred_vs_actual.png", "results/07_pred_vs_actual.png"),
]

MODEL_CARD = f"""---
library_name: scikit-learn
tags:
- tabular-regression
- ensemble
- shap
- conformal-prediction
- co2-adsorption
- materials-science
---

# CarbonLab: CO2 uptake of biomass-derived porous carbons

An entropy-TOPSIS weighted ensemble of six tree models (Random Forest,
Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost) that predicts
CO2 uptake (mmol/g) of biomass-derived porous carbons from their structure,
elemental composition and test conditions. It applies the ensemble-ML
approach of Fan et al. (2026) to the Yuan et al. (2021) data set
(526 unique rows after removing 1 duplicate).

Live app and code: {GITHUB}

## Inputs (10)
Surface area, total pore volume, micropore volume, C, H, N, O, S (%),
temperature (C), pressure (bar). The two pore volumes may be blank
(the training median is used).

## Test results (106 held-out rows)

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Ensemble (entropy-TOPSIS) | 0.644 | 0.477 | 0.848 |
| CatBoost alone | 0.605 | 0.446 | 0.866 |
| Ridge baseline | 0.974 | 0.745 | 0.652 |

On this test set CatBoost alone scored slightly better than the weighted
ensemble; the ensemble is kept because it follows the base paper's method.

## Enhancement over the base paper: prediction intervals
The base paper reports point predictions. We add split-conformal prediction
intervals calibrated on 5-fold out-of-fold residuals of the development
data (420 rows) and checked on the untouched test rows:

| Target level | Interval half-width (mmol/g) | Actual test coverage |
|---|---|---|
| 80% | 0.724 | 78.3% |
| 90% | 1.092 | 92.5% |
| 95% | 1.437 | 96.2% |

The half-width is the same for every input (it does not adapt to how
uncertain a particular prediction is).

## Files
- model_bundle.joblib : dict with the six fitted pipelines (median imputer + model), ensemble weights, feature names
- conformal.json : interval half-widths
- feature_info.json : min / max / median of each input in the training data
- results/ : test metrics, interval coverage, plots

## Usage
```python
import joblib
from huggingface_hub import hf_hub_download
path = hf_hub_download("{REPO_ID}", "model_bundle.joblib")
bundle = joblib.load(path)   # only load pickle files from sources you trust
```
Requires scikit-learn 1.7.2, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10,
pandas 2.3.3, numpy 2.3.5 (the versions it was trained with).

## Limitations
Trained on only 526 samples at a few temperatures (0-30 C) and pressures
(0.1-1.01 bar); predictions outside the training ranges are unreliable.
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. See the instructions at the top of this file.")
    missing = [str(p) for p, _ in FILES if not p.exists()]
    if missing:
        sys.exit("These files are missing:\n  " + "\n  ".join(missing))

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    for local, remote in FILES:
        print("uploading", remote, "...")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="model")
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
