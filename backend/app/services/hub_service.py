"""
hub_service.py: fetches the trained model from Hugging Face when it is
not on this machine.

Why: the model file (model_bundle.joblib, ~49 MB) is in .gitignore, so it
is NOT in the GitHub repo. On your laptop it sits in artifacts/. On a fresh
server (Render, a teammate's PC) it doesn't exist, so at startup we
download it once from the public Hugging Face model repo named in the
HF_MODEL_REPO setting, e.g. adhithyasharma/carbonlab-co2-ensemble.

This must run BEFORE registry_service reads the model file, because the
registry describes the model using that file's details.
"""
from ..config import ARTIFACTS, settings

BUNDLE = "model_bundle.joblib"


def ensure_original_bundle() -> str:
    path = ARTIFACTS / BUNDLE
    if path.exists():
        return f"model found locally ({path.name})"
    if not settings.hf_model_repo:
        raise RuntimeError(
            f"{BUNDLE} is missing from artifacts/ and HF_MODEL_REPO is not set. "
            "Set HF_MODEL_REPO=<username>/<repo> to download it from Hugging Face."
        )
    from huggingface_hub import hf_hub_download   # imported here: only needed on a fresh server

    ARTIFACTS.mkdir(exist_ok=True)
    hf_hub_download(repo_id=settings.hf_model_repo, filename=BUNDLE, local_dir=ARTIFACTS)
    return f"model downloaded from Hugging Face ({settings.hf_model_repo})"
