"""
admin.py: the admin-side URLs. Every route here needs a logged-in account
with role "admin" (logging in itself happens in auth.py).
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..security import require_admin
from ..services import dataset_service, registry_service, training_service
from ..services.model_service import model_service

MAX_DATASET_BYTES = 10 * 1024 * 1024    # 10 MB

# dependencies=[...] on the ROUTER applies require_admin to every route below.
router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])


# ---------- Datasets ----------
@router.post("/datasets")
def upload_dataset(file: UploadFile = File(...)):
    raw = file.file.read()
    if len(raw) > MAX_DATASET_BYTES:
        raise HTTPException(413, "The file is larger than 10 MB.")
    try:
        return dataset_service.validate(raw, file.filename)
    except ValueError as e:
        # read_csv raises this for files it can't decode at all.
        raise HTTPException(400, str(e))


# ---------- Training ----------
class TrainRequest(BaseModel):
    dataset_id: str


@router.post("/train")
def start_training(req: TrainRequest):
    try:
        job_id = training_service.start_job(req.dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset not found. Upload it again.")
    except RuntimeError as e:
        # 409 Conflict: can't start now because another job is running.
        raise HTTPException(409, str(e))
    return {"job_id": job_id}


@router.get("/training/{job_id}")
def training_status(job_id: str):
    job = training_service.get_job(job_id)
    if job is None:
        # Jobs live in memory, so a server restart forgets them.
        raise HTTPException(404, "Job not found (the server may have restarted).")
    return job


# ---------- Model versions ----------
@router.get("/models")
def list_models():
    return {"models": registry_service.list_models()}


def _switch(version: str, allowed_status: str):
    try:
        models = registry_service.switch_to(version, allowed_status, model_service.load)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"models": models}


@router.post("/models/{version}/activate")
def activate(version: str):
    # Only a freshly trained candidate can be activated.
    return _switch(version, "candidate")


@router.post("/models/{version}/rollback")
def rollback(version: str):
    # Only a model that was active before can be rolled back to.
    return _switch(version, "previous")
