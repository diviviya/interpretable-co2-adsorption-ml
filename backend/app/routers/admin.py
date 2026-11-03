"""
admin.py: the admin-side URLs. Every route here requires the admin key.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..security import require_admin
from ..services import dataset_service

MAX_DATASET_BYTES = 10 * 1024 * 1024    # 10 MB

# dependencies=[...] on the ROUTER applies require_admin to every route
# below, including ones we add later (datasets, train, models).
router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])


@router.post("/login")
def login():
    # If we get here, require_admin already accepted the key.
    return {"ok": True}
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