"""
predict.py: the URLs the user-side pages call.

Each function is short on purpose: check input (done by the schema),
call the model service, return JSON.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import settings
from ..schemas import PredictRequest, PredictResponse
from ..services import batch_service, history_service
from ..services.model_service import model_service

router = APIRouter(prefix="/api")


@router.get("/features")
def features():
    # min / max / median of each input in the training data (feature_info.json).
    return model_service.feature_info


@router.get("/models/active")
def active_model():
    return model_service.active_model


@router.get("/models/active/importance")
def importance():
    # Global feature importance for the About page (from Notebook 08).
    return {"model_version": settings.model_version, "importance": model_service.importance}


@router.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(req: PredictRequest):
    prediction = model_service.predict_one(req.features)
    history_service.add_prediction(req.features, round(prediction, 4), settings.model_version)
    # No range warnings here: predict.js already warns about out-of-range
    # inputs, and sending them from both sides would show each one twice.
    return PredictResponse(
        prediction=round(prediction, 4),
        model_version=settings.model_version,
        base_value=round(model_service.base_value, 4),
        shap=model_service.explain_one(req.features),
        # explanation stays empty: predict.js writes the sentence from the numbers.
    )


@router.get("/predictions/history")
def history():
    # Newest first, at most 50. Used by the History page.
    return {"items": history_service.list_recent()}


MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB, same limit as the frontend


@router.post("/predict/batch")
def predict_batch(file: UploadFile = File(...)):
    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than 5 MB.")

    # Problems with the whole file -> one clear error, nothing predicted.
    try:
        df = batch_service.read_csv(raw)
        batch_service.check_columns(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Problems with single rows -> those rows are skipped and listed.
    good, errors = batch_service.parse_rows(df)

    rows = []
    if good:
        predictions = model_service.predict_many([features for _, features in good])
        rows = [
            {"row": row_no, **features, "prediction": round(float(p), 4)}
            for (row_no, features), p in zip(good, predictions)
        ]

    return {"model_version": settings.model_version, "rows": rows, "errors": errors}