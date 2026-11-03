"""
predict.py: the URLs the user-side pages call.

Each function is short on purpose: check input (done by the schema),
call the model service, return JSON.
"""
from fastapi import APIRouter

from ..config import settings
from ..schemas import PredictRequest, PredictResponse
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
    # No range warnings here: predict.js already warns about out-of-range
    # inputs, and sending them from both sides would show each one twice.
    return PredictResponse(
        prediction=round(prediction, 4),
        model_version=settings.model_version,
        base_value=round(model_service.base_value, 4),
        shap=model_service.explain_one(req.features),
        # explanation stays empty: predict.js writes the sentence from the numbers.
    )
