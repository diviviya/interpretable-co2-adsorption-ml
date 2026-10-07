from fastapi import APIRouter, Request
from app.schemas.prediction import AdsorptionFeatures, PredictionResponse

router = APIRouter()

@router.get("/health")
def health_check():
    
    return {"status": "healthy", "service": "co2-adsorption-ml"}

@router.post("/predict", response_model=PredictionResponse)
def predict_co2_uptake(payload: AdsorptionFeatures, request: Request):
    
    model_service = request.app.state.model_service
    
    
    prediction = model_service.predict(payload)
    
    return PredictionResponse(predicted_co2_uptake=prediction)