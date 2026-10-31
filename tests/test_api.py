from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["features_count"] == 10

def test_prediction_valid():
    payload = {
        "surface_area": 1420.0,
        "total_pore_volume": 0.74,
        "micropore_volume": 0.48,
        "carbon": 78.855,
        "hydrogen": 1.5,
        "nitrogen": 2.26,
        "oxygen": 15.05,
        "sulfur": 0.0,
        "temperature": 25.0,
        "pressure": 1.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert data["unit"] == "mmol/g"
    assert len(data["individual_predictions"]) == 6
    assert len(data["ensemble_weights"]) == 6

def test_prediction_missing_values_imputed():
    payload = {
        "surface_area": 850.0,
        "total_pore_volume": 0.5,
        "micropore_volume": None,  # Should be median-imputed by pipeline
        "carbon": 75.0,
        "hydrogen": 2.0,
        "nitrogen": 1.0,
        "oxygen": 15.0,
        "sulfur": 0.5,
        "temperature": 25.0,
        "pressure": 1.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] > 0

def test_prediction_invalid_type():
    payload = {
        "surface_area": "not_a_number",
        "temperature": 25.0,
        "pressure": 1.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422

def test_models_endpoint():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "evaluation_metrics" in data
    assert len(data["evaluation_metrics"]) > 0
