"""Security regression tests for authenticated prediction and user history."""
import uuid

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.storage_service import JSONStorage

client = TestClient(app)


def create_account(role="user"):
    username = f"auth_scope_{uuid.uuid4().hex[:10]}"
    password = "a-strong-test-password"
    if role == "admin":
        JSONStorage.create_user(username, password, "admin")
    else:
        JSONStorage.create_user(username, password, "user")
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return username, response.json()["access_token"]


VALID_FEATURES = {
    "surface_area": 1000, "total_pore_volume": 0.6, "micropore_volume": 0.3,
    "carbon": 75, "hydrogen": 2, "nitrogen": 1, "oxygen": 20,
    "sulfur": 0.2, "temperature": 25, "pressure": 1,
}

VALID_WHAT_IF = {"baseline": VALID_FEATURES, "modified": {**VALID_FEATURES, "pressure": 2}}


def test_prediction_routes_require_authentication():
    single = client.post("/api/predict", json=VALID_FEATURES)
    batch = client.post("/api/predict/batch", files={"file": ("input.csv", b"a,b\n1,2\n", "text/csv")})
    batch_download = client.post("/api/predict/batch/download", files={"file": ("input.csv", b"a,b\n1,2\n", "text/csv")})
    what_if = client.post("/api/what-if", json=VALID_WHAT_IF)
    history = client.get("/api/history")
    assert [single.status_code, batch.status_code, batch_download.status_code, what_if.status_code, history.status_code] == [401] * 5


def test_authenticated_user_can_access_prediction_workflows(monkeypatch):
    _, token = create_account()
    headers = {"Authorization": f"Bearer {token}"}
    def fake_predict_single(values):
        import pandas as pd
        from backend.main import FEATURE_COLUMNS
        from backend.services.prediction_service import FEATURE_MAP
        row = {FEATURE_MAP[key]: value for key, value in values.items()}
        return 3.25, pd.DataFrame([row], columns=FEATURE_COLUMNS)

    monkeypatch.setattr("backend.main.predict_single", fake_predict_single)
    monkeypatch.setattr("backend.main.compute_shap", lambda frame: (3.0, []))
    monkeypatch.setattr("backend.main.predict_batch_csv", lambda content, filename: {
        "filename": filename, "rows_processed": 1, "mean_uptake_mmol_g": 3.2,
        "min_uptake_mmol_g": 3.2, "max_uptake_mmol_g": 3.2,
        "predictions_preview": [], "csv_content": "predicted\n3.2\n"
    })
    single = client.post("/api/predict", json=VALID_FEATURES, headers=headers)
    batch = client.post("/api/predict/batch", files={"file": ("input.csv", b"header\nvalue\n", "text/csv")}, headers=headers)
    batch_download = client.post("/api/predict/batch/download", files={"file": ("input.csv", b"header\nvalue\n", "text/csv")}, headers=headers)
    what_if = client.post("/api/what-if", json=VALID_WHAT_IF, headers=headers)
    assert single.status_code == 200
    assert batch.status_code == 200
    assert batch_download.status_code == 200
    assert what_if.status_code == 200


def test_user_history_is_scoped_and_admin_can_view_all(monkeypatch):
    user_a, token_a = create_account("user")
    _, token_b = create_account("user")
    _, admin_token = create_account("admin")
    calls = []

    def fake_get_predictions(username, limit=50):
        calls.append(username)
        if username is None:
            return [{"username": user_a}, {"username": "another-user"}]
        return [{"username": username}]

    monkeypatch.setattr(JSONStorage, "get_predictions", fake_get_predictions)
    response_a = client.get("/api/history", headers={"Authorization": f"Bearer {token_a}"})
    response_b = client.get("/api/history", headers={"Authorization": f"Bearer {token_b}"})
    response_admin = client.get("/api/history", headers={"Authorization": f"Bearer {admin_token}"})

    assert response_a.status_code == response_b.status_code == response_admin.status_code == 200
    assert calls == [user_a, response_b.json()["history"][0]["username"], None]
    assert len(response_admin.json()["history"]) == 2


def test_local_explanation_requires_authentication_but_global_explanation_remains_public():
    global_response = client.post("/api/explain", json={})
    local_response = client.post("/api/explain", json=VALID_FEATURES)
    assert global_response.status_code == 200
    assert local_response.status_code == 401
