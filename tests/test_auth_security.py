from fastapi.testclient import TestClient
import uuid

from backend.main import app

client = TestClient(app)
TEST_USERNAME = "role_test_" + uuid.uuid4().hex[:8]
TEST_PASSWORD = "a-strong-test-password"

def test_public_registration_cannot_assign_admin_role():
    response = client.post(
        "/api/auth/register",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD, "role": "admin"},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "user"

def test_registered_user_cannot_access_admin_model_registry():
    login = client.post(
        "/api/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    response = client.get("/api/admin/models", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
