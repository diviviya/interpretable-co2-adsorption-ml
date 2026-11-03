from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_models_endpoint_reports_active_registry_metadata_not_static_csv():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    active = data["active_model"]
    assert active["status"] == "active"
    assert active["version"]
    assert active["weights"]
    assert abs(sum(active["weights"].values()) - 1.0) < 1e-5
    assert active["metrics_scope"] in {"held_out_test", "validation"}
    assert "artifact_path" not in active


def test_default_registry_uses_portable_initial_bundle():
    from backend.services.storage_service import JSONStorage
    active = JSONStorage.get_active_model_version()
    assert active is not None
    if active.get("version") == "v1.0-ensemble":
        assert active["artifact_path"].replace("\\", "/").endswith(
            "artifacts/deployment/portable_ensemble/co2_adsorption_ensemble.joblib"
        )
