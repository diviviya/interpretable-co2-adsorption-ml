import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.selection_gate_service import (
    BIOMASS_PRECURSORS,
    run_selection_gate,
)

client = TestClient(app)


def test_biomass_precursors_catalog():
    assert len(BIOMASS_PRECURSORS) == 9
    ids = [p["id"] for p in BIOMASS_PRECURSORS]
    assert "coconut-shell" in ids
    assert "walnut-shell" in ids
    assert "bamboo-biochar" in ids
    assert "sugarcane-bagasse" in ids
    assert "chitosan-biochar" in ids
    assert "coffee-grounds" in ids
    assert "microalgae-biochar" in ids
    assert "rice-husk" in ids
    assert "pine-sawdust" in ids


def test_run_selection_gate_default():
    result = run_selection_gate(target_uptake=3.0, temperature=25.0, pressure=1.0)
    assert "summary" in result
    assert "candidates" in result

    summary = result["summary"]
    assert summary["total_screened"] == 9
    assert summary["approved_count"] + summary["rejected_count"] == 9
    assert summary["top_recommended"] is not None

    candidates = result["candidates"]
    assert len(candidates) == 9
    # Verify descending sort order
    uptakes = [c["predicted_uptake"] for c in candidates]
    assert uptakes == sorted(uptakes, reverse=True)


def test_run_selection_gate_filter():
    result = run_selection_gate(
        target_uptake=2.5,
        temperature=25.0,
        pressure=1.0,
        precursor_filter="bamboo-biochar",
    )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["id"] == "bamboo-biochar"


def test_run_selection_gate_validation_errors():
    with pytest.raises(ValueError):
        run_selection_gate(target_uptake=-1.0)

    with pytest.raises(ValueError):
        run_selection_gate(target_uptake=3.0, temperature=1000.0)

    with pytest.raises(ValueError):
        run_selection_gate(target_uptake=3.0, pressure=-5.0)


def test_selection_gate_api_precursors():
    response = client.get("/api/selection-gate/precursors")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 9
    assert len(data["precursors"]) == 9


def test_selection_gate_api_post():
    payload = {
        "target_uptake": 3.0,
        "temperature": 25.0,
        "pressure": 1.0,
        "precursor_filter": "all",
    }
    response = client.post("/api/selection-gate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["summary"]["total_screened"] == 9
    assert len(data["candidates"]) == 9
    for cand in data["candidates"]:
        assert "name" in cand
        assert "predicted_uptake" in cand
        assert "gate_status" in cand
        assert cand["gate_status"] in ["APPROVED", "REJECTED"]
