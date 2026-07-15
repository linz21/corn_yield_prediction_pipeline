"""
Tests for the FastAPI prediction API.
Run:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app, YieldRequest

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "ok"


class TestPredictInputValidation:
    def test_valid_state_title_case(self):
        req = YieldRequest(year=2024, state="illinois")
        assert req.state == "Illinois"

    def test_commodity_upper_case(self):
        req = YieldRequest(year=2024, state="Iowa", commodity="corn")
        assert req.commodity == "CORN"

    def test_year_lower_bound_rejected(self):
        with pytest.raises(Exception):
            YieldRequest(year=1800, state="Illinois")

    def test_year_upper_bound_rejected(self):
        with pytest.raises(Exception):
            YieldRequest(year=2100, state="Illinois")

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(Exception):
            YieldRequest(year=2024, state="Illinois", confidence=1.5)

    def test_negative_planted_acres_rejected(self):
        with pytest.raises(Exception):
            YieldRequest(year=2024, state="Illinois", planted_acres=-100)

    def test_default_values(self):
        req = YieldRequest(year=2024, state="Iowa")
        assert req.commodity == "CORN"
        assert req.n_bootstrap == 200
        assert req.confidence == 0.95


class TestPredictEndpoint:
    """
    These tests require a trained model in mlruns/ (run train.py first).
    Marked to skip gracefully if no model is available, so CI doesn't
    fail on a fresh checkout without training data.
    """

    def test_predict_returns_valid_response_or_503(self):
        payload = {
            "year": 2024,
            "state": "Illinois",
            "planted_acres": 61300,
        }
        response = client.post("/predict", json=payload)
        # 503 is acceptable if no model has been trained yet (e.g. fresh CI run)
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "predicted_yield_bu_per_acre" in data
            assert "ci_lower" in data
            assert "ci_upper" in data
            assert data["ci_upper"] >= data["ci_lower"]
            assert data["confidence_level"] == 0.95

    def test_predict_ci_bounds_are_ordered(self):
        payload = {"year": 2023, "state": "Iowa"}
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["ci_lower"] <= data["predicted_yield_bu_per_acre"] <= data["ci_upper"]

    def test_predict_invalid_year_returns_422(self):
        payload = {"year": 1800, "state": "Illinois"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_state_returns_422(self):
        payload = {"year": 2024}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
