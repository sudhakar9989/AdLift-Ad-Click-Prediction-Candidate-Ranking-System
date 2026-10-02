"""Tests for FastAPI endpoints: /health, /predict, and /rank."""

import json
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient

from adlift.api.app import create_app
from adlift.ranking import rank_candidates


@pytest.fixture
def client():
    """Create FastAPI test client with loaded model."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_candidates():
    """Load sample synthetic candidate records."""
    candidates_path = os.path.join("data", "demo", "synthetic_candidates.json")
    with open(candidates_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_health_endpoint(client):
    """Verify /health returns 200 with model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "LightGBMClickModel"
    assert "version" in data


def test_predict_endpoint_valid_request(client, sample_candidates):
    """Verify /predict returns valid probabilities in [0, 1]."""
    impressions = [
        {
            "impression_id": c["candidate_id"],
            "cpo": c["cpo"],
            "time_since_last_click": c["time_since_last_click"],
            "campaign": c["campaign"],
            "cat1": c["cat1"],
            "cat2": c["cat2"],
            "cat3": c["cat3"],
            "cat4": c["cat4"],
            "cat5": c["cat5"],
            "cat6": c["cat6"],
            "cat7": c["cat7"],
            "cat8": c["cat8"],
            "cat9": c["cat9"],
        }
        for c in sample_candidates[:3]
    ]

    response = client.post("/predict", json={"impressions": impressions})
    assert response.status_code == 200
    data = response.json()

    assert data["predictions_count"] == 3
    for p in data["predictions"]:
        prob = p["predicted_click_probability"]
        assert 0.0 <= prob <= 1.0
        assert p["predicted_click_binary"] in (0, 1)


def test_predict_invalid_request_returns_422(client):
    """Verify that malformed requests return 422 with validation errors."""
    # 1. Missing required field 'cpo'
    bad_request = {
        "impressions": [
            {
                "time_since_last_click": 100,
                "campaign": 12345,
                # 'cpo' missing
                "cat1": 1, "cat2": 2, "cat3": 3, "cat4": 4,
                "cat5": 5, "cat6": 6, "cat7": 7, "cat8": 8, "cat9": 9,
            }
        ]
    }
    resp = client.post("/predict", json=bad_request)
    assert resp.status_code == 422
    assert "Validation Error" in resp.json()["error"]

    # 2. Empty impressions list
    resp_empty = client.post("/predict", json={"impressions": []})
    assert resp_empty.status_code == 422


def test_rank_endpoint_matches_offline_scorer(client, sample_candidates):
    """Verify that POST /rank matches the offline rank_candidates scorer exactly."""
    # Offline ranking result
    offline_result = rank_candidates(sample_candidates)

    # API ranking result
    api_response = client.post("/rank", json={"candidates": sample_candidates})
    assert api_response.status_code == 200
    api_result = api_response.json()

    assert api_result["candidates_count"] == len(sample_candidates)
    assert api_result["is_synthetic_demonstration"] is True
    assert "DEMO NOTICE" in api_result["disclaimer"]

    # Check that candidate ranks and IDs match bitwise
    offline_ranked = offline_result["ranked_candidates"]
    api_ranked = api_result["ranked_candidates"]

    assert len(offline_ranked) == len(api_ranked)
    for off, api in zip(offline_ranked, api_ranked):
        assert off["rank"] == api["rank"]
        assert off["candidate_id"] == api["candidate_id"]
        np.testing.assert_allclose(
            off["predicted_click_probability"],
            api["predicted_click_probability"],
            atol=1e-5,
        )


def test_rank_invalid_request_returns_422(client):
    """Verify that invalid candidate data in /rank returns 422."""
    bad_rank_request = {
        "candidates": [
            {
                "candidate_id": "TEST_CAND_01",
                # missing all features
            }
        ]
    }
    resp = client.post("/rank", json=bad_rank_request)
    assert resp.status_code == 422
