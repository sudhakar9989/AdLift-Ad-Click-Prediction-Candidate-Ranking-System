"""Tests for candidate ad ranking function, tie-breaking, and validation."""

import os
import json
import numpy as np
import pytest

from adlift.features import APPROVED_INPUT_FEATURES
from adlift.ranking import OFFLINE_RANKING_DISCLAIMER, rank_candidates


@pytest.fixture
def mock_candidates():
    """Create a sample list of 4 candidate records with full approved features."""
    cands = []
    for i in range(4):
        cands.append({
            "candidate_id": f"TEST_CANDIDATE_{i+1:02d}",
            "creative_theme": f"Test Theme {i+1}",
            "is_synthetic": True,
            "cpo": 0.1 * (i + 1),
            "time_since_last_click": 500 * i,
            "campaign": 1000 + i,
            "cat1": 101,
            "cat2": 102,
            "cat3": 103,
            "cat4": 104,
            "cat5": 105,
            "cat6": 106,
            "cat7": 107,
            "cat8": 108,
            "cat9": 109,
        })
    return cands


class DummyModel:
    """Mock model returning controllable predictable probabilities."""
    def __init__(self, scores):
        self.scores = scores

    def predict_proba(self, X):
        probs = np.array(self.scores[:len(X)])
        return np.column_stack([1.0 - probs, probs])


def test_ranking_descending_order(mock_candidates):
    """Verify candidates are returned in strictly descending predicted probability order."""
    # Assign known probabilities: 0.20, 0.85, 0.45, 0.60
    mock_model = DummyModel([0.20, 0.85, 0.45, 0.60])

    result = rank_candidates(mock_candidates, model=mock_model)
    ranked = result["ranked_candidates"]

    assert len(ranked) == 4
    # Expected ranking order by probability:
    # 1. TEST_CANDIDATE_02 (0.85)
    # 2. TEST_CANDIDATE_04 (0.60)
    # 3. TEST_CANDIDATE_03 (0.45)
    # 4. TEST_CANDIDATE_01 (0.20)
    assert ranked[0]["candidate_id"] == "TEST_CANDIDATE_02"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_id"] == "TEST_CANDIDATE_04"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["candidate_id"] == "TEST_CANDIDATE_03"
    assert ranked[2]["rank"] == 3
    assert ranked[3]["candidate_id"] == "TEST_CANDIDATE_01"
    assert ranked[3]["rank"] == 4

    for i in range(len(ranked) - 1):
        assert ranked[i]["predicted_click_probability"] >= ranked[i + 1]["predicted_click_probability"]


def test_deterministic_tie_breaking(mock_candidates):
    """Verify that when probabilities are identical, candidate_id is used as a deterministic tie-breaker."""
    # Provide two candidates with identical probability (0.50)
    cands = [
        dict(mock_candidates[0], candidate_id="ZEBRA_AD"),
        dict(mock_candidates[1], candidate_id="ALPHA_AD"),
    ]
    mock_model = DummyModel([0.50, 0.50])

    result = rank_candidates(cands, model=mock_model)
    ranked = result["ranked_candidates"]

    # ALPHA_AD should precede ZEBRA_AD alphabetically
    assert ranked[0]["candidate_id"] == "ALPHA_AD"
    assert ranked[1]["candidate_id"] == "ZEBRA_AD"


def test_missing_features_validation_error(mock_candidates):
    """Verify that omitting an approved feature raises ValueError with informative message."""
    invalid_cand = mock_candidates.copy()
    # Delete 'campaign' from one candidate
    del invalid_cand[0]["campaign"]

    with pytest.raises(ValueError, match="missing required feature"):
        rank_candidates(invalid_cand, model=DummyModel([0.5]))


def test_empty_candidate_list_raises_error():
    """Verify passing an empty list raises a clear ValueError."""
    with pytest.raises(ValueError, match="Candidates list cannot be empty"):
        rank_candidates([], model=DummyModel([]))


def test_repeatable_predictions(mock_candidates):
    """Verify repeated calls return exactly identical results."""
    mock_model = DummyModel([0.15, 0.72, 0.38, 0.91])

    res1 = rank_candidates(mock_candidates, model=mock_model)
    res2 = rank_candidates(mock_candidates, model=mock_model)

    assert res1 == res2


def test_disclaimer_and_synthetic_flag_present(mock_candidates):
    """Verify the demo explicitly communicates synthetic inputs and observational limitations."""
    res = rank_candidates(mock_candidates, model=DummyModel([0.2, 0.4, 0.6, 0.8]))

    assert res["is_synthetic_demonstration"] is True
    assert "DEMO NOTICE" in res["disclaimer"]
    assert "DOES NOT prove" in res["disclaimer"]
    for c in res["ranked_candidates"]:
        assert c["is_synthetic"] is True
