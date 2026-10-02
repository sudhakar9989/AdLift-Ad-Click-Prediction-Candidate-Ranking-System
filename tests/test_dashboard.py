"""Unit and integration tests for the AdLift interactive dashboard."""

import os
from typing import Any, Dict

import pandas as pd
import pytest

from adlift.dashboard import (
    format_candidate_for_scoring,
    load_calibration_data,
    load_candidates,
    load_comparison_data,
    load_model,
)
from adlift.features import APPROVED_INPUT_FEATURES, PROHIBITED_FEATURES
from adlift.ranking import OFFLINE_RANKING_DISCLAIMER, rank_candidates


def test_load_candidates() -> None:
    """Verify synthetic candidates demo data loads correctly."""
    candidates = load_candidates()
    assert isinstance(candidates, list)
    assert len(candidates) >= 5

    for c in candidates:
        assert "candidate_id" in c
        assert "cpo" in c
        assert "time_since_last_click" in c
        assert "campaign" in c
        assert c.get("is_synthetic") is True

        # Check that no prohibited target leakage columns are in candidates
        for leak_col in PROHIBITED_FEATURES:
            assert leak_col not in c, f"Target leakage feature '{leak_col}' found in candidate!"


def test_load_comparison_data() -> None:
    """Verify model comparison report loads and contains all evaluated models."""
    data = load_comparison_data()
    assert data is not None
    assert "models" in data

    expected_models = ["ConstantBaseline", "LogisticRegression", "LightGBM", "PyTorch_EmbeddingMLP"]
    for model_name in expected_models:
        assert model_name in data["models"]
        metrics = data["models"][model_name]
        assert "log_loss" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "brier_score" in metrics


def test_load_calibration_data() -> None:
    """Verify calibration analysis report loads and contains summary and slices."""
    cal_data = load_calibration_data()
    assert cal_data is not None
    assert "calibration_summary" in cal_data
    summary = cal_data["calibration_summary"]
    assert "expected_calibration_error" in summary
    assert "brier_score" in summary
    assert "log_loss" in summary
    assert "score_buckets" in cal_data
    assert "performance_slices" in cal_data


@pytest.mark.parametrize("model_name", [
    "LightGBM",
    "Logistic Regression",
    "Constant Baseline",
    "PyTorch Embedding-MLP",
])
def test_load_model_artifacts(model_name: str) -> None:
    """Verify all four model artifacts can be loaded and have predict_proba."""
    model = load_model(model_name)
    assert model is not None
    assert hasattr(model, "predict_proba")

    # Predict on a dummy single-row DataFrame
    dummy_row = {
        "cpo": 0.25,
        "time_since_last_click": 3600,
        "campaign": 22589171,
        "cat1": 5824233,
        "cat2": 9312274,
        "cat3": 3490278,
        "cat4": 29196072,
        "cat5": 11409686,
        "cat6": 1973606,
        "cat7": 25162884,
        "cat8": 29196072,
        "cat9": 29196072,
    }
    df = pd.DataFrame([dummy_row])
    probs = model.predict_proba(df)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_format_candidate_for_scoring() -> None:
    """Verify helper formats a single impression with all approved features."""
    cat_vals = {f"cat{i}": 1000 + i for i in range(1, 10)}
    rec = format_candidate_for_scoring(
        cpo=0.50,
        time_since_last_click=1200,
        campaign=12345,
        cat_values=cat_vals,
    )
    assert isinstance(rec, dict)
    assert rec["cpo"] == 0.50
    assert rec["time_since_last_click"] == 1200
    assert rec["campaign"] == 12345

    for f in APPROVED_INPUT_FEATURES:
        assert f in rec

    for leak in PROHIBITED_FEATURES:
        assert leak not in rec


def test_dashboard_candidate_ranking() -> None:
    """Verify ranking candidates through the loaded LightGBM model works deterministically."""
    model = load_model("LightGBM")
    candidates = load_candidates()
    result = rank_candidates(candidates, model=model)

    assert result["disclaimer"] == OFFLINE_RANKING_DISCLAIMER
    assert result["is_synthetic_demonstration"] is True
    assert result["candidates_count"] == len(candidates)

    ranked = result["ranked_candidates"]
    assert len(ranked) == len(candidates)

    # Verify descending probability order
    for i in range(len(ranked) - 1):
        assert ranked[i]["predicted_click_probability"] >= ranked[i + 1]["predicted_click_probability"]


def test_dashboard_headless_execution() -> None:
    """Verify full Streamlit dashboard executes headlessly without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest

    dashboard_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "adlift", "dashboard.py"))
    at = AppTest.from_file(dashboard_file)
    at.run(timeout=30)
    assert len(at.exception) == 0, f"AppTest raised unexpected exception(s): {at.exception}"
