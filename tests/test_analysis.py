"""Tests for calibration metrics, slice diagnostics, and visualization generation."""

import os
import json
import numpy as np
import pandas as pd
import pytest

from adlift.analysis import (
    compute_calibration_metrics,
    compute_performance_slices,
    generate_evaluation_figures,
)
from adlift.features import APPROVED_CATEGORICAL_FEATURES, APPROVED_INPUT_FEATURES
from adlift.schema import TARGET_LABEL


@pytest.fixture
def synthetic_eval_data():
    """Create a synthetic dataset for testing calibration and slicing."""
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.35, size=n)
    # Simulated well-calibrated probabilities
    y_prob = np.clip(y_true * 0.4 + np.random.uniform(0.1, 0.4, size=n), 0.05, 0.95)

    df = pd.DataFrame({
        TARGET_LABEL: y_true,
        "cpo": np.random.uniform(0.05, 0.5, size=n),
        "time_since_last_click": np.random.choice([-1, 100, 5000, 100000], size=n),
    })
    for col in APPROVED_CATEGORICAL_FEATURES:
        df[col] = np.random.randint(1, 10, size=n)

    return df, y_true, y_prob


def test_calibration_metrics(synthetic_eval_data):
    """Verify Expected Calibration Error and bin calculations."""
    _, y_true, y_prob = synthetic_eval_data
    calib = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    assert "expected_calibration_error" in calib
    assert "brier_score" in calib
    assert "log_loss" in calib
    assert 0.0 <= calib["expected_calibration_error"] <= 1.0

    total_count = sum(b["count"] for b in calib["bins"])
    assert total_count == len(y_true)


def test_performance_slices(synthetic_eval_data):
    """Verify performance slicing handles known fields without crashing."""
    df, _, y_prob = synthetic_eval_data
    slices = compute_performance_slices(df, y_prob)

    assert "user_recency_slices" in slices
    assert "bid_cpo_slices" in slices

    recency_slices = slices["user_recency_slices"]
    assert len(recency_slices) > 0
    slice_names = [s["slice_name"] for s in recency_slices]
    assert any("First-Time User" in name for name in slice_names)

    # Check total slice count matches dataset
    total_recency_samples = sum(s["sample_count"] for s in recency_slices)
    assert total_recency_samples == len(df)


def test_figure_generation(synthetic_eval_data, tmp_path):
    """Verify evaluation plots are saved with positive byte size and proper split labeling."""
    df, y_true, y_prob = synthetic_eval_data
    calib = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    output_dir = str(tmp_path / "figures")
    split_label = "Test Split Label (123 rows)"
    fig_paths = generate_evaluation_figures(
        y_true=y_true,
        y_prob=y_prob,
        calib_metrics=calib,
        output_dir=output_dir,
        split_name=split_label,
    )

    assert len(fig_paths) == 3
    for path in fig_paths:
        assert os.path.exists(path), f"Figure not found: {path}"
        assert os.path.getsize(path) > 5000, f"Figure too small / empty: {path}"


def test_calibration_report_numbers_agree():
    """Verify metrics in calibration report match model_comparison.json."""
    comp_path = os.path.join("reports", "model_comparison.json")
    calib_path = os.path.join("reports", "calibration_and_error_analysis.json")

    if os.path.exists(comp_path) and os.path.exists(calib_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        lgb_comp = comp_data["models"]["LightGBM"]
        calib_summary = calib_data["calibration_summary"]

        # Check Brier score and Log Loss agreement
        np.testing.assert_allclose(lgb_comp["brier_score"], calib_summary["brier_score"], atol=1e-4)
        np.testing.assert_allclose(lgb_comp["log_loss"], calib_summary["log_loss"], atol=1e-4)
