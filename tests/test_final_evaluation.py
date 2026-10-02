"""Unit and schema tests for the final quarantined test set evaluation."""

import json
import os
import pytest

from adlift.features import APPROVED_INPUT_FEATURES, PROHIBITED_FEATURES
from adlift.schema import TARGET_LABEL

REPORT_JSON_PATH = os.path.join("reports", "test_set_evaluation.json")
REPORT_MD_PATH = os.path.join("reports", "test_set_evaluation.md")
TEST_CSV_PATH = os.path.join("data", "processed", "test.csv")
VAL_CSV_PATH = os.path.join("data", "processed", "validation.csv")


def test_test_set_file_exists_and_uncontaminated() -> None:
    """Verify test split file exists and has correct length and columns."""
    assert os.path.exists(TEST_CSV_PATH), f"Quarantined test file missing at {TEST_CSV_PATH}"

    import pandas as pd
    df = pd.read_csv(TEST_CSV_PATH)
    assert len(df) == 15000, f"Expected 15,000 test rows, got {len(df)}"
    assert TARGET_LABEL in df.columns

    for f in APPROVED_INPUT_FEATURES:
        assert f in df.columns, f"Approved feature {f} missing from test.csv"

    # Confirm prohibited post-impression leakage features are absent
    for leak in ["conversion", "conversion_timestamp", "conversion_id", "attribution", "click_pos", "click_nb"]:
        assert leak not in df.columns, f"Leakage feature {leak} found in test.csv!"


def test_chronological_test_boundary_invariant() -> None:
    """Verify that test split timestamps strictly follow validation split timestamps."""
    import pandas as pd
    val_df = pd.read_csv(VAL_CSV_PATH, usecols=["timestamp"])
    test_df = pd.read_csv(TEST_CSV_PATH, usecols=["timestamp"])

    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    assert min_test_ts >= max_val_ts, (
        f"Chronological split violation: min test timestamp ({min_test_ts}) "
        f"is earlier than max validation timestamp ({max_val_ts})"
    )


def test_test_evaluation_reports_exist() -> None:
    """Verify test evaluation JSON and Markdown reports exist and are non-empty."""
    assert os.path.exists(REPORT_JSON_PATH), f"Missing {REPORT_JSON_PATH}"
    assert os.path.exists(REPORT_MD_PATH), f"Missing {REPORT_MD_PATH}"

    assert os.path.getsize(REPORT_JSON_PATH) > 500
    assert os.path.getsize(REPORT_MD_PATH) > 500


def test_test_evaluation_json_schema() -> None:
    """Verify schema and metric sanity in test_set_evaluation.json."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_sample_count"] == 15000
    assert data["test_clicks_count"] == 5002
    assert 0.30 <= data["empirical_test_ctr"] <= 0.40
    assert "models" in data

    models = data["models"]
    for expected_model in ["ConstantBaseline", "LogisticRegression", "LightGBM", "PyTorch_EmbeddingMLP"]:
        assert expected_model in models, f"Model {expected_model} missing from test evaluation report"
        m = models[expected_model]["metrics"]
        assert "log_loss" in m
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert "brier_score" in m
        assert "expected_calibration_error" in m
        assert "inference_latency_ms_per_item" in m

    # LightGBM should beat the ConstantBaseline and LogisticRegression on test ROC AUC
    lgb_auc = models["LightGBM"]["metrics"]["roc_auc"]
    base_auc = models["ConstantBaseline"]["metrics"]["roc_auc"]
    lr_auc = models["LogisticRegression"]["metrics"]["roc_auc"]

    assert lgb_auc > base_auc
    assert lgb_auc >= lr_auc
    assert lgb_auc > 0.665

    # Verification of zero severe temporal degradation
    for name, res in models.items():
        comp = res.get("validation_comparison", {})
        if comp:
            delta_auc = abs(comp.get("delta_roc_auc", 0.0))
            delta_ll = abs(comp.get("delta_log_loss", 0.0))
            assert delta_auc < 0.02, f"Severe temporal ROC AUC drift ({delta_auc}) in {name}"
            assert delta_ll < 0.03, f"Severe temporal Log Loss drift ({delta_ll}) in {name}"
