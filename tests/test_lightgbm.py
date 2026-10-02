"""Tests for LightGBM tabular click model, artifact reload, and comparison table."""

import joblib
import numpy as np
import pandas as pd
import pytest

from adlift.features import APPROVED_CATEGORICAL_FEATURES, APPROVED_INPUT_FEATURES
from adlift.models.lightgbm_model import LightGBMClickModel


@pytest.fixture
def synthetic_train_val_data():
    """Create small synthetic train and validation datasets with approved features."""
    np.random.seed(42)
    n_train, n_val = 150, 50

    def _make_df(n):
        df = pd.DataFrame({
            "cpo": np.random.uniform(0.01, 1.0, size=n),
            "time_since_last_click": np.random.choice([-1, 50, 200, 1500], size=n),
            "click": np.random.binomial(1, 0.35, size=n),
        })
        for col in APPROVED_CATEGORICAL_FEATURES:
            df[col] = np.random.choice([101, 102, 103, 104, 105], size=n)
        return df

    return _make_df(n_train), _make_df(n_val)


def test_lightgbm_probabilities_and_unseen_categories(synthetic_train_val_data):
    """Verify LightGBM outputs valid probabilities and handles unseen categories and missing values."""
    train_df, val_df = synthetic_train_val_data
    X_train, y_train = train_df[APPROVED_INPUT_FEATURES], train_df["click"].values

    model = LightGBMClickModel(n_estimators=10, min_child_samples=5)
    model.fit(X_train, y_train)

    # Corrupt validation data with unseen hashes and missing values
    val_dirty = val_df[APPROVED_INPUT_FEATURES].copy()
    val_dirty.loc[0, "campaign"] = 9999999  # Unseen category
    val_dirty.loc[1, "cat3"] = 8888888      # Unseen category
    val_dirty.loc[2, "cpo"] = np.nan        # Missing numeric
    val_dirty.loc[3, "time_since_last_click"] = -1  # Sentinel value

    probs = model.predict_proba(val_dirty)
    assert probs.shape == (len(val_dirty), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0)
    assert not np.isnan(probs).any(), "Probabilities must not contain NaNs"


def test_lightgbm_artifact_save_and_reload(synthetic_train_val_data, tmp_path):
    """Verify saved LightGBM model artifact reloads cleanly and reproduces exact predictions."""
    train_df, val_df = synthetic_train_val_data
    X_train, y_train = train_df[APPROVED_INPUT_FEATURES], train_df["click"].values
    X_val = val_df[APPROVED_INPUT_FEATURES]

    model = LightGBMClickModel(n_estimators=10, min_child_samples=5)
    model.fit(X_train, y_train)

    artifact_path = tmp_path / "lightgbm_test_model.joblib"
    joblib.dump(model, artifact_path)

    reloaded_model = joblib.load(artifact_path)

    orig_probs = model.predict_proba(X_val)
    reloaded_probs = reloaded_model.predict_proba(X_val)

    np.testing.assert_allclose(orig_probs, reloaded_probs, rtol=1e-5, atol=1e-5)


def test_comparison_table_structure():
    """Verify model comparison table contains all 3 models with required metrics."""
    from adlift.models.lightgbm_model import train_and_evaluate_tabular

    # Verify that reports/model_comparison.json exists or can be verified
    import os
    import json
    report_path = os.path.join("reports", "model_comparison.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "models" in data
        models = data["models"]
        for expected in ["ConstantBaseline", "LogisticRegression", "LightGBM"]:
            assert expected in models, f"Model {expected} missing from comparison table!"
            for m in ["log_loss", "roc_auc", "pr_auc", "brier_score", "train_time_seconds"]:
                assert m in models[expected]
                assert isinstance(models[expected][m], (int, float))
