"""Tests for click-prediction baselines, evaluation metrics, and artifact reload."""

import joblib
import numpy as np
import pandas as pd
import pytest

from adlift.features import APPROVED_CATEGORICAL_FEATURES, APPROVED_INPUT_FEATURES
from adlift.models.baselines import (
    ConstantBaseline,
    build_logistic_regression_pipeline,
    evaluate_predictions,
)


@pytest.fixture
def synthetic_train_val_data():
    """Create small synthetic train and validation datasets with approved features."""
    np.random.seed(42)
    n_train, n_val = 120, 40

    def _make_df(n):
        df = pd.DataFrame({
            "cpo": np.random.uniform(0.01, 1.0, size=n),
            "time_since_last_click": np.random.choice([-1, 100, 500, 2000], size=n),
            "click": np.random.binomial(1, 0.35, size=n),
        })
        for col in APPROVED_CATEGORICAL_FEATURES:
            df[col] = np.random.choice([101, 102, 103, 104, 105], size=n)
        return df

    return _make_df(n_train), _make_df(n_val)


def test_constant_baseline_predictions(synthetic_train_val_data):
    """Verify ConstantBaseline produces valid probabilities equal to training prior."""
    train_df, val_df = synthetic_train_val_data
    X_train, y_train = train_df[APPROVED_INPUT_FEATURES], train_df["click"].values
    X_val = val_df[APPROVED_INPUT_FEATURES]

    model = ConstantBaseline()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)
    assert probs.shape == (len(val_df), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0)

    # Click probability should match train click rate
    expected_prior = np.mean(y_train)
    np.testing.assert_allclose(probs[:, 1], expected_prior)


def test_logistic_regression_predictions(synthetic_train_val_data):
    """Verify LogisticRegression produces valid probabilities and handles unseen categories."""
    train_df, val_df = synthetic_train_val_data
    X_train, y_train = train_df[APPROVED_INPUT_FEATURES], train_df["click"].values

    pipeline = build_logistic_regression_pipeline(max_iter=50)
    pipeline.fit(X_train, y_train)

    # Introduce unseen category hashes and nulls in validation data
    val_dirty = val_df[APPROVED_INPUT_FEATURES].copy()
    val_dirty.loc[0, "campaign"] = 99999999  # Completely new unseen category
    val_dirty.loc[1, "cat1"] = 88888888      # Completely new unseen category
    val_dirty.loc[2, "cpo"] = np.nan         # Missing numeric value
    val_dirty.loc[3, "time_since_last_click"] = -1  # Sentinel value

    probs = pipeline.predict_proba(val_dirty)
    assert probs.shape == (len(val_dirty), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0)
    assert not np.isnan(probs).any(), "Probabilities must not contain NaNs!"


def test_model_artifact_save_and_reload(synthetic_train_val_data, tmp_path):
    """Verify saved model artifacts reload cleanly and reproduce exact probabilities."""
    train_df, val_df = synthetic_train_val_data
    X_train, y_train = train_df[APPROVED_INPUT_FEATURES], train_df["click"].values
    X_val = val_df[APPROVED_INPUT_FEATURES]

    # 1. Train and save constant baseline
    const_model = ConstantBaseline().fit(X_train, y_train)
    const_path = tmp_path / "constant_baseline.joblib"
    joblib.dump(const_model, const_path)

    # 2. Train and save logistic regression
    lr_pipeline = build_logistic_regression_pipeline(max_iter=50)
    lr_pipeline.fit(X_train, y_train)
    lr_path = tmp_path / "logistic_regression.joblib"
    joblib.dump(lr_pipeline, lr_path)

    # 3. Reload models
    reloaded_const = joblib.load(const_path)
    reloaded_lr = joblib.load(lr_path)

    # 4. Assert predictions match exactly
    const_preds_original = const_model.predict_proba(X_val)
    const_preds_reloaded = reloaded_const.predict_proba(X_val)
    np.testing.assert_allclose(const_preds_original, const_preds_reloaded)

    lr_preds_original = lr_pipeline.predict_proba(X_val)
    lr_preds_reloaded = reloaded_lr.predict_proba(X_val)
    np.testing.assert_allclose(lr_preds_original, lr_preds_reloaded)


def test_evaluate_predictions_metrics():
    """Verify evaluation metric calculations."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

    metrics = evaluate_predictions(y_true, y_prob)
    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] == 1.0  # Perfect ranking order in this toy sample
    assert 0.0 <= metrics["brier_score"] <= 1.0
