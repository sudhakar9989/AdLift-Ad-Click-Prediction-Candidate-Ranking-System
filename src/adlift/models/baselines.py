"""Baseline models and evaluation utilities for click prediction."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adlift.features import (
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_INPUT_FEATURES,
    APPROVED_NUMERIC_FEATURES,
    validate_feature_columns,
)
from adlift.schema import TARGET_LABEL


class ConstantBaseline(BaseEstimator, ClassifierMixin):
    """Predicts a constant probability equal to the empirical training set click rate."""

    def __init__(self) -> None:
        self.prior_: float = 0.5
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: Any, y: np.ndarray) -> "ConstantBaseline":
        y_arr = np.asarray(y)
        self.prior_ = float(np.mean(y_arr))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        p1 = np.full((n_samples, 1), self.prior_, dtype=np.float64)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate key offline evaluation metrics for binary click probability predictions.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted click probabilities in range [0, 1].

    Returns:
        Dict containing log_loss, roc_auc, pr_auc, and brier_score.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    # For constant predictions, ROC AUC is defined as 0.5 and PR AUC as empirical base rate
    if len(np.unique(y_prob)) <= 1:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    loss = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "log_loss": round(loss, 6),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier_score": round(brier, 6),
    }


def build_logistic_regression_pipeline(
    C: float = 0.1,
    max_iter: int = 500,
    min_frequency: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Construct a scikit-learn Pipeline with preprocessing strictly for approved features.

    Preprocessing:
    - time_since_last_click: SimpleImputer treats -1 sentinel as missing, adds indicator, and scales.
    - cpo: Median imputer + StandardScaler.
    - Categorical features: OneHotEncoder with min_frequency to group rare hashes, handle_unknown='ignore'.
    - Classifier: L2-regularized LogisticRegression.
    """
    # 1. Pipeline for time_since_last_click handling sentinel -1
    ts_pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=-1, strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    # 2. Pipeline for cpo
    cpo_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 3. Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=min_frequency,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cpo_prep", cpo_pipeline, ["cpo"]),
            ("ts_prep", ts_pipeline, ["time_since_last_click"]),
            ("cat_prep", cat_pipeline, APPROVED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def train_and_evaluate_baselines(
    train_path: str = os.path.join("data", "processed", "train.csv"),
    val_path: str = os.path.join("data", "processed", "validation.csv"),
    models_dir: str = "models",
    reports_path: str = os.path.join("reports", "baseline_validation_metrics.json"),
) -> Dict[str, Any]:
    """Train Constant and Logistic Regression baselines on train split and evaluate on validation."""
    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading validation data: {val_path}")
    val_df = pd.read_csv(val_path)

    # Validate that only approved features enter model inputs
    validate_feature_columns(APPROVED_INPUT_FEATURES)

    X_train = train_df[APPROVED_INPUT_FEATURES]
    y_train = train_df[TARGET_LABEL].values

    X_val = val_df[APPROVED_INPUT_FEATURES]
    y_val = val_df[TARGET_LABEL].values

    results: Dict[str, Any] = {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "approved_features": APPROVED_INPUT_FEATURES,
        "models": {},
    }

    os.makedirs(models_dir, exist_ok=True)

    # 1. Constant Baseline
    print("\n--- Training Constant Baseline ---")
    t0 = time.time()
    constant_model = ConstantBaseline()
    constant_model.fit(X_train, y_train)
    const_train_time = time.time() - t0

    const_val_probs = constant_model.predict_proba(X_val)[:, 1]
    const_metrics = evaluate_predictions(y_val, const_val_probs)
    const_metrics["train_time_seconds"] = round(const_train_time, 4)
    const_metrics["prior_click_rate"] = round(constant_model.prior_, 6)
    results["models"]["ConstantBaseline"] = const_metrics

    const_artifact_path = os.path.join(models_dir, "constant_baseline.joblib")
    joblib.dump(constant_model, const_artifact_path)
    print(f"Saved Constant Baseline to {const_artifact_path}")
    print(f"  Metrics: {const_metrics}")

    # 2. Logistic Regression Baseline
    print("\n--- Training Regularized Logistic Regression ---")
    t0 = time.time()
    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)
    lr_train_time = time.time() - t0

    lr_val_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_metrics = evaluate_predictions(y_val, lr_val_probs)
    lr_metrics["train_time_seconds"] = round(lr_train_time, 4)
    lr_metrics["hyperparameters"] = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "lbfgs",
        "min_frequency": 5,
    }
    results["models"]["LogisticRegression"] = lr_metrics

    lr_artifact_path = os.path.join(models_dir, "logistic_regression.joblib")
    joblib.dump(lr_pipeline, lr_artifact_path)
    print(f"Saved Logistic Regression to {lr_artifact_path}")
    print(f"  Metrics: {lr_metrics}")

    # Save summary report
    if reports_path:
        os.makedirs(os.path.dirname(os.path.abspath(reports_path)), exist_ok=True)
        with open(reports_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved validation metrics report to {reports_path}")

    return results
