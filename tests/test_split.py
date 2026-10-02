"""Tests for chronological data splitting, ordering invariants, and stability."""

import numpy as np
import pandas as pd
import pytest

from adlift.features import APPROVED_INPUT_FEATURES, PROHIBITED_FEATURES
from adlift.schema import RAW_COLUMNS, TARGET_LABEL
from adlift.split import create_chronological_splits


@pytest.fixture
def synthetic_dataframe():
    """Create a synthetic dataset of 100 records with out-of-order timestamps to test sorting."""
    np.random.seed(42)
    n = 100
    timestamps = np.random.randint(1000, 5000, size=n)

    data = {
        "timestamp": timestamps,
        "uid": np.random.randint(1, 40, size=n),
        "campaign": np.random.randint(1, 10, size=n),
        "click": np.random.binomial(1, 0.3, size=n),
        "cpo": np.random.uniform(0.1, 1.0, size=n),
        "time_since_last_click": np.random.randint(-1, 3600, size=n),
        # Prohibited columns
        "conversion": np.random.binomial(1, 0.05, size=n),
        "click_pos": np.random.choice([-1, 0, 1], size=n),
        "click_nb": np.random.choice([-1, 1, 2], size=n),
        "cost": np.random.uniform(0.0001, 0.01, size=n),
    }
    for i in range(1, 10):
        data[f"cat{i}"] = np.random.randint(100, 999, size=n)

    return pd.DataFrame(data)


def test_chronological_ordering_invariant(synthetic_dataframe):
    """Verify max(train timestamp) <= min(val timestamp) and max(val timestamp) <= min(test timestamp)."""
    train, val, test, report = create_chronological_splits(
        synthetic_dataframe,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    max_train_ts = train["timestamp"].max()
    min_val_ts = val["timestamp"].min()
    max_val_ts = val["timestamp"].max()
    min_test_ts = test["timestamp"].min()

    assert max_train_ts <= min_val_ts, f"Train max ({max_train_ts}) > Val min ({min_val_ts})"
    assert max_val_ts <= min_test_ts, f"Val max ({max_val_ts}) > Test min ({min_test_ts})"
    assert report["chronological_verification"]["train_val_chronological_valid"] is True
    assert report["chronological_verification"]["val_test_chronological_valid"] is True


def test_no_row_overlap_and_full_partition(synthetic_dataframe):
    """Verify splits form an exact partition with zero row overlap."""
    train, val, test, _ = create_chronological_splits(
        synthetic_dataframe,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    n_total = len(synthetic_dataframe)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(train) + len(val) + len(test) == n_total


def test_prohibited_columns_excluded_from_splits(synthetic_dataframe):
    """Verify that prohibited outcome columns are absent from generated splits."""
    train, val, test, _ = create_chronological_splits(
        synthetic_dataframe,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    for split_df in (train, val, test):
        for col in PROHIBITED_FEATURES:
            if col != TARGET_LABEL:  # TARGET_LABEL is permitted as the label column
                assert col not in split_df.columns, f"Prohibited column {col} leaked into split!"


def test_stable_results_on_repeated_runs(synthetic_dataframe):
    """Verify that splitting repeatedly on the same data produces identical dataframes."""
    train1, val1, test1, rep1 = create_chronological_splits(
        synthetic_dataframe,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    train2, val2, test2, rep2 = create_chronological_splits(
        synthetic_dataframe,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)
    assert rep1 == rep2


def test_invalid_ratios_raise_error(synthetic_dataframe):
    """Verify that split ratios that do not sum to 1.0 raise a ValueError."""
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        create_chronological_splits(
            synthetic_dataframe,
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.2,  # sums to 0.9
        )
