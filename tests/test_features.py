"""Tests for feature categorization, allowlist integrity, and target leakage prevention."""

import pandas as pd
import pytest

from adlift.features import (
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_INPUT_FEATURES,
    APPROVED_NUMERIC_FEATURES,
    FEATURE_SPEC,
    IDENTIFIER_COLUMNS,
    PROHIBITED_FEATURES,
    filter_eligible_features,
    validate_feature_columns,
)
from adlift.schema import RAW_COLUMNS, TARGET_LABEL


def test_all_raw_columns_categorized():
    """Verify every raw dataset column is cataloged in FEATURE_SPEC."""
    assert len(RAW_COLUMNS) == 22
    for col in RAW_COLUMNS:
        assert col in FEATURE_SPEC, f"Column '{col}' is missing from FEATURE_SPEC"


def test_no_leakage_overlap():
    """Verify approved inputs and prohibited/outcome features have zero overlap."""
    approved_set = set(APPROVED_INPUT_FEATURES)
    prohibited_set = set(PROHIBITED_FEATURES)
    identifier_set = set(IDENTIFIER_COLUMNS)

    assert len(approved_set.intersection(prohibited_set)) == 0, (
        f"CRITICAL LEAKAGE: Prohibited features found in allowlist: "
        f"{approved_set.intersection(prohibited_set)}"
    )
    assert len(approved_set.intersection(identifier_set)) == 0, (
        f"Direct identifiers found in allowlist: {approved_set.intersection(identifier_set)}"
    )
    assert TARGET_LABEL not in approved_set, "Target label must never be in approved inputs!"


def test_approved_feature_composition():
    """Verify approved inputs match the intended 12 features (2 numeric, 10 categorical: campaign + cat1..cat9)."""
    assert len(APPROVED_NUMERIC_FEATURES) == 2
    assert "cpo" in APPROVED_NUMERIC_FEATURES
    assert "time_since_last_click" in APPROVED_NUMERIC_FEATURES

    assert len(APPROVED_CATEGORICAL_FEATURES) == 10
    assert "campaign" in APPROVED_CATEGORICAL_FEATURES
    for i in range(1, 10):
        assert f"cat{i}" in APPROVED_CATEGORICAL_FEATURES

    assert len(APPROVED_INPUT_FEATURES) == 12


@pytest.mark.parametrize(
    "prohibited_col",
    [
        "click",
        "conversion",
        "conversion_timestamp",
        "conversion_id",
        "attribution",
        "click_pos",
        "click_nb",
        "cost",
    ],
)
def test_prohibited_columns_fail_validation(prohibited_col):
    """Ensure validate_feature_columns raises ValueError when any prohibited feature is passed."""
    candidate_features = APPROVED_INPUT_FEATURES + [prohibited_col]
    with pytest.raises(ValueError, match="TARGET LEAKAGE VIOLATION"):
        validate_feature_columns(candidate_features)


@pytest.mark.parametrize("identifier_col", ["timestamp", "uid"])
def test_identifier_columns_fail_validation(identifier_col):
    """Ensure raw timestamp and user ID raise errors if passed as direct features."""
    candidate_features = APPROVED_INPUT_FEATURES + [identifier_col]
    with pytest.raises(ValueError, match="IDENTIFIER/SPLIT VIOLATION"):
        validate_feature_columns(candidate_features)


def test_unknown_columns_fail_validation():
    """Ensure unapproved, arbitrary columns raise an allowlist error."""
    with pytest.raises(ValueError, match="UNAPPROVED FEATURES"):
        validate_feature_columns(["campaign", "cat1", "unapproved_mystery_field"])


def test_valid_columns_pass():
    """Ensure exact approved features pass validation cleanly."""
    validated = validate_feature_columns(APPROVED_INPUT_FEATURES)
    assert validated == APPROVED_INPUT_FEATURES


def test_filter_eligible_features():
    """Test filter_eligible_features retains only approved columns and drops prohibited ones."""
    # Construct a dummy dataframe containing all 22 raw columns
    data = {col: [0, 1] for col in RAW_COLUMNS}
    raw_df = pd.DataFrame(data)

    filtered = filter_eligible_features(raw_df, include_label=False)
    assert set(filtered.columns) == set(APPROVED_INPUT_FEATURES)
    for prohibited in PROHIBITED_FEATURES:
        assert prohibited not in filtered.columns

    # When include_label is requested for training
    filtered_with_label = filter_eligible_features(raw_df, include_label=True)
    assert set(filtered_with_label.columns) == set(APPROVED_INPUT_FEATURES + [TARGET_LABEL])
