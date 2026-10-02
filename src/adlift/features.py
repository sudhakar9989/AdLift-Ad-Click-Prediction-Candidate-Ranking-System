"""Feature specification, categorization, and strict leakage prevention allowlists for AdLift."""

from enum import Enum
from typing import Dict, Iterable, List, Set
import pandas as pd

from adlift.schema import RAW_COLUMNS, TARGET_LABEL


class FeatureCategory(str, Enum):
    """Classification of columns according to impression-time availability and safety."""
    ELIGIBLE_INPUT = "eligible_input"
    LABEL = "label"
    POST_IMPRESSION_OUTCOME = "post_impression_outcome"
    METADATA_IDENTIFIER = "metadata_identifier"
    UNCERTAIN_TIMING = "uncertain_timing"


# Detailed specification for every raw column observed in criteo_attribution_dataset.tsv.gz
FEATURE_SPEC: Dict[str, Dict[str, str]] = {
    "timestamp": {
        "category": FeatureCategory.METADATA_IDENTIFIER,
        "type": "int64",
        "description": "Elapsed seconds from dataset start. Used strictly for chronological train/val/test splitting. Excluded as raw linear/tabular input to prevent temporal trend memorization.",
    },
    "uid": {
        "category": FeatureCategory.METADATA_IDENTIFIER,
        "type": "int64",
        "description": "Hashed user ID. Extremely high cardinality (~87% unique in small slices). Excluded from direct tabular inputs to avoid severe overfitting and cold-start failure.",
    },
    "campaign": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Hashed campaign ID. Known at auction/scoring time. High predictive relevance for ad creative and advertiser intent.",
    },
    "conversion": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Conversion indicator (0 or 1). Occurs downstream after a click/visit. STRICT LEAKAGE - EXCLUDED.",
    },
    "conversion_timestamp": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Timestamp of downstream conversion. STRICT LEAKAGE - EXCLUDED.",
    },
    "conversion_id": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Identifier of downstream conversion. STRICT LEAKAGE - EXCLUDED.",
    },
    "attribution": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Conversion attribution label (0 or 1). STRICT LEAKAGE - EXCLUDED.",
    },
    "click": {
        "category": FeatureCategory.LABEL,
        "type": "int64",
        "description": "Target prediction label (1 = clicked, 0 = impression not clicked). NEVER an input feature.",
    },
    "click_pos": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Position of click in the attribution path (-1 if no click). Complete target leakage - EXCLUDED.",
    },
    "click_nb": {
        "category": FeatureCategory.POST_IMPRESSION_OUTCOME,
        "type": "int64",
        "description": "Number of clicks in conversion path (-1 if no click). Complete target leakage - EXCLUDED.",
    },
    "cost": {
        "category": FeatureCategory.UNCERTAIN_TIMING,
        "type": "float64",
        "description": "Impression cost. In ad exchanges, realized clearing price is determined post-auction/post-impression. EXCLUDED to prevent leakage.",
    },
    "cpo": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "float64",
        "sub_type": "numeric",
        "description": "Cost-per-order / target bid constraint set by the advertiser before auction. Known at impression time.",
    },
    "time_since_last_click": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "numeric",
        "description": "Seconds since user's previous recorded click (-1 if no prior click). Pre-impression user recency feature.",
    },
    "cat1": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature (e.g. publisher / placement). Available at impression time.",
    },
    "cat2": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat3": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat4": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat5": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat6": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat7": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat8": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
    "cat9": {
        "category": FeatureCategory.ELIGIBLE_INPUT,
        "type": "int64",
        "sub_type": "categorical",
        "description": "Anonymized contextual categorical feature. Available at impression time.",
    },
}

# Explicit allowlist of input features for training & inference
APPROVED_NUMERIC_FEATURES: List[str] = [
    "cpo",
    "time_since_last_click",
]

APPROVED_CATEGORICAL_FEATURES: List[str] = [
    "campaign",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
]

APPROVED_INPUT_FEATURES: List[str] = APPROVED_NUMERIC_FEATURES + APPROVED_CATEGORICAL_FEATURES

# Strict set of prohibited columns that must NEVER enter model inputs
PROHIBITED_FEATURES: Set[str] = {
    TARGET_LABEL,
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click_pos",
    "click_nb",
    "cost",
}

# Special identifiers that require split or aggregation treatment rather than raw inputs
IDENTIFIER_COLUMNS: Set[str] = {
    "timestamp",
    "uid",
}


def validate_feature_columns(columns: Iterable[str]) -> List[str]:
    """Validate that requested feature columns belong strictly to the approved allowlist.

    Raises:
        ValueError: If any prohibited, outcome, label, or unapproved column is present.
    """
    col_set = set(columns)
    forbidden_present = col_set.intersection(PROHIBITED_FEATURES)
    if forbidden_present:
        raise ValueError(
            f"TARGET LEAKAGE VIOLATION: Prohibited column(s) detected in model inputs: {sorted(forbidden_present)}. "
            f"These represent labels, post-impression outcomes, or uncertain pricing data."
        )

    identifier_present = col_set.intersection(IDENTIFIER_COLUMNS)
    if identifier_present:
        raise ValueError(
            f"IDENTIFIER/SPLIT VIOLATION: Direct identifier/temporal column(s) detected: {sorted(identifier_present)}. "
            f"Use 'timestamp' only for chronological partitioning, and handle 'uid' only via aggregates."
        )

    unapproved = col_set.difference(set(APPROVED_INPUT_FEATURES))
    if unapproved:
        raise ValueError(
            f"UNAPPROVED FEATURES: Column(s) not in allowlist: {sorted(unapproved)}. "
            f"Approved features are: {APPROVED_INPUT_FEATURES}"
        )

    return [c for c in APPROVED_INPUT_FEATURES if c in col_set]


def filter_eligible_features(
    df: pd.DataFrame,
    include_label: bool = False,
) -> pd.DataFrame:
    """Filter a dataframe to contain only approved input features (and optionally target label).

    Args:
        df: Input raw or preprocessed dataframe.
        include_label: Whether to include the target label column (for training/evaluation).

    Returns:
        DataFrame containing strictly allowlisted features (and label if requested).
    """
    missing_approved = [c for c in APPROVED_INPUT_FEATURES if c in df.columns]
    if not missing_approved:
        raise ValueError(f"None of the approved features {APPROVED_INPUT_FEATURES} were found in the dataframe.")

    cols_to_select = [c for c in APPROVED_INPUT_FEATURES if c in df.columns]
    if include_label:
        if TARGET_LABEL not in df.columns:
            raise KeyError(f"Target label '{TARGET_LABEL}' missing from dataframe.")
        cols_to_select = cols_to_select + [TARGET_LABEL]

    return df[cols_to_select].copy()
