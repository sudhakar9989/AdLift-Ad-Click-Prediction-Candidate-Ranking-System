"""Reproducible chronological splitting and temporal partition diagnostics for AdLift."""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from adlift.features import APPROVED_INPUT_FEATURES, PROHIBITED_FEATURES
from adlift.schema import TARGET_LABEL


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create reproducible chronological train, validation, and test splits based on timestamp.

    Args:
        df: Raw or filtered dataframe containing 'timestamp', 'click', and features.
        train_ratio: Proportion of earliest records for training (e.g., 0.70).
        val_ratio: Proportion of middle records for validation (e.g., 0.15).
        test_ratio: Proportion of latest records for held-out test (e.g., 0.15).
        output_dir: Directory to save train.csv, validation.csv, and test.csv.
        report_path: Path to save data_split_summary.json.

    Returns:
        Tuple of (train_df, val_df, test_df, split_report_dict).
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    if "timestamp" not in df.columns:
        raise KeyError("Dataframe must contain 'timestamp' column for chronological splitting.")
    if TARGET_LABEL not in df.columns:
        raise KeyError(f"Dataframe must contain target label '{TARGET_LABEL}'.")

    # 1. Enforce strict chronological ordering (stable mergesort)
    sorted_df = df.sort_values(by="timestamp", kind="mergesort").reset_index(drop=True)

    n_total = len(sorted_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # 2. Strict chronological verification invariant
    max_train_ts = int(train_df["timestamp"].max())
    min_val_ts = int(val_df["timestamp"].min())
    max_val_ts = int(val_df["timestamp"].max())
    min_test_ts = int(test_df["timestamp"].min())
    max_test_ts = int(test_df["timestamp"].max())

    if max_train_ts > min_val_ts:
        raise ValueError(
            f"Chronological split violation: max train timestamp ({max_train_ts}) > "
            f"min validation timestamp ({min_val_ts})"
        )
    if max_val_ts > min_test_ts:
        raise ValueError(
            f"Chronological split violation: max validation timestamp ({max_val_ts}) > "
            f"min test timestamp ({min_test_ts})"
        )

    # 3. Clean splits: Retain timestamp, uid, approved features, and label; strip prohibited columns
    cols_to_keep = ["timestamp"]
    if "uid" in sorted_df.columns:
        cols_to_keep.append("uid")
    for feat in APPROVED_INPUT_FEATURES:
        if feat in sorted_df.columns:
            cols_to_keep.append(feat)
    cols_to_keep.append(TARGET_LABEL)

    train_clean = train_df[cols_to_keep].copy()
    val_clean = val_df[cols_to_keep].copy()
    test_clean = test_df[cols_to_keep].copy()

    # 4. Compute split metrics and overlap diagnostics
    def _split_stats(split_data: pd.DataFrame) -> Dict[str, Any]:
        clicks = int(split_data[TARGET_LABEL].sum())
        total = len(split_data)
        rate = round(clicks / total, 6) if total > 0 else 0.0
        ts_min = int(split_data["timestamp"].min())
        ts_max = int(split_data["timestamp"].max())
        users = set(split_data["uid"].dropna().unique()) if "uid" in split_data.columns else set()
        campaigns = set(split_data["campaign"].dropna().unique()) if "campaign" in split_data.columns else set()
        return {
            "row_count": total,
            "total_clicks": clicks,
            "click_rate": rate,
            "timestamp_min": ts_min,
            "timestamp_max": ts_max,
            "timestamp_span_seconds": ts_max - ts_min,
            "timestamp_span_hours": round((ts_max - ts_min) / 3600.0, 2),
            "distinct_users": len(users),
            "distinct_campaigns": len(campaigns),
        }

    train_stats = _split_stats(train_clean)
    val_stats = _split_stats(val_clean)
    test_stats = _split_stats(test_clean)

    # Entity overlap calculations
    train_users = set(train_clean["uid"].unique()) if "uid" in train_clean.columns else set()
    val_users = set(val_clean["uid"].unique()) if "uid" in val_clean.columns else set()
    test_users = set(test_clean["uid"].unique()) if "uid" in test_clean.columns else set()

    train_camps = set(train_clean["campaign"].unique()) if "campaign" in train_clean.columns else set()
    val_camps = set(val_clean["campaign"].unique()) if "campaign" in val_clean.columns else set()
    test_camps = set(test_clean["campaign"].unique()) if "campaign" in test_clean.columns else set()

    val_user_overlap = len(val_users.intersection(train_users))
    test_user_overlap = len(test_users.intersection(train_users))

    val_camp_overlap = len(val_camps.intersection(train_camps))
    test_camp_overlap = len(test_camps.intersection(train_camps))

    report: Dict[str, Any] = {
        "total_records_processed": n_total,
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "chronological_verification": {
            "max_train_timestamp": max_train_ts,
            "min_validation_timestamp": min_val_ts,
            "train_val_chronological_valid": max_train_ts <= min_val_ts,
            "max_validation_timestamp": max_val_ts,
            "min_test_timestamp": min_test_ts,
            "val_test_chronological_valid": max_val_ts <= min_test_ts,
        },
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
        "overlap_diagnostics": {
            "validation_users_seen_in_train": {
                "count": val_user_overlap,
                "percentage_of_val_users": round(
                    val_user_overlap / len(val_users) * 100 if val_users else 0.0, 2
                ),
            },
            "test_users_seen_in_train": {
                "count": test_user_overlap,
                "percentage_of_test_users": round(
                    test_user_overlap / len(test_users) * 100 if test_users else 0.0, 2
                ),
            },
            "validation_campaigns_seen_in_train": {
                "count": val_camp_overlap,
                "percentage_of_val_campaigns": round(
                    val_camp_overlap / len(val_camps) * 100 if val_camps else 0.0, 2
                ),
            },
            "test_campaigns_seen_in_train": {
                "count": test_camp_overlap,
                "percentage_of_test_campaigns": round(
                    test_camp_overlap / len(test_camps) * 100 if test_camps else 0.0, 2
                ),
            },
        },
        "small_data_scope_and_limits": {
            "full_dataset_total_impressions": "~16.47 million impressions spanning 30.9 days",
            "small_data_impressions": n_total,
            "dataset_portion_percentage": round((n_total / 16468027) * 100, 3),
            "temporal_span_hours": round((max_test_ts - sorted_df['timestamp'].min()) / 3600.0, 2),
            "methodological_limits": [
                "Captures intra-day behavioral variation but cannot observe multi-day trends or day-of-week seasonality.",
                "Short time window results in high user cold-start ratio (most users observed only once).",
                "Infrequent/long-tail campaigns may have very few impressions in this window.",
                "Enables lightning-fast model training, low-latency evaluation, and zero OOM risk on standard laptop CPUs.",
            ],
        },
    }

    # Save processed splits and summary if paths provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        val_path = os.path.join(output_dir, "validation.csv")
        test_path = os.path.join(output_dir, "test.csv")

        train_clean.to_csv(train_path, index=False)
        val_clean.to_csv(val_path, index=False)
        test_clean.to_csv(test_path, index=False)
        print(f"Saved splits to {output_dir}: train.csv, validation.csv, test.csv")

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Split report saved to: {report_path}")

    return train_clean, val_clean, test_clean, report
