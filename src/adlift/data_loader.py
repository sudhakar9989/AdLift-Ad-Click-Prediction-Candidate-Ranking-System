"""Safe chunked data inspection and loading for compressed Criteo datasets."""

import json
import os
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from adlift.schema import RAW_COLUMNS, TARGET_LABEL


def inspect_compressed_tsv(
    filepath: str,
    row_limit: int = 100_000,
    chunk_size: int = 25_000,
    output_summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect a compressed TSV dataset in streaming chunks without loading all rows into memory.

    Args:
        filepath: Path to the .tsv.gz file.
        row_limit: Maximum number of rows to inspect.
        chunk_size: Number of rows per chunk.
        output_summary_path: Optional path to save the JSON summary report.

    Returns:
        Dictionary containing inspection metrics and column schema.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    total_inspected = 0
    total_clicks = 0
    min_timestamp = float("inf")
    max_timestamp = float("-inf")
    distinct_uids: Set[int] = set()
    distinct_campaigns: Set[int] = set()
    missing_counts: Dict[str, int] = {}
    sentinel_negative_counts: Dict[str, int] = {}
    inferred_types: Dict[str, str] = {}
    columns_observed = []

    print(f"Streaming inspection of: {filepath}")
    print(f"Row limit: {row_limit:,} | Chunk size: {chunk_size:,}\n")

    # Read in chunks to strictly control peak memory usage
    chunk_iter = pd.read_csv(
        filepath,
        sep="\t",
        compression="gzip",
        chunksize=chunk_size,
    )

    for chunk_idx, chunk in enumerate(chunk_iter):
        rows_in_chunk = len(chunk)
        if total_inspected + rows_in_chunk > row_limit:
            rows_needed = row_limit - total_inspected
            chunk = chunk.iloc[:rows_needed]
            rows_in_chunk = len(chunk)

        if chunk_idx == 0:
            columns_observed = list(chunk.columns)
            missing_counts = {col: 0 for col in columns_observed}
            sentinel_negative_counts = {col: 0 for col in columns_observed}
            inferred_types = {col: str(dtype) for col, dtype in chunk.dtypes.items()}

        # Accumulate missing counts
        for col in columns_observed:
            missing_counts[col] += int(chunk[col].isna().sum())
            if np.issubdtype(chunk[col].dtype, np.number):
                sentinel_negative_counts[col] += int((chunk[col] == -1).sum())

        # Accumulate label (click) metrics
        if TARGET_LABEL in chunk.columns:
            total_clicks += int(chunk[TARGET_LABEL].sum())

        # Accumulate distinct identifiers (bounded by row_limit on integer IDs)
        if "uid" in chunk.columns:
            distinct_uids.update(chunk["uid"].dropna().astype(np.int64).unique())
        if "campaign" in chunk.columns:
            distinct_campaigns.update(chunk["campaign"].dropna().astype(np.int64).unique())

        # Accumulate timestamp range
        if "timestamp" in chunk.columns:
            chunk_min = float(chunk["timestamp"].min())
            chunk_max = float(chunk["timestamp"].max())
            if chunk_min < min_timestamp:
                min_timestamp = chunk_min
            if chunk_max > max_timestamp:
                max_timestamp = chunk_max

        total_inspected += rows_in_chunk
        print(f"  Processed chunk {chunk_idx + 1:2d}: {total_inspected:,} / {row_limit:,} rows...")

        if total_inspected >= row_limit:
            break

    click_rate = (total_clicks / total_inspected) if total_inspected > 0 else 0.0
    ts_min = int(min_timestamp) if min_timestamp != float("inf") else 0
    ts_max = int(max_timestamp) if max_timestamp != float("-inf") else 0

    summary: Dict[str, Any] = {
        "dataset_file": os.path.basename(filepath),
        "total_rows_inspected": total_inspected,
        "target_label": TARGET_LABEL,
        "total_clicks": total_clicks,
        "click_rate": round(click_rate, 6),
        "distinct_users_count": len(distinct_uids),
        "distinct_campaigns_count": len(distinct_campaigns),
        "timestamp_range": {
            "min_timestamp": ts_min,
            "max_timestamp": ts_max,
            "span_seconds": ts_max - ts_min,
        },
        "columns_count": len(columns_observed),
        "columns": columns_observed,
        "inferred_types": inferred_types,
        "missing_value_counts": missing_counts,
        "sentinel_negative_one_counts": sentinel_negative_counts,
    }

    # Print human-readable summary without leaking raw user IDs
    print("\n" + "=" * 65)
    print("                DATA INSPECTION SUMMARY                ")
    print("=" * 65)
    print(f"Target Label                  : {summary['target_label']}")
    print(f"Total Rows Inspected          : {summary['total_rows_inspected']:,}")
    print(f"Total Clicks Observed         : {summary['total_clicks']:,}")
    print(f"Empirical Click Rate          : {summary['click_rate'] * 100:.3f}%")
    print(f"Distinct Users (inspected)    : {summary['distinct_users_count']:,}")
    print(f"Distinct Campaigns (inspected): {summary['distinct_campaigns_count']:,}")
    print(f"Timestamp Range               : [{ts_min:,} -> {ts_max:,}] ({ts_max - ts_min:,} seconds span)")
    print(f"Total Columns                 : {summary['columns_count']}")
    print("-" * 65)
    print("Column Types and Missing Value Counts:")
    for col in columns_observed:
        null_cnt = missing_counts.get(col, 0)
        neg_cnt = sentinel_negative_counts.get(col, 0)
        dtype = inferred_types.get(col, "unknown")
        print(f"  {col:>22} | type: {dtype:<8} | nulls: {null_cnt:<5} | -1 sentinels: {neg_cnt:<6}")
    print("=" * 65 + "\n")

    if output_summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_summary_path)), exist_ok=True)
        with open(output_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {output_summary_path}")

    return summary
