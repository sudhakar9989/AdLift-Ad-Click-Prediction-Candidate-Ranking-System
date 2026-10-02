"""Tests for chunked data inspection and loader."""

import csv
import gzip
import json
import os
import pytest

from adlift.data_loader import inspect_compressed_tsv
from adlift.schema import RAW_COLUMNS, TARGET_LABEL


@pytest.fixture
def synthetic_criteo_gzip_tsv(tmp_path):
    """Creates a tiny synthetic gzip TSV fixture adhering to the 22-column schema."""
    tsv_gz_path = tmp_path / "synthetic_criteo.tsv.gz"

    # Define rows: 10 rows with known values
    rows = []
    for i in range(10):
        # alternate clicks: rows 0, 2, 4, 6, 8 are clicks (5 total clicks)
        is_click = 1 if i % 2 == 0 else 0
        ts = 1000 + i * 10  # 1000, 1010, ..., 1090
        uid = 100 + (i % 4)  # 4 distinct users: 100, 101, 102, 103
        campaign = 50 + (i % 2)  # 2 distinct campaigns: 50, 51

        row = {
            "timestamp": ts,
            "uid": uid,
            "campaign": campaign,
            "conversion": 1 if is_click and i == 0 else 0,
            "conversion_timestamp": 2000 if is_click and i == 0 else -1,
            "conversion_id": 9999 if is_click and i == 0 else -1,
            "attribution": 1 if is_click and i == 0 else 0,
            "click": is_click,
            "click_pos": 0 if is_click else -1,
            "click_nb": 2 if is_click else -1,
            "cost": 0.0001 * (i + 1),
            "cpo": 0.5,
            "time_since_last_click": 500 if i > 3 else -1,
            "cat1": 1111,
            "cat2": 2222,
            "cat3": 3333,
            "cat4": 4444,
            "cat5": 5555,
            "cat6": 6666,
            "cat7": 7777,
            "cat8": 8888,
            "cat9": 9999,
        }
        rows.append([str(row[col]) for col in RAW_COLUMNS])

    with gzip.open(tsv_gz_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)

    return str(tsv_gz_path)


def test_inspect_compressed_tsv_metrics(synthetic_criteo_gzip_tsv, tmp_path):
    """Test chunked inspector metrics with exact expected values."""
    output_summary = str(tmp_path / "summary.json")

    summary = inspect_compressed_tsv(
        filepath=synthetic_criteo_gzip_tsv,
        row_limit=10,
        chunk_size=3,  # Forces multiple chunks (3 + 3 + 3 + 1)
        output_summary_path=output_summary,
    )

    assert summary["target_label"] == "click"
    assert summary["total_rows_inspected"] == 10
    assert summary["total_clicks"] == 5
    assert summary["click_rate"] == 0.5
    assert summary["distinct_users_count"] == 4
    assert summary["distinct_campaigns_count"] == 2
    assert summary["timestamp_range"]["min_timestamp"] == 1000
    assert summary["timestamp_range"]["max_timestamp"] == 1090
    assert summary["timestamp_range"]["span_seconds"] == 90
    assert summary["columns_count"] == 22
    assert summary["columns"] == RAW_COLUMNS

    # Check that summary file was saved
    assert os.path.exists(output_summary)
    with open(output_summary, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["click_rate"] == 0.5


def test_inspect_row_limit(synthetic_criteo_gzip_tsv):
    """Test that inspector strictly respects the row_limit."""
    summary = inspect_compressed_tsv(
        filepath=synthetic_criteo_gzip_tsv,
        row_limit=4,
        chunk_size=2,
    )
    assert summary["total_rows_inspected"] == 4
    # Rows 0, 1, 2, 3 -> clicks at 0, 2 -> 2 clicks
    assert summary["total_clicks"] == 2
    assert summary["click_rate"] == 0.5
