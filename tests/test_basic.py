"""Basic verification tests for AdLift setup and schema."""

import io
import gzip
import csv
import os
import pytest
import adlift
from adlift.schema import RAW_COLUMNS, RAW_COLUMN_TYPES, TARGET_LABEL


def test_package_metadata():
    """Verify package version and name."""
    assert adlift.__version__ == "0.1.0"


def test_schema_definitions():
    """Verify raw schema definition matches observed Criteo dataset structure."""
    assert len(RAW_COLUMNS) == 22
    assert TARGET_LABEL == "click"
    assert TARGET_LABEL in RAW_COLUMNS
    assert "timestamp" in RAW_COLUMNS
    assert "uid" in RAW_COLUMNS
    assert "campaign" in RAW_COLUMNS
    for i in range(1, 10):
        assert f"cat{i}" in RAW_COLUMNS

    # Ensure all columns have typed specifications
    for col in RAW_COLUMNS:
        assert col in RAW_COLUMN_TYPES


def test_synthetic_gzip_reading(tmp_path):
    """Verify gzip TSV reader works on compressed stream without error."""
    test_file = tmp_path / "test_sample.tsv.gz"
    sample_rows = [
        ["timestamp", "uid", "campaign", "click"],
        ["0", "12345", "999", "0"],
        ["5", "67890", "888", "1"],
    ]

    with gzip.open(test_file, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(sample_rows)

    with gzip.open(test_file, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        rows = list(reader)

    assert headers == ["timestamp", "uid", "campaign", "click"]
    assert len(rows) == 2
    assert rows[0] == ["0", "12345", "999", "0"]
    assert rows[1] == ["5", "67890", "888", "1"]


def test_expected_directories():
    """Verify key project directories exist."""
    required_dirs = [
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        "reports",
        os.path.join("src", "adlift"),
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} does not exist"
