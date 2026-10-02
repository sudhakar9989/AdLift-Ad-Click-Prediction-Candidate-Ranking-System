"""Inspect the first few rows of the compressed Criteo dataset."""

import argparse
import csv
import gzip
import os
import sys

DEFAULT_DATA_PATH = os.path.join("data", "raw", "criteo_attribution_dataset.tsv.gz")


def inspect_dataset(data_path: str = DEFAULT_DATA_PATH, num_rows: int = 5) -> None:
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset file not found at: {data_path}", file=sys.stderr)
        print("Expected location: data/raw/criteo_attribution_dataset.tsv.gz", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting file: {data_path}")
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"Compressed file size: {file_size_mb:.2f} MB\n")

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            headers = next(reader)
        except StopIteration:
            print("File is empty!", file=sys.stderr)
            return

        print(f"Found {len(headers)} columns:")
        for idx, col in enumerate(headers):
            print(f"  [{idx:2d}] {col}")

        print(f"\nPrinting first {num_rows} data rows:\n" + "-" * 60)
        for row_idx in range(1, num_rows + 1):
            try:
                row = next(reader)
            except StopIteration:
                break
            print(f"--- Row {row_idx} ---")
            for col, val in zip(headers, row):
                print(f"  {col:>22}: {val}")
        print("-" * 60)
        print("Inspection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect head rows of Criteo dataset")
    parser.add_argument("--path", type=str, default=DEFAULT_DATA_PATH, help="Path to .tsv.gz file")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to display")
    args = parser.parse_args()
    inspect_dataset(args.path, args.rows)
