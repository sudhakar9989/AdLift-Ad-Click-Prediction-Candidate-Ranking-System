"""Command-line data inspection tool for AdLift."""

import argparse
import os
import sys

from adlift.data_loader import inspect_compressed_tsv

DEFAULT_DATA_PATH = os.path.join("data", "raw", "criteo_attribution_dataset.tsv.gz")
DEFAULT_SUMMARY_PATH = os.path.join("reports", "data_inspection_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect compressed Criteo TSV dataset in streaming chunks without high memory usage."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw compressed .tsv.gz file (default: data/raw/criteo_attribution_dataset.tsv.gz)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Configurable row limit to inspect (default: 100,000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Chunk size for streaming rows (default: 25,000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to save the summary JSON file (default: reports/data_inspection_summary.json)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: Dataset not found at: {args.data_path}", file=sys.stderr)
        print("Please place criteo_attribution_dataset.tsv.gz in data/raw/", file=sys.stderr)
        sys.exit(1)

    inspect_compressed_tsv(
        filepath=args.data_path,
        row_limit=args.limit,
        chunk_size=args.chunk_size,
        output_summary_path=args.output,
    )


if __name__ == "__main__":
    main()
