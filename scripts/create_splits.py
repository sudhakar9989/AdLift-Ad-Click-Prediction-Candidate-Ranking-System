"""CLI script to generate reproducible chronological splits from raw Criteo dataset."""

import argparse
import os
import sys

import pandas as pd

from adlift.split import create_chronological_splits

DEFAULT_RAW_PATH = os.path.join("data", "raw", "criteo_attribution_dataset.tsv.gz")
DEFAULT_OUTPUT_DIR = os.path.join("data", "processed")
DEFAULT_REPORT_PATH = os.path.join("reports", "data_split_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chronological train/validation/test splits from Criteo dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_RAW_PATH,
        help="Path to compressed raw dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Row limit for small-data laptop mode (default: 100,000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for testing (default: 0.15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save split CSVs (default: data/processed)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to save summary JSON (default: reports/data_split_summary.json)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: Dataset file not found at: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.limit:,} rows from: {args.data_path}")
    raw_df = pd.read_csv(args.data_path, sep="\t", compression="gzip", nrows=args.limit)
    print(f"Loaded {len(raw_df):,} rows successfully.")

    train_df, val_df, test_df, report = create_chronological_splits(
        df=raw_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )

    cv = report["chronological_verification"]
    print("\n" + "=" * 70)
    print("                 CHRONOLOGICAL SPLIT REPORT                ")
    print("=" * 70)
    print("Split Verification Invariant:")
    print(f"  Train max timestamp ({cv['max_train_timestamp']:,}) <= Val min timestamp ({cv['min_validation_timestamp']:,}): {cv['train_val_chronological_valid']}")
    print(f"  Val max timestamp   ({cv['max_validation_timestamp']:,}) <= Test min timestamp ({cv['min_test_timestamp']:,}): {cv['val_test_chronological_valid']}")
    print("-" * 70)
    print(f"{'Split':<12} | {'Rows':<8} | {'Clicks':<8} | {'Click Rate':<11} | {'Time Span':<12} | {'Distinct Users':<14} | {'Campaigns'}")
    print("-" * 70)
    for name, stats in report["splits"].items():
        print(
            f"{name.capitalize():<12} | "
            f"{stats['row_count']:<8,} | "
            f"{stats['total_clicks']:<8,} | "
            f"{stats['click_rate']*100:<10.3f}% | "
            f"{stats['timestamp_span_hours']:<5.2f} hours  | "
            f"{stats['distinct_users']:<14,} | "
            f"{stats['distinct_campaigns']:,}"
        )
    print("-" * 70)
    print("Entity Overlap with Training Set:")
    od = report["overlap_diagnostics"]
    print(f"  Validation users seen in train : {od['validation_users_seen_in_train']['count']:,} ({od['validation_users_seen_in_train']['percentage_of_val_users']}%)")
    print(f"  Test users seen in train       : {od['test_users_seen_in_train']['count']:,} ({od['test_users_seen_in_train']['percentage_of_test_users']}%)")
    print(f"  Validation campaigns in train  : {od['validation_campaigns_seen_in_train']['count']:,} ({od['validation_campaigns_seen_in_train']['percentage_of_val_campaigns']}%)")
    print(f"  Test campaigns in train        : {od['test_campaigns_seen_in_train']['count']:,} ({od['test_campaigns_seen_in_train']['percentage_of_test_campaigns']}%)")
    print("-" * 70)
    print(f"Small-Data Scope: {report['small_data_scope_and_limits']['dataset_portion_percentage']}% of full Criteo dataset ({report['small_data_scope_and_limits']['temporal_span_hours']} continuous hours).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
