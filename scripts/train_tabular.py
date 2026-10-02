"""CLI script to train LightGBM tabular model and produce 3-model comparison report."""

import argparse
import os
import sys

from adlift.models.lightgbm_model import train_and_evaluate_tabular

DEFAULT_TRAIN_PATH = os.path.join("data", "processed", "train.csv")
DEFAULT_VAL_PATH = os.path.join("data", "processed", "validation.csv")
DEFAULT_BASELINE_REPORT = os.path.join("reports", "baseline_validation_metrics.json")
DEFAULT_MODELS_DIR = "models"
DEFAULT_COMPARISON_JSON = os.path.join("reports", "model_comparison.json")
DEFAULT_COMPARISON_MD = os.path.join("reports", "model_comparison.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM tabular model and output experiment comparison table."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to train.csv",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=DEFAULT_VAL_PATH,
        help="Path to validation.csv",
    )
    parser.add_argument(
        "--baseline-report",
        type=str,
        default=DEFAULT_BASELINE_REPORT,
        help="Path to baseline validation metrics JSON",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_COMPARISON_JSON,
        help="Path to save comparison JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=DEFAULT_COMPARISON_MD,
        help="Path to save comparison Markdown",
    )

    args = parser.parse_args()

    for p in [args.train_path, args.val_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            print("Please run scripts/create_splits.py first.", file=sys.stderr)
            sys.exit(1)

    table = train_and_evaluate_tabular(
        train_path=args.train_path,
        val_path=args.val_path,
        baseline_metrics_path=args.baseline_report,
        models_dir=args.models_dir,
        comparison_report_path=args.output_json,
        comparison_md_path=args.output_md,
    )

    print("\n" + "=" * 80)
    print("                     MODEL VALIDATION COMPARISON                     ")
    print("=" * 80)
    print(f"{'Model':<22} | {'Log Loss':<9} | {'ROC AUC':<9} | {'PR AUC':<9} | {'Brier':<9} | {'Train Time'}")
    print("-" * 80)
    for model_name, metrics in table["models"].items():
        print(
            f"{model_name:<22} | "
            f"{metrics['log_loss']:<9.4f} | "
            f"{metrics['roc_auc']:<9.4f} | "
            f"{metrics['pr_auc']:<9.4f} | "
            f"{metrics['brier_score']:<9.4f} | "
            f"{metrics['train_time_seconds']}s"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
