"""CLI script to train and evaluate click-prediction baselines on validation split."""

import argparse
import os
import sys

from adlift.models.baselines import train_and_evaluate_baselines

DEFAULT_TRAIN_PATH = os.path.join("data", "processed", "train.csv")
DEFAULT_VAL_PATH = os.path.join("data", "processed", "validation.csv")
DEFAULT_MODELS_DIR = "models"
DEFAULT_REPORT_PATH = os.path.join("reports", "baseline_validation_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Constant and Logistic Regression baselines and evaluate on validation split."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to processed train.csv",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=DEFAULT_VAL_PATH,
        help="Path to processed validation.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to save metrics JSON report",
    )

    args = parser.parse_args()

    for p in [args.train_path, args.val_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            print("Please run scripts/create_splits.py first.", file=sys.stderr)
            sys.exit(1)

    results = train_and_evaluate_baselines(
        train_path=args.train_path,
        val_path=args.val_path,
        models_dir=args.models_dir,
        reports_path=args.report_path,
    )

    print("\n" + "=" * 75)
    print("                    BASELINE VALIDATION RESULTS                    ")
    print("=" * 75)
    print(f"{'Model':<22} | {'Log Loss':<9} | {'ROC AUC':<9} | {'PR AUC':<9} | {'Brier':<9} | {'Train Time'}")
    print("-" * 75)
    for model_name, metrics in results["models"].items():
        print(
            f"{model_name:<22} | "
            f"{metrics['log_loss']:<9.4f} | "
            f"{metrics['roc_auc']:<9.4f} | "
            f"{metrics['pr_auc']:<9.4f} | "
            f"{metrics['brier_score']:<9.4f} | "
            f"{metrics['train_time_seconds']}s"
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
