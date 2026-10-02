"""CLI script to train compact PyTorch EmbeddingMLP model and update 4-model comparison report."""

import argparse
import os
import sys

from adlift.models.pytorch_model import train_and_evaluate_pytorch

DEFAULT_TRAIN_PATH = os.path.join("data", "processed", "train.csv")
DEFAULT_VAL_PATH = os.path.join("data", "processed", "validation.csv")
DEFAULT_MODELS_DIR = "models"
DEFAULT_COMPARISON_JSON = os.path.join("reports", "model_comparison.json")
DEFAULT_COMPARISON_MD = os.path.join("reports", "model_comparison.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PyTorch EmbeddingMLP click-prediction model and update model comparison table."
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
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_COMPARISON_JSON,
        help="Path to comparison JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=DEFAULT_COMPARISON_MD,
        help="Path to comparison Markdown report",
    )

    args = parser.parse_args()

    for p in [args.train_path, args.val_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            print("Please run scripts/create_splits.py first.", file=sys.stderr)
            sys.exit(1)

    table = train_and_evaluate_pytorch(
        train_path=args.train_path,
        val_path=args.val_path,
        models_dir=args.models_dir,
        comparison_report_path=args.output_json,
        comparison_md_path=args.output_md,
    )

    print("\n" + "=" * 80)
    print("                     MODEL VALIDATION COMPARISON                     ")
    print("=" * 80)
    print(f"{'Model':<24} | {'Log Loss':<9} | {'ROC AUC':<9} | {'PR AUC':<9} | {'Brier':<9} | {'Train Time'}")
    print("-" * 80)
    for model_name, metrics in table["models"].items():
        print(
            f"{model_name:<24} | "
            f"{metrics['log_loss']:<9.4f} | "
            f"{metrics['roc_auc']:<9.4f} | "
            f"{metrics['pr_auc']:<9.4f} | "
            f"{metrics['brier_score']:<9.4f} | "
            f"{metrics['train_time_seconds']}s"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
