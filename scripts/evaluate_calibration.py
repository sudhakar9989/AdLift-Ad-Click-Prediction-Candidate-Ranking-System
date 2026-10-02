"""CLI script to evaluate model calibration, score buckets, and performance slices."""

import argparse
import os
import sys

from adlift.analysis import run_full_calibration_analysis

DEFAULT_VAL_PATH = os.path.join("data", "processed", "validation.csv")
DEFAULT_MODEL_PATH = os.path.join("models", "lightgbm_model.joblib")
DEFAULT_FIGS_DIR = os.path.join("reports", "figures")
DEFAULT_REPORT_JSON = os.path.join("reports", "calibration_and_error_analysis.json")
DEFAULT_REPORT_MD = os.path.join("reports", "calibration_and_error_analysis.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM model calibration, score buckets, slices, and generate diagnostic plots."
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=DEFAULT_VAL_PATH,
        help="Path to validation.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained lightgbm_model.joblib",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=DEFAULT_FIGS_DIR,
        help="Directory to save diagnostic plots",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_REPORT_JSON,
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=DEFAULT_REPORT_MD,
        help="Path to output Markdown report",
    )

    args = parser.parse_args()

    for p in [args.val_path, args.model_path]:
        if not os.path.exists(p):
            print(f"ERROR: Missing file: {p}", file=sys.stderr)
            print("Please ensure train_tabular.py has been run.", file=sys.stderr)
            sys.exit(1)

    report = run_full_calibration_analysis(
        val_path=args.val_path,
        model_path=args.model_path,
        figures_dir=args.figures_dir,
        report_json_path=args.output_json,
        report_md_path=args.output_md,
    )

    print("\n" + "=" * 75)
    print("                    CALIBRATION & SLICE SUMMARY                     ")
    print("=" * 75)
    print(f"Evaluated Split: {report['split_evaluated']}")
    print(f"Empirical Click Rate         : {report['empirical_click_rate']*100:.2f}%")
    print(f"Mean Predicted Probability   : {report['mean_predicted_probability']*100:.2f}%")
    print(f"Expected Calibration Error   : {report['calibration_summary']['expected_calibration_error']:.4f}")
    print(f"Brier Score                  : {report['calibration_summary']['brier_score']:.4f}")
    print("-" * 75)
    print(f"{'Score Decile':<15} | {'Count':<8} | {'Observed CTR':<14} | {'Mean Predicted':<14} | {'Gap'}")
    print("-" * 75)
    for b in report["score_buckets"]:
        print(f"{b['range']:<15} | {b['count']:<8,} | {b['observed_click_rate']*100:<13.2f}% | {b['mean_predicted_prob']*100:<13.2f}% | {b['calibration_gap']:+.4f}")
    print("-" * 75)
    print("Figures Generated:")
    for fig in report["figures_generated"]:
        print(f"  • {fig}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
