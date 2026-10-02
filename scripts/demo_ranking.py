"""CLI demonstration of candidate ranking using synthetic ad impression records."""

import argparse
import json
import os
import sys

from adlift.ranking import OFFLINE_RANKING_DISCLAIMER, rank_candidates

DEFAULT_CANDIDATES_PATH = os.path.join("data", "demo", "synthetic_candidates.json")
DEFAULT_MODEL_PATH = os.path.join("models", "lightgbm_model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidate ad impressions using offline click prediction model."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=DEFAULT_CANDIDATES_PATH,
        help="Path to synthetic candidates JSON file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model artifact",
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found at: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at: {args.model_path}", file=sys.stderr)
        print("Please train the model first using scripts/train_tabular.py", file=sys.stderr)
        sys.exit(1)

    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates_data = json.load(f)

    result = rank_candidates(candidates_data, model_path=args.model_path)

    print("\n" + "=" * 80)
    print("                      ADLIFT OFFLINE RANKING DEMO                     ")
    print("=" * 80)
    print(f"NOTE: {result['disclaimer']}")
    print("-" * 80)
    print(f"Model: {result['model_used']} | Scored Candidates: {result['candidates_count']}")
    print("-" * 80)
    print(f"{'Rank':<6} | {'Candidate ID':<20} | {'p(Click)':<10} | {'Creative Theme':<38}")
    print("-" * 80)
    for c in result["ranked_candidates"]:
        p_pct = f"{c['predicted_click_probability'] * 100:.2f}%"
        print(f"#{c['rank']:<5} | {c['candidate_id']:<20} | {p_pct:<10} | {c['creative_theme'][:38]:<38}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
