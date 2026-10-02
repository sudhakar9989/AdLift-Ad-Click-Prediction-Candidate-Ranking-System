"""Candidate ad ranking function, tie-breaking, and synthetic illustration utilities for AdLift."""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from adlift.features import APPROVED_INPUT_FEATURES

DEFAULT_MODEL_PATH = os.path.join("models", "lightgbm_model.joblib")

OFFLINE_RANKING_DISCLAIMER: str = (
    "DEMO NOTICE: Ranking these supplied candidate records illustrates how an offline "
    "model sorts potential ad impressions by predicted click probability. However, "
    "observational impression data DOES NOT prove that showing the top-ranked ad would "
    "cause more clicks or lift revenue. The dataset contains observed impressions, not an "
    "online counterfactual randomized experiment (A/B test)."
)


def load_ranking_model(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """Load the saved click-prediction model artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model artifact not found at '{model_path}'. "
            f"Please run scripts/train_tabular.py to train the model first."
        )
    return joblib.load(model_path)


def validate_candidate_record(candidate: Dict[str, Any], idx: int = 0) -> None:
    """Validate that candidate contains all approved model input features and valid types."""
    if not isinstance(candidate, dict):
        raise ValueError(f"Candidate #{idx + 1} must be a dictionary, got {type(candidate).__name__}.")

    missing_fields = [f for f in APPROVED_INPUT_FEATURES if f not in candidate]
    if missing_fields:
        cand_id = candidate.get("candidate_id", f"index_{idx}")
        raise ValueError(
            f"Candidate '{cand_id}' (item #{idx + 1}) is missing required feature(s): {missing_fields}. "
            f"All {len(APPROVED_INPUT_FEATURES)} approved features must be provided: {APPROVED_INPUT_FEATURES}"
        )


def rank_candidates(
    candidates: List[Dict[str, Any]],
    model: Optional[Any] = None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Score a list of supplied candidate records and return them sorted by predicted click probability.

    Args:
        candidates: List of candidate impression dictionaries.
        model: Optional preloaded model instance. If None, loads from model_path.
        model_path: Path to saved model artifact.

    Returns:
        Dictionary containing the offline disclaimer, synthetic flag, and ranked candidate list.
    """
    if not candidates or len(candidates) == 0:
        raise ValueError("Candidates list cannot be empty. Please provide at least one candidate record.")

    # 1. Validate every candidate input
    for idx, cand in enumerate(candidates):
        validate_candidate_record(cand, idx)

    # 2. Load model if not provided
    if model is None:
        model = load_ranking_model(model_path)

    # 3. Construct feature DataFrame
    feature_rows = []
    metadata_list = []
    for idx, cand in enumerate(candidates):
        cand_id = str(cand.get("candidate_id", f"candidate_{idx + 1}"))
        creative_theme = cand.get("creative_theme", "Standard Ad Creative")
        is_synthetic = cand.get("is_synthetic", True)

        row_features = {f: cand[f] for f in APPROVED_INPUT_FEATURES}
        feature_rows.append(row_features)
        metadata_list.append({
            "candidate_id": cand_id,
            "creative_theme": creative_theme,
            "is_synthetic": is_synthetic,
            "features": row_features,
        })

    features_df = pd.DataFrame(feature_rows)

    # 4. Predict click probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_df)[:, 1]
    else:
        raise AttributeError("Model must implement 'predict_proba' method.")

    # 5. Attach predicted score to metadata
    scored_items = []
    for meta, prob in zip(metadata_list, probs):
        scored_items.append({
            "candidate_id": meta["candidate_id"],
            "creative_theme": meta["creative_theme"],
            "is_synthetic": meta["is_synthetic"],
            "predicted_click_probability": round(float(prob), 6),
            "features": meta["features"],
        })

    # 6. Deterministic sorting:
    # Primary key: probability DESCENDING (-predicted_click_probability)
    # Secondary key: candidate_id ASCENDING (deterministic tie-breaking)
    sorted_items = sorted(
        scored_items,
        key=lambda item: (-item["predicted_click_probability"], item["candidate_id"]),
    )

    # 7. Assign 1-indexed ranks
    ranked_candidates = []
    for rank_idx, item in enumerate(sorted_items, start=1):
        ranked_candidates.append({
            "rank": rank_idx,
            "candidate_id": item["candidate_id"],
            "creative_theme": item["creative_theme"],
            "is_synthetic": item["is_synthetic"],
            "predicted_click_probability": item["predicted_click_probability"],
            "features": item["features"],
        })

    return {
        "disclaimer": OFFLINE_RANKING_DISCLAIMER,
        "is_synthetic_demonstration": True,
        "model_used": type(model).__name__,
        "candidates_count": len(ranked_candidates),
        "ranked_candidates": ranked_candidates,
    }
