"""Pydantic request and response schemas for AdLift FastAPI service."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ImpressionFeatureRecord(BaseModel):
    """Features required to score an individual ad impression at auction time."""
    model_config = ConfigDict(extra="forbid")

    impression_id: Optional[str] = Field(default=None, description="Optional unique identifier for tracking")
    cpo: float = Field(..., description="Cost-per-order / target bid constraint set by advertiser", ge=0.0)
    time_since_last_click: int = Field(..., description="Seconds since prior click (-1 if no prior click)")
    campaign: int = Field(..., description="Hashed campaign identifier")
    cat1: int = Field(..., description="Hashed contextual feature 1")
    cat2: int = Field(..., description="Hashed contextual feature 2")
    cat3: int = Field(..., description="Hashed contextual feature 3")
    cat4: int = Field(..., description="Hashed contextual feature 4")
    cat5: int = Field(..., description="Hashed contextual feature 5")
    cat6: int = Field(..., description="Hashed contextual feature 6")
    cat7: int = Field(..., description="Hashed contextual feature 7")
    cat8: int = Field(..., description="Hashed contextual feature 8")
    cat9: int = Field(..., description="Hashed contextual feature 9")


class PredictRequest(BaseModel):
    """Batch prediction request for one or more impressions."""
    impressions: List[ImpressionFeatureRecord] = Field(..., min_length=1, description="List of impression records to score")


class PredictionItem(BaseModel):
    """Scored prediction for an impression."""
    impression_id: str = Field(..., description="Impression identifier")
    predicted_click_probability: float = Field(..., description="Predicted click probability in [0, 1]")
    predicted_click_binary: int = Field(..., description="Predicted binary click outcome (threshold 0.5)")


class PredictResponse(BaseModel):
    """Response containing batch prediction results."""
    model_name: str
    predictions_count: int
    predictions: List[PredictionItem]


class CandidateRecord(BaseModel):
    """Candidate ad impression to be evaluated and ranked."""
    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(..., min_length=1, description="Unique candidate creative identifier")
    creative_theme: Optional[str] = Field(default="Standard Ad Creative", description="Human-readable theme")
    is_synthetic: bool = Field(default=True, description="Flag indicating synthetic demonstration record")
    cpo: float = Field(..., description="Cost-per-order target", ge=0.0)
    time_since_last_click: int = Field(..., description="Seconds since prior click (-1 if none)")
    campaign: int = Field(..., description="Hashed campaign identifier")
    cat1: int
    cat2: int
    cat3: int
    cat4: int
    cat5: int
    cat6: int
    cat7: int
    cat8: int
    cat9: int


class RankRequest(BaseModel):
    """Request to score and rank a set of candidate ads."""
    candidates: List[CandidateRecord] = Field(..., min_length=1, description="List of candidate ads to rank")


class RankedCandidateItem(BaseModel):
    """Ranked candidate ad output."""
    rank: int = Field(..., description="1-indexed rank position (1 is highest pCTR)")
    candidate_id: str
    creative_theme: str
    is_synthetic: bool
    predicted_click_probability: float


class RankResponse(BaseModel):
    """Response containing deterministically sorted candidate ad rankings and disclaimer."""
    disclaimer: str
    model_name: str
    is_synthetic_demonstration: bool
    candidates_count: int
    ranked_candidates: List[RankedCandidateItem]


class HealthResponse(BaseModel):
    """Service health and model status response."""
    status: str
    model_loaded: bool
    model_name: str
    version: str
