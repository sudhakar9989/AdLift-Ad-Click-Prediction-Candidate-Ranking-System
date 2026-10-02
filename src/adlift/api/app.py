"""FastAPI application for AdLift click prediction and candidate ranking."""

from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Any, Optional

import fastapi
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd

from adlift import __version__
from adlift.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    RankRequest,
    RankResponse,
    RankedCandidateItem,
)
from adlift.features import APPROVED_INPUT_FEATURES
from adlift.ranking import OFFLINE_RANKING_DISCLAIMER, load_ranking_model, rank_candidates

# Configure privacy-safe logger (logs request counts and latencies only, never raw row payloads)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("adlift.api")

DEFAULT_MODEL_PATH = os.path.join("models", "lightgbm_model.joblib")


def create_app(model_path: str = DEFAULT_MODEL_PATH) -> FastAPI:
    """Application factory for AdLift FastAPI service with lifespan model loading."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"Loading click prediction model from '{model_path}'...")
        try:
            model = load_ranking_model(model_path)
            app.state.model = model
            app.state.model_name = type(model).__name__
            logger.info(f"Successfully loaded model: {app.state.model_name}")
        except Exception as e:
            logger.warning(f"Could not load primary model from {model_path}: {e}")
            app.state.model = None
            app.state.model_name = "None"
        yield
        logger.info("Shutting down AdLift API service...")

    app = FastAPI(
        title="AdLift Click Prediction & Ranking API",
        description=(
            "Production-ready FastAPI service predicting ad impression click probabilities "
            "and ranking candidate ads using leak-free, calibrated models."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # Privacy-conscious request timing middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        # Log metadata only - NEVER log raw user IDs or request bodies
        logger.info(
            f"{request.method} {request.url.path} -> status: {response.status_code} ({duration_ms:.2f}ms)"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation failure on {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Validation Error",
                "details": exc.errors(),
                "message": "One or more required impression features are missing or invalid.",
                "required_features": APPROVED_INPUT_FEATURES,
            },
        )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def get_health() -> HealthResponse:
        """Service health probe and model loading verification."""
        is_loaded = getattr(app.state, "model", None) is not None
        model_name = getattr(app.state, "model_name", "None")
        return HealthResponse(
            status="healthy" if is_loaded else "degraded",
            model_loaded=is_loaded,
            model_name=model_name,
            version=__version__,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Scoring"])
    def predict_clicks(request_data: PredictRequest) -> PredictResponse:
        """Score one or more impression records and return predicted click probabilities."""
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Click prediction model is not loaded.",
            )

        impressions = request_data.impressions
        feature_rows = []
        ids = []

        for idx, imp in enumerate(impressions):
            imp_dict = imp.model_dump()
            ids.append(imp.impression_id or f"imp_{idx + 1}")
            feature_rows.append({f: imp_dict[f] for f in APPROVED_INPUT_FEATURES})

        features_df = pd.DataFrame(feature_rows)
        probs = model.predict_proba(features_df)[:, 1]

        predictions = []
        for imp_id, prob in zip(ids, probs):
            p_val = round(float(prob), 6)
            predictions.append(
                PredictionItem(
                    impression_id=imp_id,
                    predicted_click_probability=p_val,
                    predicted_click_binary=1 if p_val >= 0.5 else 0,
                )
            )

        return PredictResponse(
            model_name=app.state.model_name,
            predictions_count=len(predictions),
            predictions=predictions,
        )

    @app.post("/rank", response_model=RankResponse, tags=["Ranking"])
    def rank_ad_candidates(request_data: RankRequest) -> RankResponse:
        """Score and sort candidate ad impressions by predicted click probability."""
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Click prediction model is not loaded.",
            )

        candidates_dicts = [c.model_dump() for c in request_data.candidates]
        result = rank_candidates(candidates_dicts, model=model)

        ranked_items = [
            RankedCandidateItem(
                rank=item["rank"],
                candidate_id=item["candidate_id"],
                creative_theme=item["creative_theme"],
                is_synthetic=item["is_synthetic"],
                predicted_click_probability=item["predicted_click_probability"],
            )
            for item in result["ranked_candidates"]
        ]

        return RankResponse(
            disclaimer=result["disclaimer"],
            model_name=app.state.model_name,
            is_synthetic_demonstration=True,
            candidates_count=len(ranked_items),
            ranked_candidates=ranked_items,
        )

    return app


# Default ASGI application instance
app = create_app()
