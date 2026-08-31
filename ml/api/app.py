"""Minimal FastAPI service for ML fraud prediction.

Implements the ML/Fraud Intelligence Service HTTP interface defined in
``docs/ml-architecture.md`` (L42–L117).

Endpoints:

  ``POST /predict``  — Score a single transaction for fraud.
  ``GET  /health``   — Service health and model availability.

The model is loaded **once** at application startup via a lifespan
handler and reused for all requests — no retraining or refitting
per request.

Run locally::

    uvicorn ml.api.app:app --host 0.0.0.0 --port 8001

Environment:
  ``ML_MODEL_PATH``  — Override the default artifact path.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ml.features.engineer import FEATURE_LIST
from ml.predict.bundle import model_exists
from ml.predict.predictor import FraudPredictor, PredictionResult

# ── Lifespan: load model once at startup ──────────────────────────────

_predictor: FraudPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; release on shutdown."""
    global _predictor
    model_path = os.environ.get("ML_MODEL_PATH")
    try:
        _predictor = FraudPredictor(bundle_path=model_path)
        print(f"[ml-api] Model loaded: {_predictor.model_version}")
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ml-api] Model not available: {exc}")
        _predictor = None
    yield
    _predictor = None


app = FastAPI(
    title="ML / Fraud Intelligence Service",
    description="Internal prediction service for fraud detection.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────


class FeatureInput(BaseModel):
    """Engineered feature vector for a single transaction.

    All 24 features produced by ``ml.features.engineer`` must be
    provided.  Extra fields are rejected.
    """

    amount: float = Field(..., description="Transaction amount")
    amount_deviation: float = Field(..., description="Z-score deviation from customer avg")
    amount_to_avg_ratio: float = Field(..., description="Ratio to customer average")
    location_country: float = Field(..., description="Encoded country")
    location_region: float = Field(..., description="Encoded region")
    location_is_new: int = Field(..., ge=0, le=1, description="New location flag")
    location_change: int = Field(..., ge=0, le=1, description="Location change flag")
    device_fingerprint: str = Field(..., description="Device identifier or 'no_device_data'")
    is_new_device: int = Field(..., ge=0, le=1, description="New device flag")
    hour_of_day_raw: int = Field(..., ge=0, le=23, description="Hour of day (integer)")
    hour_of_day_sin: float = Field(..., description="Hour sin encoding")
    hour_of_day_cos: float = Field(..., description="Hour cos encoding")
    day_of_week_raw: int = Field(..., ge=0, le=6, description="Day of week (integer)")
    day_of_week_sin: float = Field(..., description="Day sin encoding")
    day_of_week_cos: float = Field(..., description="Day cos encoding")
    is_unusual_hour: int = Field(..., ge=0, le=1, description="Unusual hour flag")
    tx_velocity_1h: int = Field(..., ge=0, description="Transactions in last 1h")
    tx_velocity_24h: int = Field(..., ge=0, description="Transactions in last 24h")
    tx_velocity_7d: int = Field(..., ge=0, description="Transactions in last 7d")
    merchant_category: int = Field(..., description="Encoded merchant category")
    merchant_is_new: int = Field(..., ge=0, le=1, description="New merchant flag")
    avg_spend_30d: float = Field(..., ge=0, description="30-day avg spend")
    previous_suspicious_count: int = Field(..., ge=0, description="Prior flagged count")
    has_identity_data: int = Field(..., ge=0, le=1, description="Identity data flag")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return v


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    fraud_probability: float = Field(..., description="Fraud probability ∈ [0, 1]")
    fraud_prediction: int = Field(..., description="Binary prediction (0=legit, 1=fraud)")
    threshold: float = Field(..., description="Decision threshold used")
    model_version: str = Field(..., description="Model version string")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="'ready' or 'model_unavailable'")
    model_version: str | None = Field(None, description="Loaded model version")
    features: int | None = Field(None, description="Number of model features")


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
def predict(request: FeatureInput) -> PredictionResponse:
    """Score a single transaction for fraud.

    Requires a loaded model.  Returns 503 if the model is unavailable.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Run `python -m ml.predict.save_model` first.",
        )

    # Convert Pydantic model to DataFrame with correct column order
    data = request.model_dump()
    df = pd.DataFrame([data])

    try:
        result: PredictionResult = _predictor.predict(df)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return PredictionResponse(
        fraud_probability=result.fraud_probability,
        fraud_prediction=result.fraud_prediction,
        threshold=result.threshold,
        model_version=result.model_version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check — reports model availability."""
    if _predictor is None:
        return HealthResponse(status="model_unavailable")
    return HealthResponse(
        status="ready",
        model_version=_predictor.model_version,
        features=len(_predictor.feature_names),
    )
