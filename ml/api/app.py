"""Minimal FastAPI service for ML fraud prediction.

Implements the ML/Fraud Intelligence Service HTTP interface defined in
``docs/ml-architecture.md`` (L42–L117).

Endpoints:

  ``POST /predict``  — Score a single raw transaction for fraud.
  ``GET  /health``   — Service health and model availability.

The model is loaded **once** at application startup via a lifespan
handler and reused for all requests — no retraining or refitting
per request.

The ``POST /predict`` endpoint accepts raw transaction data (as sent
by the backend ``TransactionCreate`` payload), computes the 24
engineered features internally via
:func:`ml.features.engineer.engineer_features_for_inference`, then
runs prediction and SHAP explanation.

Run locally::

    uvicorn ml.api.app:app --host 0.0.0.0 --port 8001

Environment:
  ``ML_MODEL_PATH``  — Override the default artifact path.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from ml.features.engineer import engineer_features_for_inference
from ml.features.history import history_store, record_transaction
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

_FORBIDDEN_KEYS = frozenset({"isFraud", "TransactionID"})


class CustomerHistory(BaseModel):
    """Optional customer history for richer feature engineering.

    When provided, historical features can use real values instead
    of cold-start defaults.  All fields are optional.
    """

    transaction_count_30d: int | None = Field(
        None, ge=0, description="Transactions in last 30 days"
    )
    avg_amount_30d: float | None = Field(
        None, ge=0, description="Average amount in last 30 days"
    )
    std_amount_30d: float | None = Field(
        None, ge=0, description="Std deviation of amounts in last 30 days"
    )
    last_transaction_country: str | None = Field(
        None, description="Country of last transaction"
    )
    known_device_fingerprints: list[str] | None = Field(
        None, description="Previously seen device fingerprints"
    )
    known_merchant_ids: list[str] | None = Field(
        None, description="Previously seen merchant identifiers"
    )
    previous_flagged_count: int | None = Field(
        None, ge=0, description="Previously flagged transactions"
    )


class RawTransactionInput(BaseModel):
    """Raw transaction payload — matches backend ``TransactionCreate``.

    Required fields align with the backend schema so the backend can
    send ``request.model_dump()`` directly.  Additional optional
    fields allow richer feature engineering when available.
    """

    # -- required (from backend TransactionCreate) -----------------------
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3)
    merchant_name: str = Field(..., min_length=1, max_length=255)
    merchant_category: str = Field(..., max_length=10)
    transaction_type: str = Field(
        ..., pattern=r"^(purchase|transfer|withdrawal)$"
    )
    location_country: str = Field(..., min_length=1, max_length=100)
    location_city: str = Field(..., max_length=100)
    device_fingerprint: str = Field(..., min_length=1, max_length=255)
    device_type: str = Field(..., pattern=r"^(mobile|desktop|pos)$")
    ip_address: str = Field(..., min_length=7, max_length=45)

    # -- optional: customer identification ---------------------------------
    customer_id: str | None = Field(
        None, min_length=1, max_length=255,
        description="Customer identifier (falls back to device_fingerprint)",
    )

    # -- optional: raw dataset field mappings ----------------------------
    timestamp: int | None = Field(
        None, ge=0, description="TransactionDT (seconds from reference)"
    )
    card1: int | None = Field(
        None, description="Card identifier for grouping"
    )
    addr1: int | None = Field(
        None, description="Region code (integer)"
    )
    addr2: int | None = Field(
        None, description="Country code (integer)"
    )
    ProductCD: str | None = Field(
        None, max_length=1, description="Product code (W/X/Y/Z/S)"
    )
    id_19: str | None = Field(None, description="Identity field id_19")
    id_20: str | None = Field(None, description="Identity field id_20")
    DeviceType: str | None = Field(
        None, description="Device type from identity table"
    )
    has_identity_data: int | None = Field(
        None, ge=0, le=1, description="Whether identity data exists"
    )

    # -- optional: customer history --------------------------------------
    customer_history: CustomerHistory | None = None

    # -- leakage protection ----------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def reject_forbidden_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            found = _FORBIDDEN_KEYS & set(values.keys())
            if found:
                raise ValueError(
                    f"Forbidden fields in request: {sorted(found)}. "
                    f"isFraud and TransactionID are not allowed."
                )
        return values


class FactorOutput(BaseModel):
    """Single SHAP feature attribution."""

    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="SHAP contribution value")


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    fraud_probability: float = Field(
        ..., description="Fraud probability in [0, 1]"
    )
    fraud_prediction: int = Field(
        ..., description="Binary prediction (0=legit, 1=fraud)"
    )
    threshold: float = Field(..., description="Decision threshold used")
    model_version: str = Field(..., description="Model version string")
    explanation: list[FactorOutput] | None = Field(
        None,
        description="Top SHAP feature attributions (sorted by |importance| desc)",
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="'ready' or 'model_unavailable'")
    model_version: str | None = Field(None, description="Loaded model version")
    features: int | None = Field(None, description="Number of model features")


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawTransactionInput) -> PredictionResponse:
    """Score a single raw transaction for fraud.

    Accepts raw transaction data, computes 24 engineered features
    internally, runs the XGBoost model, and returns the prediction
    with SHAP explanations.

    Returns 503 if the model is unavailable.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Run `python -m ml.predict.save_model` first.",
        )

    # Convert raw transaction to 24 engineered features (with history)
    raw_data = request.model_dump()
    try:
        features_df = engineer_features_for_inference(
            raw_data, history_store=history_store
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering failed: {exc}",
        )

    # Run prediction + SHAP explanation
    try:
        result: PredictionResult = _predictor.predict(features_df, explain=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Build explanation list for response
    factors = None
    if result.explanation is not None:
        factors = [
            FactorOutput(feature=f["feature"], importance=f["importance"])
            for f in result.explanation
        ]

    # Record transaction in history store for future lookups.
    # Best-effort: a recording failure must never block prediction.
    try:
        record_transaction(history_store, raw_data)
    except Exception:
        pass

    return PredictionResponse(
        fraud_probability=result.fraud_probability,
        fraud_prediction=result.fraud_prediction,
        threshold=result.threshold,
        model_version=result.model_version,
        explanation=factors,
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
