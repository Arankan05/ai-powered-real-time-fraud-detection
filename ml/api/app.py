"""Minimal FastAPI service for ML fraud prediction.

Implements the ML/Fraud Intelligence Service HTTP interface defined in
``docs/ml-architecture.md`` (L42–L117).

Endpoints:

  ``POST /predict``  — Score a single raw transaction for fraud.
  ``GET  /health``   — Service health and model availability (legacy).
  ``GET  /live``     — Liveness probe (process alive).
  ``GET  /ready``    — Readiness probe (model loaded, service operational).

The model is loaded **once** at application startup via a lifespan
handler and reused for all requests — no retraining or refitting
per request.

Run locally::

    uvicorn ml.api.app:app --host 0.0.0.0 --port 8001

Environment:
  ``ML_MODEL_PATH``        — Override the default artifact path.
  ``ML_HISTORY_DB_PATH``   — SQLite history store path.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from ml.features.engineer import engineer_features_for_inference
from ml.features.history import (
    InMemoryHistoryStore,
    SQLiteHistoryRepository,
    record_transaction,
)
import ml.features.history as _history_module
from ml.predict.bundle import ModelLoadError, model_exists
from ml.predict.predictor import FraudPredictor, PredictionResult
from ml.risk.aggregator import aggregate_risk
from ml.rules.engine import evaluate_rules

logger = logging.getLogger(__name__)

# ── Lifespan: load model once at startup ──────────────────────────────

_predictor: FraudPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialise persistent history store; release on shutdown."""
    global _predictor

    # ── Model ─────────────────────────────────────────────────────
    model_path = os.environ.get("ML_MODEL_PATH")
    try:
        _predictor = FraudPredictor(bundle_path=model_path)
        logger.info("Model loaded: version=%s", _predictor.model_version)
    except FileNotFoundError as exc:
        logger.warning("Model not available: %s", exc)
        _predictor = None
    except (ModelLoadError, KeyError) as exc:
        logger.warning("Model load failed: %s", exc)
        _predictor = None
    except Exception as exc:
        # Catch-all: any unexpected loading failure should not crash startup.
        logger.warning("Unexpected model load failure (%s); service starts without model", type(exc).__name__)
        _predictor = None

    # ── History store ─────────────────────────────────────────────
    # Try SQLite (persistent); fall back to in-memory on failure.
    db_path = os.environ.get("ML_HISTORY_DB_PATH", "data/ml_history.db")
    sqlite_store: SQLiteHistoryRepository | None = None
    try:
        sqlite_store = SQLiteHistoryRepository(db_path=db_path)
        _history_module.history_store = sqlite_store
        logger.info("History store: SQLite")
    except Exception as exc:
        logger.warning("SQLite history unavailable (%s); using in-memory store", exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    _predictor = None
    if sqlite_store is not None:
        try:
            sqlite_store.close()
        except Exception:
            pass


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

    Step 42: hardened validation — upper bounds, format checks,
    and sanitised error messages.
    """

    # -- required (from backend TransactionCreate) -----------------------
    amount: float = Field(
        ..., gt=0, le=10_000_000,
        description="Transaction amount (0, 10M]",
    )
    currency: str = Field(
        ..., pattern=r"^[A-Z]{3}$",
        description="ISO 4217 currency code (3 uppercase letters)",
    )
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
        description="Server-authoritative customer identifier (Step 41)",
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
        None, pattern=r"^[WXYZS]$", description="Product code (W/X/Y/Z/S)"
    )
    id_19: str | None = Field(None, max_length=100, description="Identity field id_19")
    id_20: str | None = Field(None, max_length=100, description="Identity field id_20")
    DeviceType: str | None = Field(
        None, max_length=50, description="Device type from identity table"
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


class RuleTriggerOutput(BaseModel):
    """Single triggered risk rule."""

    rule: str = Field(..., description="Rule identifier")
    contribution: int = Field(..., description="Score contribution")
    reason: str = Field(..., description="Human-readable explanation")


class BehaviourSignalOutput(BaseModel):
    """Single behavioural anomaly signal."""

    signal: str = Field(..., description="Signal identifier")
    severity: float = Field(..., description="Severity in [0, 1]")
    reason: str = Field(..., description="Human-readable explanation")


class ExplanationOutput(BaseModel):
    """Composite explanation from ML, behaviour, and rule components."""

    ml_top_factors: list[FactorOutput] = Field(
        default_factory=list, description="Top SHAP feature attributions"
    )
    behaviour_signals: list[BehaviourSignalOutput] = Field(
        default_factory=list, description="Behavioural anomaly signals"
    )
    rules_triggered: list[RuleTriggerOutput] = Field(
        default_factory=list, description="Triggered risk rules"
    )


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
    # Legacy explanation field (backward-compatible)
    explanation: list[FactorOutput] | None = Field(
        None,
        description="Top SHAP feature attributions (sorted by |importance| desc)",
    )
    timestamp: int | None = Field(
        None,
        description="Transaction timestamp used (for outcome feedback reference)",
    )
    # Risk scores (architecture §5)
    ml_score: int | None = Field(
        None, description="ML probability scaled to [0, 100]"
    )
    behaviour_score: int | None = Field(
        None, description="Behavioural anomaly score [0, 100]"
    )
    rule_score: int | None = Field(
        None, description="Rule-based risk score [0, 100]"
    )
    risk_score: int | None = Field(
        None, description="Aggregated risk score [0, 100]"
    )
    risk_level: str | None = Field(
        None, description="Risk level: LOW, MEDIUM, or HIGH"
    )
    decision: str | None = Field(
        None, description="Decision: APPROVE, VERIFY, or HOLD"
    )
    # Structured explanation (architecture §6)
    explanation_detail: ExplanationOutput | None = Field(
        None, description="Full structured explanation with behaviour and rule signals"
    )
    risk_factors: list[str] | None = Field(
        None, description="Combined list of all risk factor identifiers"
    )


class OutcomeUpdateRequest(BaseModel):
    """Request to update the fraud outcome of a previously recorded transaction."""

    customer_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Customer identifier used when the transaction was recorded",
    )
    timestamp: int = Field(
        ..., ge=0,
        description="Transaction timestamp (as returned in the prediction response)",
    )
    is_fraud: int = Field(
        ..., ge=0, le=1,
        description="Confirmed fraud label (0=legitimate, 1=fraudulent)",
    )


class OutcomeUpdateResponse(BaseModel):
    """Response from the /outcome endpoint."""

    updated: bool = Field(..., description="Whether the record was found and updated")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: int = Field(..., description="Transaction timestamp that was updated")
    is_fraud: int = Field(..., description="New fraud label value")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="'ready' or 'model_unavailable'")
    model_version: str | None = Field(None, description="Loaded model version")
    features: int | None = Field(None, description="Number of model features")
    history_store: str | None = Field(
        None, description="History store type ('sqlite' or 'in_memory')"
    )


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawTransactionInput) -> PredictionResponse:
    """Score a single raw transaction for fraud.

    Accepts raw transaction data, computes 24 engineered features
    internally, evaluates rule-based risk signals, runs the XGBoost
    model, and returns the prediction with SHAP explanations and
    risk scores.

    Returns 503 if the model is unavailable.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available. Contact service administrator.",
        )

    # Convert raw transaction to 24 engineered features (with history)
    raw_data = request.model_dump()

    # Auto-generate timestamp when backend does not provide one so that
    # historical velocity features can accumulate across requests.
    if not raw_data.get("timestamp"):
        raw_data["timestamp"] = int(time.time())

    _store = _history_module.history_store

    # Retrieve history for both feature engineering and rule evaluation.
    # The history lookup uses the same temporal safety as the feature pipeline.
    history_records: list[dict] = []
    if _store is not None:
        from ml.features.engineer import _resolve_customer_id
        cid = _resolve_customer_id(raw_data)
        ts = int(raw_data.get("timestamp", 0))
        try:
            history_records = _store.get(cid, before_timestamp=ts)
        except Exception:
            history_records = []

    try:
        features_df = engineer_features_for_inference(
            raw_data, history_store=_store
        )
    except Exception as exc:
        logger.warning("Feature engineering failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Transaction data could not be processed for feature extraction.",
        )

    # Run prediction + SHAP explanation
    try:
        result: PredictionResult = _predictor.predict(features_df, explain=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction processing failed.",
        )

    # Validate prediction output is sane
    if not (0.0 <= result.fraud_probability <= 1.0):
        logger.error("Model returned out-of-range probability: %s", result.fraud_probability)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model produced an invalid prediction.",
        )

    # Evaluate rule-based risk signals and behavioural anomalies
    try:
        rule_result = evaluate_rules(features_df, raw_data, history_records)
    except Exception:
        # Rule evaluation must never block prediction — fall back to empty.
        logger.warning("Rule evaluation failed; falling back to empty result", exc_info=True)
        from ml.rules.engine import RuleResult
        rule_result = RuleResult(
            rule_score=0, behaviour_score=0,
            rules_triggered=[], behaviour_signals=[],
        )

    # Build SHAP explanation list (legacy field)
    factors = None
    if result.explanation is not None:
        factors = [
            FactorOutput(feature=f["feature"], importance=f["importance"])
            for f in result.explanation
        ]

    # Compute risk scores via the dedicated aggregator (architecture §5)
    assessment = aggregate_risk(
        fraud_probability=result.fraud_probability,
        behaviour_score=rule_result.behaviour_score,
        rule_score=rule_result.rule_score,
    )

    # Build structured explanation (architecture §6)
    explanation_detail = ExplanationOutput(
        ml_top_factors=factors or [],
        behaviour_signals=[
            BehaviourSignalOutput(
                signal=s.signal, severity=s.severity, reason=s.reason
            )
            for s in rule_result.behaviour_signals
        ],
        rules_triggered=[
            RuleTriggerOutput(
                rule=r.rule, contribution=r.contribution, reason=r.reason
            )
            for r in rule_result.rules_triggered
        ],
    )

    # Build combined risk_factors list
    risk_factors: list[str] = []
    if factors:
        risk_factors.extend(f.feature for f in factors[:5])
    risk_factors.extend(s.signal for s in rule_result.behaviour_signals)
    risk_factors.extend(r.rule for r in rule_result.rules_triggered)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_factors: list[str] = []
    for f in risk_factors:
        if f not in seen:
            seen.add(f)
            unique_factors.append(f)

    # Record transaction in history store for future lookups.
    # Best-effort: a recording failure must never block prediction.
    try:
        record_transaction(_store, raw_data)
    except Exception:
        logger.warning("Failed to record transaction in history", exc_info=True)

    logger.info(
        "Prediction complete: model=%s risk=%s decision=%s prob=%.4f",
        result.model_version, assessment.risk_level, assessment.decision,
        result.fraud_probability,
    )

    return PredictionResponse(
        fraud_probability=result.fraud_probability,
        fraud_prediction=result.fraud_prediction,
        threshold=result.threshold,
        model_version=result.model_version,
        explanation=factors,
        timestamp=raw_data.get("timestamp"),
        ml_score=assessment.ml_score,
        behaviour_score=assessment.behaviour_score,
        rule_score=assessment.rule_score,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        decision=assessment.decision,
        explanation_detail=explanation_detail,
        risk_factors=unique_factors if unique_factors else None,
    )


@app.post("/outcome", response_model=OutcomeUpdateResponse)
def update_outcome(request: OutcomeUpdateRequest) -> OutcomeUpdateResponse:
    """Update the fraud outcome of a previously recorded transaction.

    Used for the label feedback loop: after a transaction is later
    confirmed as fraudulent or legitimate, this endpoint updates the
    stored history record so future predictions use the correct label.

    Returns 404 if the target transaction cannot be found.
    """
    _store = _history_module.history_store
    try:
        updated = _store.record_outcome(
            customer_id=request.customer_id,
            timestamp=request.timestamp,
            is_fraud=request.is_fraud,
        )
    except Exception:
        logger.error("Failed to update outcome", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update outcome.",
        )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction record not found for the given customer_id and timestamp.",
        )

    logger.info("Outcome updated: is_fraud=%s", request.is_fraud)

    return OutcomeUpdateResponse(
        updated=True,
        customer_id=request.customer_id,
        timestamp=request.timestamp,
        is_fraud=request.is_fraud,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check — reports model and history store status.

    Legacy endpoint.  Prefer /live (liveness) and /ready (readiness)
    for production orchestration.
    """
    store = _history_module.history_store
    store_type = _store_type_label(store)
    if _predictor is None:
        return HealthResponse(status="model_unavailable", history_store=store_type)
    return HealthResponse(
        status="ready",
        model_version=_predictor.model_version,
        features=len(_predictor.feature_names),
        history_store=store_type,
    )


class LivenessResponse(BaseModel):
    """Liveness probe — process is alive."""
    status: str = Field(..., description="Always 'alive' if the process responds")


class ReadinessResponse(BaseModel):
    """Readiness probe — service can perform predictions."""
    status: str = Field(..., description="'ready' or 'not_ready'")
    model_version: str | None = Field(None, description="Loaded model version")
    features: int | None = Field(None, description="Number of model features")
    history_store: str | None = Field(
        None, description="History store type ('sqlite' or 'in_memory')"
    )


@app.get("/live", response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    """Liveness probe — always returns 200 if the process is running."""
    return LivenessResponse(status="alive")


@app.get("/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    """Readiness probe — reports whether the model is loaded and service can predict."""
    store = _history_module.history_store
    store_type = _store_type_label(store)
    if _predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadinessResponse(
                status="not_ready", history_store=store_type
            ).model_dump(),
        )
    return ReadinessResponse(
        status="ready",
        model_version=_predictor.model_version,
        features=len(_predictor.feature_names),
        history_store=store_type,
    )


def _store_type_label(store) -> str:
    """Return a safe label for the history store type."""
    if isinstance(store, SQLiteHistoryRepository):
        return "sqlite"
    return "in_memory"


# ── Global error handlers (Step 42: prevent information leakage) ─────


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: never expose internal exceptions to API clients."""
    logger.error("Unhandled exception: %s", type(exc).__name__, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
