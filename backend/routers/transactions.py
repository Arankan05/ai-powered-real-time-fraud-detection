"""Transaction router — ``POST /api/v1/transactions``.

Accepts a raw transaction, calls the ML / Fraud Intelligence Service
for fraud scoring, and returns the enriched response including ML
predictions, SHAP explanations, and risk decisions.

This module wires the :class:`MLServiceClient` into the transaction
flow.  Authentication and database persistence are placeholder
stubs that will be completed by the backend developer (Developer A).

Architecture reference: ``docs/api-contract.md`` L142–L234.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.schemas import (
    AlertSummary,
    MLExplanation,
    MLPredictionResponse,
    OutcomeResponse,
    OutcomeUpdate,
    TransactionCreate,
    TransactionResponse,
)
from backend.services.ml_client import (
    MLServiceClient,
    MLServiceResponseError,
    MLServiceTimeoutError,
    MLServiceUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transactions"])

# Module-level client — replaced at app startup via dependency injection
# or direct assignment.  Default: localhost:8001 with 5 s timeout.
_ml_client: MLServiceClient | None = None

# Alert repository — set at app startup
_alert_repo = None


def set_ml_client(client: MLServiceClient) -> None:
    """Set the ML service client (called during app startup)."""
    global _ml_client
    _ml_client = client


def set_alert_repository(repo: Any) -> None:
    """Set the alert repository (called during app startup)."""
    global _alert_repo
    _alert_repo = repo


def get_ml_client() -> MLServiceClient:
    """Return the active ML service client."""
    if _ml_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service client not configured.",
        )
    return _ml_client


# ── Endpoint ──────────────────────────────────────────────────────────


@router.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_transaction(request: TransactionCreate) -> TransactionResponse:
    """Submit a transaction and run fraud detection.

    1. Validate the raw transaction (Pydantic).
    2. Build the ML service payload from the transaction fields.
    3. Call the ML / Fraud Intelligence Service.
    4. Merge fraud results into the transaction response.

    Returns 503 if the ML service is unavailable.
    """
    client = get_ml_client()

    # Build the payload for the ML service.
    # The current ML service expects 24 engineered features.
    # In the full integration the ML service will accept raw
    # transaction data and compute features internally.
    ml_payload = _build_ml_payload(request)

    # Call ML service
    try:
        ml_result: MLPredictionResponse = await client.predict(ml_payload)
    except MLServiceUnavailableError as exc:
        logger.error("ML service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service is unavailable.",
        )
    except MLServiceTimeoutError as exc:
        logger.error("ML service timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service timed out.",
        )
    except MLServiceResponseError as exc:
        logger.error("ML service error: %s", exc)
        if exc.status_code == 503:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML fraud detection returned an error.",
        )

    # Build response — merge transaction data with fraud results.
    # Prefer the structured explanation_detail (architecture §6) when
    # available; fall back to the legacy explanation list for older
    # ML service versions.
    explanation = None
    expl_detail = ml_result.explanation_detail
    if expl_detail is not None:
        # explanation_detail is now a typed MLExplanation model
        explanation = expl_detail
    elif ml_result.explanation is not None:
        # Legacy path — ml_result.explanation is list[MLFactor]
        explanation = MLExplanation(
            ml_top_factors=list(ml_result.explanation),
        )

    # Generate a transaction identifier for this request
    transaction_id = str(_uuid.uuid4())

    # Create alert if decision is HOLD (high-risk transaction)
    alert_summary = _maybe_create_alert(
        ml_result=ml_result,
        transaction_id=transaction_id,
        request=request,
        explanation=explanation,
    )

    return TransactionResponse(
        amount=request.amount,
        currency=request.currency,
        merchant_name=request.merchant_name,
        merchant_category=request.merchant_category,
        transaction_type=request.transaction_type,
        location_country=request.location_country,
        location_city=request.location_city,
        device_fingerprint=request.device_fingerprint,
        device_type=request.device_type,
        ip_address=request.ip_address,
        # ML results
        fraud_probability=ml_result.fraud_probability,
        fraud_prediction=ml_result.fraud_prediction,
        ml_score=ml_result.ml_score,
        behaviour_score=ml_result.behaviour_score,
        rule_score=ml_result.rule_score,
        risk_score=ml_result.risk_score,
        risk_level=ml_result.risk_level,
        decision=ml_result.decision,
        explanation=explanation,
        risk_factors=ml_result.risk_factors,
        model_version=ml_result.model_version,
        timestamp=ml_result.timestamp,
        alert=alert_summary,
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _build_ml_payload(request: TransactionCreate) -> dict[str, Any]:
    """Convert a raw transaction to the ML service request payload.

    Currently passes the raw transaction fields through.  When the ML
    service is updated to accept raw transaction data (and compute
    features internally), this function requires no changes.

    When the backend includes customer history look-up, that data
    will be added here as ``customer_id`` and ``customer_history``.
    """
    return request.model_dump()


def _maybe_create_alert(
    *,
    ml_result: MLPredictionResponse,
    transaction_id: str,
    request: TransactionCreate,
    explanation: MLExplanation | None,
) -> AlertSummary | None:
    """Create an OPEN alert if the transaction decision is HOLD.

    Returns an :class:`AlertSummary` if an alert was created, or
    ``None`` otherwise.  Alert creation is best-effort: a failure
    logs a warning but never blocks the transaction response.
    """
    if ml_result.decision != "HOLD":
        return None

    if _alert_repo is None:
        logger.warning(
            "Alert repository not configured; skipping alert creation"
        )
        return None

    # Prevent duplicate alerts for the same transaction
    existing = _alert_repo.get_by_transaction_id(transaction_id)
    if existing is not None:
        return AlertSummary(
            id=existing["id"],
            status=existing["status"],
            created_at=existing["created_at"],
        )

    # Build explanation dict for storage
    expl_json = None
    if explanation is not None:
        expl_json = explanation.model_dump()

    try:
        alert = _alert_repo.create(
            transaction_id=transaction_id,
            risk_score=ml_result.risk_score or 0,
            risk_level=ml_result.risk_level or "HIGH",
            decision=ml_result.decision,
            fraud_probability=ml_result.fraud_probability,
            model_version=ml_result.model_version,
            risk_factors=ml_result.risk_factors,
            explanation_json=expl_json,
            amount=request.amount,
            currency=request.currency,
            merchant_name=request.merchant_name,
            transaction_type=request.transaction_type,
            timestamp=ml_result.timestamp,
        )
        logger.info(
            "Alert created: id=%s transaction=%s risk_score=%s",
            alert["id"], transaction_id, alert["risk_score"],
        )
        return AlertSummary(
            id=alert["id"],
            status=alert["status"],
            created_at=alert["created_at"],
        )
    except Exception:
        logger.warning("Failed to create alert", exc_info=True)
        return None


# ── Outcome feedback endpoint ─────────────────────────────────────────


@router.patch(
    "/transactions/outcome",
    response_model=OutcomeResponse,
)
async def update_transaction_outcome(
    request: OutcomeUpdate,
) -> OutcomeResponse:
    """Update the fraud outcome of a previously recorded transaction.

    Used for the label feedback loop — after a transaction is later
    confirmed as fraudulent or legitimate, this endpoint forwards the
    update to the ML / Fraud Intelligence Service.

    Returns 404 if the target transaction cannot be found.
    """
    client = get_ml_client()

    payload = {
        "customer_id": request.customer_id,
        "timestamp": request.timestamp,
        "is_fraud": request.is_fraud,
    }

    try:
        result = await client.update_outcome(payload)
    except MLServiceUnavailableError as exc:
        logger.error("ML service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service is unavailable.",
        )
    except MLServiceTimeoutError as exc:
        logger.error("ML service timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service timed out.",
        )
    except MLServiceResponseError as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction record not found.",
            )
        logger.error("ML service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML outcome update returned an error.",
        )

    return OutcomeResponse(
        updated=result.get("updated", True),
        customer_id=result.get("customer_id", request.customer_id),
        timestamp=result.get("timestamp", request.timestamp),
        is_fraud=result.get("is_fraud", request.is_fraud),
    )
