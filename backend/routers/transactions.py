"""Transaction router — ``POST /api/v1/transactions``.

Accepts a raw transaction, calls the ML / Fraud Intelligence Service
for fraud scoring, and returns the enriched response including ML
predictions, SHAP explanations, and risk decisions.

Step 44: idempotent transaction processing via an optional
``Idempotency-Key`` request header and explicit ML failure handling.

Architecture reference: ``docs/api-contract.md`` L142–L234.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from backend.db.audit_repository import (
    ALERT_CREATED,
    DECISION_MADE,
    ML_FAILURE,
    OUTCOME_RECORDED,
    build_explanation_summary,
    build_rule_signal_summary,
    normalize_failure_category,
)
from backend.schemas import (
    AlertSummary,
    MLExplanation,
    MLPredictionResponse,
    OutcomeResponse,
    OutcomeUpdate,
    TransactionCreate,
    TransactionListItem,
    TransactionListResponse,
    TransactionResponse,
)
from backend.security.deps import get_current_user, require_roles
from backend.services.ml_client import (
    MLServiceClient,
    MLServiceResponseError,
    MLServiceTimeoutError,
    MLServiceUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transactions"])

# POST /transactions requires authentication (contract: "Auth required
# (customer)").  During this integration phase any active authenticated
# user (customer / fraud_analyst / admin) may submit transactions so
# analysts can exercise the full fraud-detection flow.
#
# Step 41: customer_id is now derived server-side from the authenticated
# user (JWT → user record → customer_id).  The client cannot forge or
# override this value.  The server-controlled customer_id is injected
# into the ML payload (for customer-specific historical features) and
# into alert creation (for customer-data association).
_require_authenticated = get_current_user

# Label feedback mutates ML training data — analyst/admin only.
_require_feedback_role = require_roles("fraud_analyst", "admin")

# Module-level client — replaced at app startup via dependency injection
# or direct assignment.  Default: localhost:8001 with 5 s timeout.
_ml_client: MLServiceClient | None = None

# Alert repository — set at app startup
_alert_repo = None

# Step 44: idempotency store — set at app startup
_idempotency_store = None

# Step 45: audit repository — set at app startup
_audit_repo = None

# Transaction repository — set at app startup (PostgreSQL persistence
# of submitted transactions with their fraud results)
_transaction_repo = None


def set_ml_client(client: MLServiceClient) -> None:
    """Set the ML service client (called during app startup)."""
    global _ml_client
    _ml_client = client


def set_alert_repository(repo: Any) -> None:
    """Set the alert repository (called during app startup)."""
    global _alert_repo
    _alert_repo = repo


def set_idempotency_store(store: Any) -> None:
    """Set the idempotency store (called during app startup)."""
    global _idempotency_store
    _idempotency_store = store


def set_audit_repository(repo: Any) -> None:
    """Set the audit repository (called during app startup)."""
    global _audit_repo
    _audit_repo = repo


def set_transaction_repository(repo: Any) -> None:
    """Set the transaction repository (called during app startup)."""
    global _transaction_repo
    _transaction_repo = repo


def get_ml_client() -> MLServiceClient:
    """Return the active ML service client."""
    if _ml_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service client not configured.",
        )
    return _ml_client


# ── Idempotency-key validation ────────────────────────────────────────


def _validate_idempotency_key(raw_key: str | None) -> str | None:
    """Validate and normalise an optional idempotency key.

    Returns the trimmed key or ``None``.  Raises :class:`HTTPException`
    (422) for keys that are empty-whitespace-only, too long, or contain
    control characters.
    """
    if raw_key is None:
        return None
    trimmed = raw_key.strip()
    if not trimmed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Idempotency-Key must not be empty.",
        )
    if len(trimmed) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Idempotency-Key must be at most 255 characters.",
        )
    # Reject control characters (0x00–0x1F) that could interfere with
    # storage or logging.
    if any(ord(ch) < 0x20 for ch in trimmed):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Idempotency-Key contains invalid characters.",
        )
    return trimmed


# ── Endpoint ──────────────────────────────────────────────────────────


@router.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_transaction(
    request: TransactionCreate,
    current_user: dict = Depends(_require_authenticated),
    response: Response = None,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> TransactionResponse:
    """Submit a transaction and run fraud detection.

    1. Validate the raw transaction (Pydantic).
    2. Check idempotency (if key provided) — return cached result for
       duplicate requests.
    3. Build the ML service payload from the transaction fields.
    4. Call the ML / Fraud Intelligence Service.
    5. Merge fraud results into the transaction response.

    Requires a valid Bearer token.  Returns 503 if the ML service is
    unavailable (Step 44: explicit failure state, no fabricated
    predictions).
    """
    client = get_ml_client()

    # Step 41: derive customer_id from the authenticated user.
    # The server controls this value — the client cannot forge it.
    customer_id = current_user.get("customer_id")

    # Step 44: validate idempotency key
    key = _validate_idempotency_key(idempotency_key)

    # Generate transaction ID early — needed for both success and failure
    transaction_id = str(_uuid.uuid4())

    # ── Idempotency check ─────────────────────────────────────────
    if key is not None and _idempotency_store is not None:
        record = _idempotency_store.try_reserve(customer_id, key)
        if record is not None:
            if record.status == "completed":
                # Duplicate request — return cached result
                cached = record.response_json
                if cached is not None:
                    if isinstance(cached, str):
                        import json
                        cached = json.loads(cached)
                    if response is not None:
                        response.status_code = status.HTTP_200_OK
                    logger.info(
                        "Idempotent replay: key=%s customer=%s",
                        key, customer_id,
                    )
                    return TransactionResponse.model_validate(cached)

            elif record.status == "processing":
                # Another request with the same key is being processed
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Request with this idempotency key is being processed.",
                )

            # status == "failed": previous attempt failed — allow retry
            # by re-reserving (the record still exists; we just proceed
            # and will update it on success/failure).

    # Build the payload for the ML service.
    # Exclude the idempotency_key from the ML payload — it is a
    # client-side duplicate-prevention key, not a transaction feature.
    ml_payload = _build_ml_payload(request, customer_id=customer_id)

    # Call ML service — explicit failure handling (Step 44)
    ml_result: MLPredictionResponse | None = None
    try:
        ml_result = await client.predict(ml_payload)
    except MLServiceUnavailableError as exc:
        logger.error("ML service unavailable: %s", exc)
        return _handle_ml_failure(
            transaction_id=transaction_id,
            customer_id=customer_id,
            request=request,
            key=key,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service is unavailable.",
            failure_category="service_unavailable",
        )
    except MLServiceTimeoutError as exc:
        logger.error("ML service timeout: %s", exc)
        return _handle_ml_failure(
            transaction_id=transaction_id,
            customer_id=customer_id,
            request=request,
            key=key,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML fraud detection service timed out.",
            failure_category="service_timeout",
        )
    except MLServiceResponseError as exc:
        logger.error("ML service error: %s", exc)
        if exc.status_code == 503:
            return _handle_ml_failure(
                transaction_id=transaction_id,
                customer_id=customer_id,
                request=request,
                key=key,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available.",
                failure_category="service_unavailable",
            )
        return _handle_ml_failure(
            transaction_id=transaction_id,
            customer_id=customer_id,
            request=request,
            key=key,
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML fraud detection returned an error.",
            failure_category="service_error",
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

    # Persist the transaction with its fraud result (Alembic schema).
    # A decision that is not durably recorded must not be reported as
    # success — persistence failure is a hard 500, and the idempotency
    # record (if any) is marked failed so the client may retry.
    # Analysts have no customer identity (customer_id is None); the
    # transactions table requires one, so those submissions stay
    # unpersisted (warning logged) and remain covered by the audit trail.
    if _transaction_repo is not None and customer_id is not None:
        expl_json = explanation.model_dump() if explanation is not None else None
        try:
            _transaction_repo.create(
                transaction_id=transaction_id,
                customer_id=customer_id,
                merchant_name=request.merchant_name,
                merchant_category=request.merchant_category,
                amount=request.amount,
                currency=request.currency,
                transaction_type=request.transaction_type,
                location_country=request.location_country,
                location_city=request.location_city,
                device_fingerprint=request.device_fingerprint,
                device_type=request.device_type,
                ip_address=request.ip_address,
                ml_score=ml_result.ml_score,
                behaviour_score=ml_result.behaviour_score,
                rule_score=ml_result.rule_score,
                risk_score=ml_result.risk_score,
                risk_level=ml_result.risk_level,
                decision=ml_result.decision,
                explanation_json=expl_json,
                model_version=ml_result.model_version,
                status="COMPLETED",
            )
        except Exception:
            logger.error(
                "Failed to persist transaction %s", transaction_id, exc_info=True
            )
            if key is not None and _idempotency_store is not None:
                _idempotency_store.mark_failed(customer_id, key)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record transaction.",
            )
    elif customer_id is None:
        logger.warning(
            "Transaction %s submitted without customer identity; "
            "not persisted to the transactions table",
            transaction_id,
        )

    # Step 45: audit successful ML decision
    _audit_decision(
        transaction_id=transaction_id,
        customer_id=customer_id,
        ml_result=ml_result,
        explanation=explanation,
    )

    # Create alert if decision is HOLD (high-risk transaction)
    alert_summary = _maybe_create_alert(
        ml_result=ml_result,
        transaction_id=transaction_id,
        request=request,
        explanation=explanation,
        customer_id=customer_id,
    )

    # Step 45: audit alert creation
    if alert_summary is not None:
        _audit_alert_created(
            transaction_id=transaction_id,
            customer_id=customer_id,
            alert_id=alert_summary.id,
            ml_result=ml_result,
        )

    txn_response = TransactionResponse(
        transaction_id=transaction_id,
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
        # Server-derived customer identity (Step 41)
        customer_id=customer_id,
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
        # Step 44: idempotency metadata
        idempotent=(key is not None),
    )

    # Step 44: cache successful result in idempotency store
    if key is not None and _idempotency_store is not None:
        _idempotency_store.mark_completed(
            customer_id,
            key,
            transaction_id,
            txn_response.model_dump(mode="json"),
        )

    return txn_response


# ── ML failure handler ─────────────────────────────────────────────────


def _handle_ml_failure(
    *,
    transaction_id: str,
    customer_id: str | None,
    request: TransactionCreate,
    key: str | None,
    status_code: int,
    detail: str,
    failure_category: str = "unknown",
) -> None:
    """Mark idempotency record as failed, then raise HTTPException.

    Raises :class:`HTTPException` — this function never returns.
    The idempotency record is marked "failed" so future retries with
    the same key are allowed.
    """
    # Step 45: audit ML failure
    _audit_ml_failure(
        transaction_id=transaction_id,
        customer_id=customer_id,
        failure_category=failure_category,
    )
    if key is not None and _idempotency_store is not None:
        _idempotency_store.mark_failed(customer_id, key)
    raise HTTPException(status_code=status_code, detail=detail)


# ── Helpers ───────────────────────────────────────────────────────────


def _build_ml_payload(
    request: TransactionCreate,
    *,
    customer_id: str | None = None,
) -> dict[str, Any]:
    """Convert a raw transaction to the ML service request payload.

    Step 41: injects the server-derived ``customer_id`` (from the
    authenticated user) so the ML service can look up the correct
    customer history.  The client cannot supply or override this value.

    Step 44: excludes ``idempotency_key`` from the ML payload — it is
    a client-side duplicate-prevention key, not a transaction feature.
    """
    payload = request.model_dump(exclude={"idempotency_key"})
    if customer_id is not None:
        payload["customer_id"] = customer_id
    return payload


def _maybe_create_alert(
    *,
    ml_result: MLPredictionResponse,
    transaction_id: str,
    request: TransactionCreate,
    explanation: MLExplanation | None,
    customer_id: str | None = None,
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
            customer_id=customer_id,
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


# ── Collection & Single Resource Endpoints ─────────────────────────────


@router.get(
    "/transactions",
    response_model=TransactionListResponse,
)
async def list_transactions(
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    risk_level: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    current_user: dict = Depends(_require_authenticated),
) -> TransactionListResponse:
    """List transactions with optional status/risk filters and pagination.

    - Customers can ONLY view their own transactions.
    - Analysts and admins can view transaction data across customers.
    """
    if _transaction_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction repository not configured.",
        )

    user_role = current_user.get("role")
    customer_id_filter: str | None = None

    if user_role == "customer":
        customer_id_filter = current_user.get("customer_id")
        if not customer_id_filter:
            return TransactionListResponse(
                items=[], total=0, page=page, per_page=per_page
            )

    per_page = min(max(1, per_page), 100)
    page = max(1, page)

    items_raw, total = _transaction_repo.list_transactions(
        customer_id=customer_id_filter,
        status=status,
        risk_level=risk_level,
        from_date=from_date,
        to_date=to_date,
        page=page,
        per_page=per_page,
    )

    items = []
    for item in items_raw:
        items.append(
            TransactionListItem(
                id=str(item["id"]),
                customer_id=str(item["customer_id"]) if item.get("customer_id") else None,
                merchant_name=item.get("merchant_name") or "N/A",
                amount=float(item["amount"]),
                currency=item.get("currency") or "USD",
                transaction_type=item.get("transaction_type") or "purchase",
                timestamp=str(item.get("timestamp") or item.get("created_at") or ""),
                status=item.get("status") or "COMPLETED",
                risk_score=item.get("risk_score"),
                risk_level=item.get("risk_level"),
                decision=item.get("decision"),
            )
        )

    return TransactionListResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionResponse,
)
async def get_transaction(
    transaction_id: str,
    current_user: dict = Depends(_require_authenticated),
) -> TransactionResponse:
    """Get single transaction detail including full fraud scoring.

    - Customers can ONLY view their own transaction.
    - Analysts and admins can view any transaction.
    """
    if _transaction_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction repository not configured.",
        )

    txn = _transaction_repo.get_by_id(transaction_id)
    if txn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    user_role = current_user.get("role")
    if user_role == "customer":
        user_customer_id = current_user.get("customer_id")
        if not user_customer_id or str(txn.get("customer_id")) != str(user_customer_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found.",
            )

    expl = txn.get("explanation") or txn.get("explanation_json")
    explanation_model = None
    if expl and isinstance(expl, dict):
        try:
            explanation_model = MLExplanation.model_validate(expl)
        except Exception:
            pass

    alert_summary = None
    if _alert_repo is not None:
        alert = _alert_repo.get_by_transaction_id(transaction_id)
        if alert is not None:
            alert_summary = AlertSummary(
                id=str(alert["id"]),
                status=alert["status"],
                created_at=str(alert["created_at"]),
            )

    return TransactionResponse(
        transaction_id=str(txn["id"]),
        id=str(txn["id"]),
        amount=float(txn["amount"]),
        currency=txn.get("currency") or "USD",
        merchant_name=txn.get("merchant_name") or "N/A",
        merchant_category=txn.get("merchant_category") or "N/A",
        transaction_type=txn.get("transaction_type") or "purchase",
        location_country=txn.get("location_country") or "Unknown",
        location_city=txn.get("location_city") or "Unknown",
        device_fingerprint=txn.get("device_fingerprint") or "Unknown",
        device_type=txn.get("device_type") or "desktop",
        ip_address=txn.get("ip_address") or "0.0.0.0",
        customer_id=str(txn["customer_id"]) if txn.get("customer_id") else None,
        status=txn.get("status") or "COMPLETED",
        ml_score=txn.get("ml_score"),
        behaviour_score=txn.get("behaviour_score"),
        rule_score=txn.get("rule_score"),
        risk_score=txn.get("risk_score"),
        risk_level=txn.get("risk_level"),
        decision=txn.get("decision"),
        explanation=explanation_model,
        risk_factors=txn.get("risk_factors") or [],
        model_version=txn.get("model_version"),
        timestamp=txn.get("timestamp"),
        alert=alert_summary,
    )


# ── Outcome feedback endpoint ─────────────────────────────────────────


@router.patch(
    "/transactions/outcome",
    response_model=OutcomeResponse,
)
async def update_transaction_outcome(
    request: OutcomeUpdate,
    current_user: dict = Depends(_require_feedback_role),
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

    # Step 45: audit outcome feedback
    _audit_outcome(
        customer_id=request.customer_id,
        actor_id=current_user["id"],
        actor_role=current_user.get("role"),
        is_fraud=request.is_fraud,
    )

    return OutcomeResponse(
        updated=result.get("updated", True),
        customer_id=result.get("customer_id", request.customer_id),
        timestamp=result.get("timestamp", request.timestamp),
        is_fraud=result.get("is_fraud", request.is_fraud),
    )


# ── Step 45: Audit helpers ────────────────────────────────────────────


def _audit_decision(
    *,
    transaction_id: str,
    customer_id: str | None,
    ml_result: MLPredictionResponse,
    explanation: MLExplanation | None,
) -> None:
    """Audit a successful ML decision (best-effort, never blocks)."""
    if _audit_repo is None or customer_id is None:
        return
    try:
        _audit_repo.create(
            transaction_id=transaction_id,
            customer_id=customer_id,
            event_type=DECISION_MADE,
            decision=ml_result.decision,
            risk_score=ml_result.risk_score,
            risk_level=ml_result.risk_level,
            fraud_probability=ml_result.fraud_probability,
            model_version=ml_result.model_version,
            explanation_summary=build_explanation_summary(explanation),
            rule_signal_summary=build_rule_signal_summary(
                ml_result.risk_factors, explanation,
            ),
        )
    except Exception:
        logger.warning("Failed to audit decision event", exc_info=True)


def _audit_ml_failure(
    *,
    transaction_id: str,
    customer_id: str | None,
    failure_category: str,
) -> None:
    """Audit an ML failure (best-effort, never blocks)."""
    if _audit_repo is None or customer_id is None:
        return
    try:
        _audit_repo.create(
            transaction_id=transaction_id,
            customer_id=customer_id,
            event_type=ML_FAILURE,
            failure_category=normalize_failure_category(failure_category),
        )
    except Exception:
        logger.warning("Failed to audit ML failure event", exc_info=True)


def _audit_alert_created(
    *,
    transaction_id: str,
    customer_id: str | None,
    alert_id: str,
    ml_result: MLPredictionResponse,
) -> None:
    """Audit alert creation (best-effort, never blocks)."""
    if _audit_repo is None or customer_id is None:
        return
    try:
        _audit_repo.create(
            transaction_id=transaction_id,
            customer_id=customer_id,
            event_type=ALERT_CREATED,
            decision=ml_result.decision,
            risk_score=ml_result.risk_score,
            risk_level=ml_result.risk_level,
            alert_id=alert_id,
        )
    except Exception:
        logger.warning("Failed to audit alert creation event", exc_info=True)


def _audit_outcome(
    *,
    customer_id: str,
    actor_id: str,
    actor_role: str | None,
    is_fraud: int,
) -> None:
    """Audit outcome feedback (best-effort, never blocks).

    Outcome feedback targets the ML service's history store, not a
    specific transaction_id.  We use customer_id as the transaction_id
    reference to maintain the audit chain.
    """
    if _audit_repo is None:
        return
    try:
        _audit_repo.create(
            # Outcome feedback does not target a specific transaction;
            # use a deterministic placeholder UUID so the audit record
            # is still queryable by customer context.
            transaction_id="00000000-0000-4000-8000-000000000000",
            customer_id=customer_id,
            event_type=OUTCOME_RECORDED,
            actor_id=actor_id,
            actor_role=actor_role,
            metadata={"is_fraud": is_fraud},
        )
    except Exception:
        logger.warning("Failed to audit outcome event", exc_info=True)
