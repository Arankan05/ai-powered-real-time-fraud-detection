"""Transaction business logic — orchestrates the full fraud detection pipeline.

Flow:

1. Validate request
2. Verify customer ownership / authorization
3. Look up or create merchant
4. Build customer history for ML request
5. Call ML/Fraud Intelligence Service
6. Persist transaction with fraud results
7. Create alert if decision == HOLD
8. Return complete response
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.errors import (
    AppException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
)
from app.models.user import User
from app.repositories.transaction import (
    AlertRepository,
    AuditLogRepository,
    CustomerRepository,
    MerchantRepository,
    TransactionRepository,
)
from app.schemas.transaction import (
    AlertSummary,
    ExplanationResponse,
    FraudCheckRequest,
    FraudCheckResponse,
    TransactionCreateRequest,
    TransactionDetailResponse,
    TransactionListResponse,
    TransactionQueryParams,
    TransactionSummaryResponse,
)
from app.services.ml import (
    MLInvalidResponseError,
    MLServiceClient,
    MLServiceUnavailableError,
    build_customer_history,
    build_transaction_payload,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_risk_factors(explanation: dict) -> list[str]:
    """Extract risk factor names from the ML explanation object."""
    factors: list[str] = []
    for f in explanation.get("ml_top_factors", []):
        if "feature" in f:
            factors.append(f["feature"])
    for s in explanation.get("behaviour_signals", []):
        if "signal" in s:
            factors.append(s["signal"])
    for r in explanation.get("rules_triggered", []):
        if "rule" in r:
            factors.append(r["rule"])
    return factors


def _build_detail_response(
    *,
    txn_id: UUID,
    customer_id: UUID,
    merchant_id: UUID | None,
    amount: Decimal,
    currency: str,
    merchant_name: str,
    merchant_category: str | None,
    transaction_type: str,
    location_country: str | None,
    location_city: str | None,
    device_fingerprint: str | None,
    device_type: str | None,
    ip_address: str | None,
    timestamp: datetime,
    status: str,
    ml_score: int | None,
    behaviour_score: int | None,
    rule_score: int | None,
    risk_score: int | None,
    risk_level: str | None,
    decision: str | None,
    explanation_json: dict | None,
    risk_factors: list[str],
    model_version: str | None,
    alert_data: AlertSummary | None,
) -> TransactionDetailResponse:
    explanation = None
    if explanation_json is not None:
        explanation = ExplanationResponse(**explanation_json)

    return TransactionDetailResponse(
        id=txn_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=amount,
        currency=currency,
        merchant_name=merchant_name,
        merchant_category=merchant_category,
        transaction_type=transaction_type,
        location_country=location_country,
        location_city=location_city,
        device_fingerprint=device_fingerprint,
        device_type=device_type,
        ip_address=ip_address,
        timestamp=timestamp,
        status=status,
        ml_score=ml_score,
        behaviour_score=behaviour_score,
        rule_score=rule_score,
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision,
        explanation=explanation,
        risk_factors=risk_factors,
        model_version=model_version,
        alert=alert_data,
    )


def _txn_to_detail(txn: Any, alert: Any | None = None) -> TransactionDetailResponse:
    """Map a Transaction ORM object to a detail response."""
    merchant_name = ""
    merchant_category = None
    if txn.merchant:
        merchant_name = txn.merchant.name
        merchant_category = txn.merchant.category_code

    risk_factors = _extract_risk_factors(txn.explanation_json or {})

    alert_summary = None
    if alert is not None:
        alert_summary = AlertSummary(
            id=alert.id, status=alert.status, created_at=alert.created_at,
        )
    elif txn.alert is not None:
        alert_summary = AlertSummary(
            id=txn.alert.id,
            status=txn.alert.status,
            created_at=txn.alert.created_at,
        )

    return _build_detail_response(
        txn_id=txn.id,
        customer_id=txn.customer_id,
        merchant_id=txn.merchant_id,
        amount=txn.amount,
        currency=txn.currency,
        merchant_name=merchant_name,
        merchant_category=merchant_category,
        transaction_type=txn.transaction_type,
        location_country=txn.location_country,
        location_city=txn.location_city,
        device_fingerprint=txn.device_fingerprint,
        device_type=txn.device_type,
        ip_address=txn.ip_address,
        timestamp=txn.timestamp,
        status=txn.status,
        ml_score=txn.ml_score,
        behaviour_score=txn.behaviour_score,
        rule_score=txn.rule_score,
        risk_score=txn.risk_score,
        risk_level=txn.risk_level,
        decision=txn.decision,
        explanation_json=txn.explanation_json,
        risk_factors=risk_factors,
        model_version=txn.model_version,
        alert_data=alert_summary,
    )


def _txn_to_summary(txn: Any) -> TransactionSummaryResponse:
    """Map a Transaction ORM object to a summary response."""
    merchant_name = ""
    if txn.merchant:
        merchant_name = txn.merchant.name

    return TransactionSummaryResponse(
        id=txn.id,
        customer_id=txn.customer_id,
        merchant_name=merchant_name,
        amount=txn.amount,
        currency=txn.currency,
        transaction_type=txn.transaction_type,
        timestamp=txn.timestamp,
        status=txn.status,
        risk_score=txn.risk_score,
        risk_level=txn.risk_level,
        decision=txn.decision,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TransactionService:
    """Orchestrates transaction creation and fraud detection pipeline."""

    def __init__(self, db: Session, ml_client: MLServiceClient | None = None) -> None:
        self._db = db
        self._txn_repo = TransactionRepository(db)
        self._merchant_repo = MerchantRepository(db)
        self._customer_repo = CustomerRepository(db)
        self._alert_repo = AlertRepository(db)
        self._audit_repo = AuditLogRepository(db)
        self._ml_client = ml_client or MLServiceClient()

    # -- Create transaction -----------------------------------------------

    def create_transaction(
        self,
        data: TransactionCreateRequest,
        current_user: User,
    ) -> TransactionDetailResponse:
        """Full transaction creation + fraud detection pipeline."""
        # 1. Verify user is a customer
        if current_user.role != "customer":
            raise ForbiddenException(detail="Only customers can create transactions")

        customer_id = current_user.customer_id
        if customer_id is None:
            raise ForbiddenException(detail="User has no linked customer profile")

        # 2. Verify customer exists
        customer = self._customer_repo.get_by_id(customer_id)
        if customer is None:
            raise NotFoundException(detail="Customer not found")

        # 3. Look up or create merchant
        merchant = self._merchant_repo.get_or_create(
            name=data.merchant_name,
            category_code=data.merchant_category,
        )

        # 4. Build customer history for ML
        history_stats = self._txn_repo.get_customer_history_stats(customer_id)
        customer_history = build_customer_history(**history_stats)

        # 5. Build transaction payload for ML
        now = datetime.now(timezone.utc)
        txn_payload = build_transaction_payload(
            amount=float(data.amount),
            currency=data.currency,
            merchant_id=str(merchant.id),
            merchant_name=data.merchant_name,
            merchant_category=data.merchant_category,
            transaction_type=data.transaction_type,
            location_country=data.location_country,
            location_city=data.location_city,
            device_fingerprint=data.device_fingerprint,
            device_type=data.device_type,
            ip_address=data.ip_address,
            timestamp=now.isoformat(),
        )

        # 6. Call ML service FIRST (before persisting)
        try:
            ml_result = self._ml_client.predict(
                customer_id=customer_id,
                customer_history=customer_history,
                transaction_data=txn_payload,
            )
        except MLServiceUnavailableError as exc:
            raise AppException(
                status_code=503,
                detail="ML service is currently unavailable",
                error_code="ML_SERVICE_UNAVAILABLE",
            )
        except MLInvalidResponseError as exc:
            logger.error("Invalid ML response: %s", exc)
            raise AppException(
                status_code=500,
                detail="An internal server error occurred",
                error_code="INTERNAL_ERROR",
            )

        # 7. Persist transaction with fraud results
        txn = self._txn_repo.create(
            customer_id=customer_id,
            merchant_id=merchant.id,
            amount=data.amount,
            currency=data.currency,
            transaction_type=data.transaction_type,
            location_country=data.location_country,
            location_city=data.location_city,
            device_fingerprint=data.device_fingerprint,
            device_type=data.device_type,
            ip_address=data.ip_address,
        )

        self._txn_repo.update_with_fraud_results(
            txn,
            ml_score=ml_result["ml_score"],
            behaviour_score=ml_result["behaviour_score"],
            rule_score=ml_result["rule_score"],
            risk_score=ml_result["risk_score"],
            risk_level=ml_result["risk_level"],
            decision=ml_result["decision"],
            explanation_json=ml_result["explanation"],
            model_version=ml_result.get("model_version"),
            status="COMPLETED",
        )

        # 8. Create alert if HOLD
        alert = None
        if ml_result["decision"] == "HOLD":
            alert = self._alert_repo.create(
                transaction_id=txn.id,
                risk_score=ml_result["risk_score"],
                risk_level=ml_result["risk_level"],
                decision=ml_result["decision"],
                explanation_json=ml_result["explanation"],
            )

        # 9. Audit logging
        self._audit_repo.create(
            actor_id=current_user.id,
            action="transaction_created",
            resource_type="transaction",
            resource_id=str(txn.id),
            details_json={
                "amount": float(data.amount),
                "currency": data.currency,
                "decision": ml_result["decision"],
                "risk_level": ml_result["risk_level"],
            },
        )
        if alert is not None:
            self._audit_repo.create(
                actor_id=current_user.id,
                action="alert_created",
                resource_type="alert",
                resource_id=str(alert.id),
                details_json={
                    "transaction_id": str(txn.id),
                    "risk_level": ml_result["risk_level"],
                    "decision": ml_result["decision"],
                },
            )

        self._db.commit()
        self._db.refresh(txn)

        return _txn_to_detail(txn, alert=alert)

    # -- Get transaction by ID -------------------------------------------

    def get_transaction(
        self, transaction_id: UUID, current_user: User,
    ) -> TransactionDetailResponse:
        """Get a single transaction with full details."""
        txn = self._txn_repo.get_by_id(transaction_id)
        if txn is None:
            raise NotFoundException(detail="Transaction not found")

        # Ownership check: customers see own only
        if current_user.role == "customer":
            if txn.customer_id != current_user.customer_id:
                raise ForbiddenException(detail="Insufficient permissions")

        return _txn_to_detail(txn)

    # -- List transactions ------------------------------------------------

    def list_transactions(
        self, current_user: User, params: TransactionQueryParams,
    ) -> TransactionListResponse:
        """List transactions with pagination and filters."""
        customer_id = None
        if current_user.role == "customer":
            customer_id = current_user.customer_id

        items, total = self._txn_repo.list_transactions(
            customer_id=customer_id,
            status=params.status,
            risk_level=params.risk_level,
            from_date=params.from_date,
            to_date=params.to_date,
            page=params.page,
            per_page=params.per_page,
        )

        return TransactionListResponse(
            items=[_txn_to_summary(t) for t in items],
            total=total,
            page=params.page,
            per_page=params.per_page,
        )

    # -- Customer transactions -------------------------------------------

    def list_customer_transactions(
        self, customer_id: UUID, current_user: User, params: TransactionQueryParams,
    ) -> TransactionListResponse:
        """List transactions for a specific customer."""
        # Ownership check
        if current_user.role == "customer":
            if customer_id != current_user.customer_id:
                raise ForbiddenException(detail="Insufficient permissions")

        # Verify customer exists
        if self._customer_repo.get_by_id(customer_id) is None:
            raise NotFoundException(detail="Customer not found")

        items, total = self._txn_repo.list_transactions(
            customer_id=customer_id,
            status=params.status,
            risk_level=params.risk_level,
            from_date=params.from_date,
            to_date=params.to_date,
            page=params.page,
            per_page=params.per_page,
        )

        return TransactionListResponse(
            items=[_txn_to_summary(t) for t in items],
            total=total,
            page=params.page,
            per_page=params.per_page,
        )

    # -- Fraud check (no persistence) ------------------------------------

    def fraud_check(
        self, data: FraudCheckRequest, current_user: User,
    ) -> FraudCheckResponse:
        """Run fraud analysis without persisting (analysts/admins only)."""
        # Verify customer exists
        customer = self._customer_repo.get_by_id(data.customer_id)
        if customer is None:
            raise NotFoundException(detail="Customer not found")

        # Build customer history
        history_stats = self._txn_repo.get_customer_history_stats(data.customer_id)
        customer_history = build_customer_history(**history_stats)

        now = datetime.now(timezone.utc)
        txn_payload = build_transaction_payload(
            amount=float(data.amount),
            currency=data.currency,
            merchant_id=None,
            merchant_name=data.merchant_name,
            merchant_category=data.merchant_category,
            transaction_type=data.transaction_type,
            location_country=data.location_country,
            location_city=data.location_city,
            device_fingerprint=data.device_fingerprint,
            device_type=data.device_type,
            ip_address=data.ip_address,
            timestamp=now.isoformat(),
        )

        try:
            ml_result = self._ml_client.predict(
                customer_id=data.customer_id,
                customer_history=customer_history,
                transaction_data=txn_payload,
            )
        except MLServiceUnavailableError:
            raise AppException(
                status_code=503,
                detail="ML service is currently unavailable",
                error_code="ML_SERVICE_UNAVAILABLE",
            )
        except MLInvalidResponseError as exc:
            logger.error("Invalid ML response: %s", exc)
            raise AppException(
                status_code=500,
                detail="An internal server error occurred",
                error_code="INTERNAL_ERROR",
            )

        explanation = ExplanationResponse(**ml_result["explanation"])

        return FraudCheckResponse(
            ml_score=ml_result["ml_score"],
            behaviour_score=ml_result["behaviour_score"],
            rule_score=ml_result["rule_score"],
            risk_score=ml_result["risk_score"],
            risk_level=ml_result["risk_level"],
            decision=ml_result["decision"],
            explanation=explanation,
            risk_factors=ml_result.get("risk_factors", []),
            model_version=ml_result["model_version"],
        )
