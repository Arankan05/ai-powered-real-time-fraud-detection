"""Alert business logic — list, detail, and status updates.

Authorization:
    Only ``fraud_analyst`` and ``admin`` roles can access alert endpoints.
    Customers are explicitly denied.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.errors import AppException, ForbiddenException, NotFoundException
from app.models.alert import Alert
from app.models.user import User
from app.repositories.transaction import AlertRepository, AuditLogRepository
from app.schemas.alert import (
    AlertDetailResponse,
    AlertListItem,
    AlertListResponse,
    AlertTransactionDetail,
    AlertTransactionSummary,
    AlertUpdateRequest,
    AlertUpdateResponse,
    ExplanationDetail,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"RESOLVED", "DISMISSED"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_risk_factors(explanation: dict | None) -> list[str]:
    """Extract risk factor names from the explanation JSONB."""
    if not explanation:
        return []
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


def _customer_email_from_txn(txn: Alert) -> str:
    """Best-effort customer email from the transaction relationship."""
    try:
        if txn.transaction and txn.transaction.customer:
            # We don't store email on customer; get it from user linked
            pass
    except Exception:
        pass
    # Fallback: we can query via the customer_id on the transaction
    return ""


def _build_transaction_summary(alert: Alert) -> AlertTransactionSummary:
    """Build the transaction_summary from the alert's transaction."""
    txn = alert.transaction
    merchant_name = ""
    if txn.merchant:
        merchant_name = txn.merchant.name

    # Get customer email via the customer relationship
    customer_email = ""
    if txn.customer:
        # Customer model doesn't have email; look up user
        # This is handled at the service level with an extra query
        pass

    return AlertTransactionSummary(
        amount=txn.amount,
        currency=txn.currency,
        merchant_name=merchant_name,
        transaction_type=txn.transaction_type,
        customer_email=customer_email,
        timestamp=txn.timestamp,
    )


def _build_transaction_detail(alert: Alert) -> AlertTransactionDetail:
    """Build the full transaction detail from the alert's transaction."""
    txn = alert.transaction
    merchant_name = ""
    if txn.merchant:
        merchant_name = txn.merchant.name

    return AlertTransactionDetail(
        id=txn.id,
        customer_id=txn.customer_id,
        amount=txn.amount,
        currency=txn.currency,
        merchant_name=merchant_name,
        transaction_type=txn.transaction_type,
        location_country=txn.location_country,
        location_city=txn.location_city,
        device_type=txn.device_type,
        timestamp=txn.timestamp,
        ml_score=txn.ml_score,
        behaviour_score=txn.behaviour_score,
        rule_score=txn.rule_score,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AlertService:
    """Orchestrates alert retrieval, filtering, and status management."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._alert_repo = AlertRepository(db)
        self._audit_repo = AuditLogRepository(db)

    # -- List alerts -----------------------------------------------------

    def list_alerts(
        self,
        current_user: User,
        *,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
        risk_level: str | None = None,
    ) -> AlertListResponse:
        """List alerts with pagination and filters (analyst/admin only)."""
        self._require_analyst_or_admin(current_user)

        items, total = self._alert_repo.list_alerts(
            status=status,
            risk_level=risk_level,
            page=page,
            per_page=per_page,
        )

        result_items: list[AlertListItem] = []
        for alert in items:
            # Build transaction summary with customer email
            txn = alert.transaction
            merchant_name = ""
            if txn and txn.merchant:
                merchant_name = txn.merchant.name
            customer_email = self._get_customer_email(txn) if txn else ""

            summary = AlertTransactionSummary(
                amount=txn.amount,
                currency=txn.currency,
                merchant_name=merchant_name,
                transaction_type=txn.transaction_type,
                customer_email=customer_email,
                timestamp=txn.timestamp,
            ) if txn else AlertTransactionSummary(
                amount=0, currency="USD", merchant_name="",
                transaction_type="", customer_email="",
                timestamp=datetime.now(timezone.utc),
            )

            result_items.append(AlertListItem(
                id=alert.id,
                transaction_id=alert.transaction_id,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                decision=alert.decision,
                status=alert.status,
                analyst_id=alert.analyst_id,
                notes=alert.notes,
                created_at=alert.created_at,
                resolved_at=alert.resolved_at,
                transaction_summary=summary,
            ))

        return AlertListResponse(
            items=result_items,
            total=total,
            page=page,
            per_page=per_page,
        )

    # -- Get alert detail ------------------------------------------------

    def get_alert(
        self, alert_id: UUID, current_user: User,
    ) -> AlertDetailResponse:
        """Get a single alert with full transaction detail."""
        self._require_analyst_or_admin(current_user)

        alert = self._alert_repo.get_by_id(alert_id)
        if alert is None:
            raise NotFoundException(detail="Alert not found")

        explanation = None
        if alert.explanation_json:
            explanation = ExplanationDetail(**alert.explanation_json)

        risk_factors = _extract_risk_factors(alert.explanation_json)
        txn_detail = _build_transaction_detail(alert)

        return AlertDetailResponse(
            id=alert.id,
            transaction_id=alert.transaction_id,
            risk_score=alert.risk_score,
            risk_level=alert.risk_level,
            decision=alert.decision,
            explanation=explanation,
            risk_factors=risk_factors,
            status=alert.status,
            analyst_id=alert.analyst_id,
            notes=alert.notes,
            created_at=alert.created_at,
            resolved_at=alert.resolved_at,
            transaction=txn_detail,
        )

    # -- Update alert ----------------------------------------------------

    def update_alert(
        self,
        alert_id: UUID,
        data: AlertUpdateRequest,
        current_user: User,
    ) -> AlertUpdateResponse:
        """Update alert status and/or notes with state-transition enforcement."""
        self._require_analyst_or_admin(current_user)

        # At least one field must be provided
        if data.status is None and data.notes is None:
            raise AppException(
                status_code=422,
                detail="At least one of 'status' or 'notes' must be provided",
                error_code="VALIDATION_ERROR",
            )

        alert = self._alert_repo.get_by_id(alert_id)
        if alert is None:
            raise NotFoundException(detail="Alert not found")

        # Validate status transition
        if data.status is not None:
            try:
                AlertUpdateRequest.validate_transition(alert.status, data.status)
            except ValueError as exc:
                raise AppException(
                    status_code=400,
                    detail=str(exc),
                    error_code="INVALID_STATUS_TRANSITION",
                )

        # Determine resolved_at
        resolved_at = None
        if data.status in _TERMINAL_STATUSES:
            resolved_at = datetime.now(timezone.utc)

        # Update alert
        self._alert_repo.update(
            alert,
            status=data.status,
            notes=data.notes,
            analyst_id=current_user.id,
            resolved_at=resolved_at,
        )

        # Audit log for alert update
        self._audit_repo.create(
            actor_id=current_user.id,
            action="alert_updated",
            resource_type="alert",
            resource_id=str(alert.id),
            details_json={
                "new_status": data.status,
                "previous_status": alert.status if data.status else None,
            },
        )

        self._db.commit()
        self._db.refresh(alert)

        return AlertUpdateResponse(
            id=alert.id,
            transaction_id=alert.transaction_id,
            risk_score=alert.risk_score,
            risk_level=alert.risk_level,
            decision=alert.decision,
            status=alert.status,
            analyst_id=alert.analyst_id,
            notes=alert.notes,
            created_at=alert.created_at,
            resolved_at=alert.resolved_at,
        )

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _require_analyst_or_admin(user: User) -> None:
        if user.role not in {"fraud_analyst", "admin"}:
            raise ForbiddenException(detail="Insufficient permissions")

    def _get_customer_email(self, txn) -> str:
        """Look up the customer email for a transaction."""
        from app.models.user import User as UserModel
        from sqlalchemy import select

        if txn.customer_id is None:
            return ""
        stmt = select(UserModel.email).where(
            UserModel.customer_id == txn.customer_id
        )
        result = self._db.execute(stmt).scalar_one_or_none()
        return result or ""
