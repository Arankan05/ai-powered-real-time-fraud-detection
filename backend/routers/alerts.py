"""Alert router — ``GET /api/v1/alerts`` and ``PATCH /api/v1/alerts/{id}``.

Provides fraud analyst endpoints for listing and managing alerts.
Alerts are created automatically by the transaction flow when
``decision == HOLD``; this router only exposes read and update
operations.

Architecture reference: ``docs/api-contract.md`` L352–L494.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.db.alert_repository import (
    VALID_STATUSES,
    AlertRepository,
)
from backend.schemas import (
    AlertListResponse,
    AlertResponse,
    AlertUpdate,
    MLExplanation,
    TransactionSummary,
)
from backend.security.deps import require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["alerts"])

# Alert investigation is restricted to fraud analysts and admins
# (contract §Alert Endpoints).
_require_analyst = require_roles("fraud_analyst", "admin")

# Module-level repository — set at app startup
_alert_repo: AlertRepository | None = None


def set_alert_repository(repo: AlertRepository) -> None:
    """Set the alert repository (called during app startup)."""
    global _alert_repo
    _alert_repo = repo


def get_alert_repository() -> AlertRepository:
    """Return the active alert repository."""
    if _alert_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alert service not configured.",
        )
    return _alert_repo


# ── Helpers ───────────────────────────────────────────────────────────


def _alert_dict_to_response(alert: dict[str, Any]) -> AlertResponse:
    """Convert a raw alert dict from the repository to an AlertResponse."""
    explanation = None
    expl_data = alert.get("explanation_json")
    if expl_data and isinstance(expl_data, dict):
        explanation = MLExplanation.model_validate(expl_data)

    txn_summary = None
    if alert.get("amount") is not None:
        txn_summary = TransactionSummary(
            amount=alert.get("amount"),
            currency=alert.get("currency"),
            merchant_name=alert.get("merchant_name"),
            transaction_type=alert.get("transaction_type"),
            timestamp=alert.get("timestamp"),
        )

    return AlertResponse(
        id=alert["id"],
        transaction_id=alert["transaction_id"],
        customer_id=alert.get("customer_id"),
        risk_score=alert["risk_score"],
        risk_level=alert["risk_level"],
        decision=alert["decision"],
        status=alert["status"],
        analyst_id=alert.get("analyst_id"),
        notes=alert.get("notes"),
        created_at=alert["created_at"],
        updated_at=alert.get("updated_at"),
        resolved_at=alert.get("resolved_at"),
        fraud_probability=alert.get("fraud_probability"),
        model_version=alert.get("model_version"),
        risk_factors=alert.get("risk_factors"),
        explanation=explanation,
        transaction_summary=txn_summary,
    )


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/alerts", response_model=AlertListResponse)
def list_alerts(
    status_filter: str | None = Query(
        None, alias="status",
        description="Filter by status: OPEN, IN_REVIEW, RESOLVED, DISMISSED",
    ),
    risk_level: str | None = Query(
        None, description="Filter by risk level: LOW, MEDIUM, HIGH",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: dict = Depends(_require_analyst),
) -> AlertListResponse:
    """List fraud alerts with optional filtering and pagination."""
    repo = get_alert_repository()

    # Validate filter values
    if status_filter and status_filter not in VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status filter: {status_filter}. "
            f"Must be one of: {sorted(VALID_STATUSES)}",
        )
    if risk_level and risk_level not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid risk_level filter: {risk_level}. "
            f"Must be one of: LOW, MEDIUM, HIGH",
        )

    alerts, total = repo.list_alerts(
        status=status_filter,
        risk_level=risk_level,
        page=page,
        per_page=per_page,
    )

    return AlertListResponse(
        items=[_alert_dict_to_response(a) for a in alerts],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    current_user: dict = Depends(_require_analyst),
) -> AlertResponse:
    """Get a single alert by ID with full details."""
    repo = get_alert_repository()
    alert = repo.get_by_id(alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found.",
        )
    return _alert_dict_to_response(alert)


@router.patch("/alerts/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: str,
    request: AlertUpdate,
    current_user: dict = Depends(_require_analyst),
) -> AlertResponse:
    """Update alert status and/or analyst notes.

    Valid status transitions:
    - OPEN → IN_REVIEW, RESOLVED, DISMISSED
    - IN_REVIEW → RESOLVED, DISMISSED
    - RESOLVED / DISMISSED → terminal (no further changes)

    When status changes to RESOLVED or DISMISSED, ``resolved_at`` is
    set automatically.

    ``analyst_id`` is always taken from the authenticated user — it
    cannot be supplied or overridden by the client.
    """
    # At least one field must be provided
    if request.status is None and request.notes is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'status' or 'notes' must be provided.",
        )

    # Validate status value if provided
    if request.status is not None and request.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status: {request.status}. "
            f"Must be one of: {sorted(VALID_STATUSES)}",
        )

    repo = get_alert_repository()

    if request.status is not None:
        updated = repo.update_status(
            alert_id,
            new_status=request.status,
            notes=request.notes,
            analyst_id=current_user["id"],
        )
        if updated is None:
            # Either alert not found or invalid transition
            existing = repo.get_by_id(alert_id)
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Alert not found.",
                )
            # Alert exists but transition is invalid
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status transition from "
                f"'{existing['status']}' to '{request.status}'.",
            )
    else:
        # Only notes update — get current alert, update notes
        existing = repo.get_by_id(alert_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found.",
            )
        # Use current status (no transition) to update notes
        updated = repo.update_status(
            alert_id,
            new_status=existing["status"],
            notes=request.notes,
            analyst_id=current_user["id"],
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found.",
            )

    return _alert_dict_to_response(updated)
