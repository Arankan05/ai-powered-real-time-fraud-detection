"""Audit router — ``GET /api/v1/audit/transactions/{transaction_id}``.

Step 45: Authorised retrieval of fraud decision audit trails.

Security
--------
* Authentication is required (Bearer token).
* **fraud_analyst** / **admin** roles may view any transaction's
  audit trail.
* **customer** role may only view their *own* audit trail — the
  ``customer_id`` is derived from the JWT, never from the request body
  or path parameter.
* Unauthorised lookups return 403 (not 404) to avoid information
  leakage about transaction existence.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.db.audit_repository import AuditRepository
from backend.schemas import AuditEventResponse, AuditTrailResponse
from backend.security.deps import get_current_user, require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

# Audit retrieval: analysts/admins have full access; customers can
# only access their own audit trails (enforced in the endpoint).
_require_analyst = require_roles("fraud_analyst", "admin")

# Module-level repository — set at app startup
_audit_repo: AuditRepository | None = None


def set_audit_repository(repo: AuditRepository) -> None:
    """Set the audit repository (called during app startup)."""
    global _audit_repo
    _audit_repo = repo


def get_audit_repository() -> AuditRepository:
    """Return the active audit repository."""
    if _audit_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit service not configured.",
        )
    return _audit_repo


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get(
    "/transactions/{transaction_id}",
    response_model=AuditTrailResponse,
)
def get_transaction_audit_trail(
    transaction_id: str,
    current_user: dict = Depends(get_current_user),
) -> AuditTrailResponse:
    """Retrieve the audit trail for a transaction.

    * **fraud_analyst / admin**: full access to any transaction.
    * **customer**: may only access their own audit trail.  Ownership is
      verified from the JWT ``customer_id`` claim — the client cannot
      forge or override this value.

    Returns 403 for customers trying to access another customer's trail.
    Returns 404 when no audit events exist for the transaction (for
    analysts/admins) or when the transaction does not belong to the
    requesting customer.
    """
    repo = get_audit_repository()

    user_role = current_user.get("role")
    user_customer_id = current_user.get("customer_id")

    events = repo.list_by_transaction(transaction_id)

    if not events:
        # For customers, always return 403 to avoid leaking transaction
        # existence.  For analysts/admins, return 404.
        if user_role == "customer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No audit events found for this transaction.",
        )

    # Customer isolation: customers may only see their own audit trail.
    if user_role == "customer":
        # All events for a transaction share the same customer_id.
        event_customer_id = events[0].get("customer_id")
        if event_customer_id != user_customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied.",
            )

    # Build response — filter sensitive fields (defence in depth).
    safe_events = []
    for ev in events:
        safe_events.append(AuditEventResponse(
            audit_id=ev["audit_id"],
            transaction_id=ev["transaction_id"],
            customer_id=ev["customer_id"],
            event_type=ev["event_type"],
            decision=ev.get("decision"),
            risk_score=ev.get("risk_score"),
            risk_level=ev.get("risk_level"),
            fraud_probability=ev.get("fraud_probability"),
            model_version=ev.get("model_version"),
            explanation_summary=ev.get("explanation_summary"),
            rule_signal_summary=ev.get("rule_signal_summary"),
            failure_category=ev.get("failure_category"),
            actor_id=ev.get("actor_id"),
            actor_role=ev.get("actor_role"),
            previous_state=ev.get("previous_state"),
            new_state=ev.get("new_state"),
            alert_id=ev.get("alert_id"),
            created_at=ev["created_at"],
        ))

    return AuditTrailResponse(
        transaction_id=transaction_id,
        events=safe_events,
    )
