"""Pydantic schemas for alert endpoints.

Response shapes match ``docs/api-contract.md`` exactly.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AlertTransactionSummary(BaseModel):
    """Transaction summary embedded in alert list items."""

    amount: Decimal
    currency: str
    merchant_name: str
    transaction_type: str
    customer_email: str
    timestamp: datetime

    @field_serializer("amount")
    def _serialize_amount(self, v: Decimal) -> float:
        return float(v)


class AlertTransactionDetail(BaseModel):
    """Full transaction detail embedded in alert detail response."""

    id: UUID
    customer_id: UUID
    amount: Decimal
    currency: str
    merchant_name: str
    transaction_type: str
    location_country: str | None = None
    location_city: str | None = None
    device_type: str | None = None
    timestamp: datetime
    ml_score: int | None = None
    behaviour_score: int | None = None
    rule_score: int | None = None

    @field_serializer("amount")
    def _serialize_amount(self, v: Decimal) -> float:
        return float(v)


class ExplanationDetail(BaseModel):
    """ML explanation included in the alert detail response."""

    ml_top_factors: list[dict] = []
    behaviour_signals: list[dict] = []
    rules_triggered: list[dict] = []


# ---------------------------------------------------------------------------
# Alert list item
# ---------------------------------------------------------------------------


class AlertListItem(BaseModel):
    """Single item in the paginated alert list."""

    id: UUID
    transaction_id: UUID
    risk_score: int
    risk_level: str
    decision: str
    status: str
    analyst_id: UUID | None = None
    notes: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    transaction_summary: AlertTransactionSummary


class AlertListResponse(BaseModel):
    """Paginated GET /alerts response."""

    items: list[AlertListItem]
    total: int
    page: int
    per_page: int


# ---------------------------------------------------------------------------
# Alert detail
# ---------------------------------------------------------------------------


class AlertDetailResponse(BaseModel):
    """GET /alerts/{id} response."""

    id: UUID
    transaction_id: UUID
    risk_score: int
    risk_level: str
    decision: str
    explanation: ExplanationDetail | None = None
    risk_factors: list[str] = []
    status: str
    analyst_id: UUID | None = None
    notes: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    transaction: AlertTransactionDetail


# ---------------------------------------------------------------------------
# Alert update (PATCH)
# ---------------------------------------------------------------------------


_VALID_STATUSES = {"OPEN", "IN_REVIEW", "RESOLVED", "DISMISSED"}
_TERMINAL = {"RESOLVED", "DISMISSED"}
_TRANSITIONS: dict[str, set[str]] = {
    "OPEN": {"IN_REVIEW", "RESOLVED", "DISMISSED"},
    "IN_REVIEW": {"RESOLVED", "DISMISSED"},
    "RESOLVED": set(),
    "DISMISSED": set(),
}


class AlertUpdateRequest(BaseModel):
    """PATCH /alerts/{id} request body.

    At least one of ``status`` or ``notes`` must be provided.
    """

    status: str | None = None
    notes: str | None = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}")
        return v

    @staticmethod
    def validate_transition(current_status: str, new_status: str) -> None:
        """Raise ``ValueError`` if the transition is not allowed."""
        allowed = _TRANSITIONS.get(current_status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {current_status} to {new_status}"
            )


class AlertUpdateResponse(BaseModel):
    """PATCH /alerts/{id} response."""

    id: UUID
    transaction_id: UUID
    risk_score: int
    risk_level: str
    decision: str
    status: str
    analyst_id: UUID | None = None
    notes: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
