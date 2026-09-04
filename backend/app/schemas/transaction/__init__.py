"""Pydantic schemas for transaction and fraud check endpoints.

Request/response shapes match ``docs/api-contract.md`` exactly.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer, field_validator


# ---------------------------------------------------------------------------
# Shared explanation sub-models
# ---------------------------------------------------------------------------


class MLFactor(BaseModel):
    feature: str
    importance: float


class BehaviourSignal(BaseModel):
    signal: str
    severity: float


class RuleTrigger(BaseModel):
    rule: str
    contribution: int


class ExplanationResponse(BaseModel):
    ml_top_factors: list[MLFactor] = []
    behaviour_signals: list[BehaviourSignal] = []
    rules_triggered: list[RuleTrigger] = []


class AlertSummary(BaseModel):
    id: UUID
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Transaction request / response
# ---------------------------------------------------------------------------

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class TransactionCreateRequest(BaseModel):
    """POST /api/v1/transactions request body."""

    amount: Decimal = Field(..., gt=0, max_digits=12, decimal_places=2,
                            description="Positive decimal, max 9999999999.99, min 0.01")
    currency: str = Field("USD", max_length=3, min_length=3)
    merchant_name: str = Field(..., min_length=1, max_length=255)
    merchant_category: str | None = Field(None, max_length=10)
    transaction_type: str = Field(...)
    location_country: str | None = Field(None, max_length=100)
    location_city: str | None = Field(None, max_length=100)
    device_fingerprint: str | None = Field(None, max_length=255)
    device_type: str | None = None
    ip_address: str | None = Field(None, max_length=45)

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, v: str) -> str:
        v = v.upper()
        if not _CURRENCY_RE.match(v):
            raise ValueError("Currency must be ISO 4217 3-letter code")
        return v

    @field_validator("transaction_type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {"purchase", "transfer", "withdrawal"}
        if v not in allowed:
            raise ValueError(f"transaction_type must be one of {allowed}")
        return v

    @field_validator("device_type")
    @classmethod
    def _validate_device_type(cls, v: str | None) -> str | None:
        if v is not None and v not in {"mobile", "desktop", "pos"}:
            raise ValueError("device_type must be mobile, desktop, or pos")
        return v

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: Decimal) -> Decimal:
        if v < Decimal("0.01"):
            raise ValueError("Amount must be at least 0.01")
        if v > Decimal("9999999999.99"):
            raise ValueError("Amount must not exceed 9999999999.99")
        return v


class TransactionDetailResponse(BaseModel):
    """Full transaction response (POST /transactions 201, GET /transactions/{id} 200)."""

    id: UUID
    customer_id: UUID
    merchant_id: UUID | None = None
    amount: Decimal
    currency: str
    merchant_name: str
    merchant_category: str | None = None
    transaction_type: str
    location_country: str | None = None
    location_city: str | None = None
    device_fingerprint: str | None = None
    device_type: str | None = None
    ip_address: str | None = None
    timestamp: datetime
    status: str
    ml_score: int | None = None
    behaviour_score: int | None = None
    rule_score: int | None = None
    risk_score: int | None = None
    risk_level: str | None = None
    decision: str | None = None
    explanation: ExplanationResponse | None = None
    risk_factors: list[str] = []
    model_version: str | None = None
    alert: AlertSummary | None = None

    @field_serializer("amount")
    def _serialize_amount(self, v: Decimal) -> float:
        return float(v)


class TransactionSummaryResponse(BaseModel):
    """Item in paginated GET /transactions list."""

    id: UUID
    customer_id: UUID
    merchant_name: str
    amount: Decimal
    currency: str
    transaction_type: str
    timestamp: datetime
    status: str
    risk_score: int | None = None
    risk_level: str | None = None
    decision: str | None = None

    @field_serializer("amount")
    def _serialize_amount(self, v: Decimal) -> float:
        return float(v)


class TransactionListResponse(BaseModel):
    """Paginated GET /transactions response."""

    items: list[TransactionSummaryResponse]
    total: int
    page: int
    per_page: int


# ---------------------------------------------------------------------------
# Transaction query parameters
# ---------------------------------------------------------------------------


class TransactionQueryParams(BaseModel):
    """Query parameters for GET /transactions and GET /customers/{id}/transactions."""

    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)
    status: str | None = None
    risk_level: str | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str | None) -> str | None:
        if v is not None and v not in {"PENDING", "COMPLETED", "FAILED"}:
            raise ValueError("status must be PENDING, COMPLETED, or FAILED")
        return v

    @field_validator("risk_level")
    @classmethod
    def _validate_risk_level(cls, v: str | None) -> str | None:
        if v is not None and v not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("risk_level must be LOW, MEDIUM, or HIGH")
        return v


# ---------------------------------------------------------------------------
# Fraud check request/response
# ---------------------------------------------------------------------------


class FraudCheckRequest(BaseModel):
    """POST /api/v1/fraud/check request body."""

    customer_id: UUID
    amount: Decimal = Field(..., gt=0, max_digits=12, decimal_places=2)
    currency: str = Field("USD", max_length=3, min_length=3)
    merchant_name: str = Field(..., min_length=1, max_length=255)
    merchant_category: str | None = Field(None, max_length=10)
    transaction_type: str = Field(...)
    location_country: str | None = Field(None, max_length=100)
    location_city: str | None = Field(None, max_length=100)
    device_fingerprint: str | None = Field(None, max_length=255)
    device_type: str | None = None
    ip_address: str | None = Field(None, max_length=45)

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, v: str) -> str:
        v = v.upper()
        if not _CURRENCY_RE.match(v):
            raise ValueError("Currency must be ISO 4217 3-letter code")
        return v

    @field_validator("transaction_type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {"purchase", "transfer", "withdrawal"}
        if v not in allowed:
            raise ValueError(f"transaction_type must be one of {allowed}")
        return v

    @field_validator("device_type")
    @classmethod
    def _validate_device_type(cls, v: str | None) -> str | None:
        if v is not None and v not in {"mobile", "desktop", "pos"}:
            raise ValueError("device_type must be mobile, desktop, or pos")
        return v

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: Decimal) -> Decimal:
        if v < Decimal("0.01"):
            raise ValueError("Amount must be at least 0.01")
        if v > Decimal("9999999999.99"):
            raise ValueError("Amount must not exceed 9999999999.99")
        return v


class FraudCheckResponse(BaseModel):
    """POST /api/v1/fraud/check response (200)."""

    ml_score: int
    behaviour_score: int
    rule_score: int
    risk_score: int
    risk_level: str
    decision: str
    explanation: ExplanationResponse
    risk_factors: list[str] = []
    model_version: str | None = None
