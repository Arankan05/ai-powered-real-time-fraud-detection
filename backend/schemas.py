"""Pydantic schemas for the backend ↔ ML/Fraud Intelligence integration.

These models define the data contracts between:
  - The backend transaction endpoint (request/response)
  - The ML/Fraud Intelligence Service HTTP interface

Schemas follow ``docs/api-contract.md`` and ``docs/ml-architecture.md``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ── Transaction request ───────────────────────────────────────────────


class TransactionCreate(BaseModel):
    """Raw transaction submission from the frontend (customer)."""

    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3)
    merchant_name: str = Field(..., min_length=1, max_length=255)
    merchant_category: str = Field(..., max_length=10)
    transaction_type: str = Field(..., pattern=r"^(purchase|transfer|withdrawal)$")
    location_country: str = Field(..., min_length=1, max_length=100)
    location_city: str = Field(..., max_length=100)
    device_fingerprint: str = Field(..., min_length=1, max_length=255)
    device_type: str = Field(..., pattern=r"^(mobile|desktop|pos)$")
    ip_address: str = Field(..., min_length=7, max_length=45)

    # Step 44: optional client-supplied idempotency key for duplicate
    # prevention.  Scoped to the authenticated customer; the server
    # controls the customer_id binding.
    idempotency_key: str | None = Field(
        None,
        max_length=255,
        description="Client idempotency key (prevents duplicate submissions)",
    )


# ── ML / Fraud Intelligence Service response ──────────────────────────


class MLFactor(BaseModel):
    """Single SHAP feature attribution from the ML model."""

    feature: str
    importance: float


class MLBehaviourSignal(BaseModel):
    """Single behavioural anomaly signal."""

    signal: str
    severity: float
    reason: str | None = None


class MLRuleTrigger(BaseModel):
    """Single triggered risk rule."""

    rule: str
    contribution: int
    reason: str | None = None


class MLExplanation(BaseModel):
    """Composite explanation from the ML/Fraud Intelligence Service."""

    ml_top_factors: list[MLFactor] = []
    behaviour_signals: list[MLBehaviourSignal] = []
    rules_triggered: list[MLRuleTrigger] = []


class MLPredictionResponse(BaseModel):
    """Response from the ML/Fraud Intelligence Service ``POST /predict``.

    Matches the ML service ``PredictionResponse`` schema defined in
    ``docs/ml-architecture.md`` and ``ml/api/app.py``.
    All fields are optional to support partial ML service versions.
    """

    fraud_probability: float | None = None
    fraud_prediction: int | None = None
    threshold: float | None = None
    model_version: str | None = None
    timestamp: int | None = None
    explanation: list[MLFactor] | None = None
    ml_score: int | None = None
    behaviour_score: int | None = None
    rule_score: int | None = None
    risk_score: int | None = None
    risk_level: str | None = None
    decision: str | None = None
    risk_factors: list[str] | None = None
    explanation_detail: MLExplanation | None = None

    model_config = ConfigDict(extra="allow")


# ── Alert schemas ────────────────────────────────────────────────────


class AlertSummary(BaseModel):
    """Compact alert reference embedded in TransactionResponse."""

    id: str
    status: str
    created_at: str


class TransactionSummary(BaseModel):
    """Compact transaction summary embedded in alert list responses."""

    amount: float | None = None
    currency: str | None = None
    merchant_name: str | None = None
    transaction_type: str | None = None
    timestamp: int | None = None


class AlertResponse(BaseModel):
    """Full alert response for GET /api/v1/alerts endpoints."""

    id: str
    transaction_id: str
    customer_id: str | None = None
    risk_score: int
    risk_level: str
    decision: str
    status: str
    analyst_id: str | None = None
    notes: str | None = None
    created_at: str
    updated_at: str | None = None
    resolved_at: str | None = None
    # Optional detail fields (present in detail endpoint)
    fraud_probability: float | None = None
    model_version: str | None = None
    risk_factors: list[str] | None = None
    explanation: MLExplanation | None = None
    transaction_summary: TransactionSummary | None = None


class AlertUpdate(BaseModel):
    """Request body for PATCH /api/v1/alerts/{id}."""

    status: str | None = Field(
        None,
        description="New status: IN_REVIEW, RESOLVED, or DISMISSED",
    )
    notes: str | None = Field(
        None,
        description="Analyst notes",
    )


class AlertListResponse(BaseModel):
    """Paginated list of alerts."""

    items: list[AlertResponse]
    total: int
    page: int
    per_page: int


# ── Transaction response ──────────────────────────────────────────────


class TransactionResponse(BaseModel):
    """Full transaction response including fraud scoring results.

    Matches ``docs/api-contract.md`` POST /api/v1/transactions response.
    """

    # Step 44: server-generated transaction identifier
    transaction_id: str

    amount: float
    currency: str
    merchant_name: str
    merchant_category: str
    transaction_type: str
    location_country: str
    location_city: str
    device_fingerprint: str
    device_type: str
    ip_address: str

    # Server-derived customer identity (Step 41)
    customer_id: str | None = None

    # Fraud scoring results from ML service
    ml_score: int | None = None
    behaviour_score: int | None = None
    rule_score: int | None = None
    risk_score: int | None = None
    risk_level: str | None = None
    decision: str | None = None
    explanation: MLExplanation | None = None
    risk_factors: list[str] | None = None
    model_version: str | None = None
    fraud_probability: float | None = None
    fraud_prediction: int | None = None
    timestamp: int | None = None

    # Alert reference (populated when decision == HOLD)
    alert: AlertSummary | None = None

    # Step 44: idempotency metadata
    idempotent: bool = False

    # Step 44: ML failure indicator — present when ML service was
    # unavailable or returned an error.  When ``True``, no ML
    # prediction fields are populated and no fraud decision was made.
    ml_failure: bool = False


# ── Outcome feedback ──────────────────────────────────────────────────


class OutcomeUpdate(BaseModel):
    """Request to update a transaction's fraud outcome (label feedback loop)."""

    customer_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Customer identifier",
    )
    timestamp: int = Field(
        ..., ge=0,
        description="Transaction timestamp (from prediction response)",
    )
    is_fraud: int = Field(
        ..., ge=0, le=1,
        description="Confirmed fraud label (0=legitimate, 1=fraudulent)",
    )


class OutcomeResponse(BaseModel):
    """Response from the outcome update endpoint."""

    updated: bool = Field(..., description="Whether the record was found and updated")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: int = Field(..., description="Transaction timestamp that was updated")
    is_fraud: int = Field(..., description="New fraud label value")


# ── Authentication (Step 39) ──────────────────────────────────────


# Contract: min 8, max 128 chars, at least 1 uppercase, 1 lowercase, 1 digit
# (enforced in password_complexity below — Pydantic's regex engine does
# not support look-around assertions.)


class RegisterRequest(BaseModel):
    """``POST /api/v1/auth/register`` request (contract §Authentication)."""

    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: str | None = Field(None, max_length=30)
    date_of_birth: str | None = Field(None, max_length=10)
    address: str | None = Field(None, max_length=255)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, value: str) -> str:
        """Contract: at least 1 uppercase, 1 lowercase, and 1 digit."""
        if not any(c.islower() for c in value):
            raise ValueError("password must contain at least one lowercase letter")
        if not any(c.isupper() for c in value):
            raise ValueError("password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in value):
            raise ValueError("password must contain at least one digit")
        return value


class UserResponse(BaseModel):
    """Registration response (201) — never includes the password."""

    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    customer_id: str | None = None


class LoginRequest(BaseModel):
    """``POST /api/v1/auth/login`` request."""

    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """JWT token pair returned by login and refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token lifetime (seconds)")


class RefreshRequest(BaseModel):
    """``POST /api/v1/auth/refresh`` request."""

    refresh_token: str = Field(..., min_length=1)


class MeResponse(BaseModel):
    """``GET /api/v1/auth/me`` response."""

    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    customer_id: str | None = None
    is_active: bool
    created_at: str
