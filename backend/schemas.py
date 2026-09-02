"""Pydantic schemas for the backend ↔ ML/Fraud Intelligence integration.

These models define the data contracts between:
  - The backend transaction endpoint (request/response)
  - The ML/Fraud Intelligence Service HTTP interface

Schemas follow ``docs/api-contract.md`` and ``docs/ml-architecture.md``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
