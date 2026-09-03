"""Pydantic schemas for the analytics dashboard endpoint.

Response shape matches ``docs/api-contract.md`` exactly.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RiskFactorCount(BaseModel):
    """Single entry in the ``top_risk_factors`` list."""

    factor: str
    count: int


class TransactionsDay(BaseModel):
    """Single day entry in the ``transactions_over_time`` list."""

    date: str
    total: int
    flagged: int


class RiskDistribution(BaseModel):
    """Breakdown of transactions by risk level."""

    LOW: int = 0
    MEDIUM: int = 0
    HIGH: int = 0


# ---------------------------------------------------------------------------
# Dashboard response
# ---------------------------------------------------------------------------


class DashboardResponse(BaseModel):
    """GET /api/v1/analytics/dashboard response (200)."""

    from_date: datetime
    to_date: datetime
    total_transactions: int
    flagged_transactions: int
    alerts_open: int
    alerts_resolved: int
    risk_distribution: RiskDistribution
    top_risk_factors: list[RiskFactorCount] = []
    transactions_over_time: list[TransactionsDay] = []
