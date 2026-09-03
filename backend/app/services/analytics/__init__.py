"""Analytics service — read-only aggregation for the fraud analyst dashboard.

Provides ``GET /api/v1/analytics/dashboard`` data by running efficient SQL
aggregation queries against the existing ``transactions`` and ``alerts`` tables.

No data is written.  The ML service is never called from this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.core.errors import AppException
from app.models.alert import Alert
from app.models.transaction import Transaction
from app.schemas.analytics import (
    DashboardResponse,
    RiskDistribution,
    RiskFactorCount,
    TransactionsDay,
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Read-only aggregation service for the analytics dashboard."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dashboard(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> DashboardResponse:
        """Return aggregated dashboard metrics for the given date range."""

        now = datetime.now(timezone.utc)

        if from_date is None:
            from_date = now - timedelta(days=30)
        if to_date is None:
            to_date = now

        # Ensure timezone-aware datetimes (treat naive as UTC).
        if from_date.tzinfo is None:
            from_date = from_date.replace(tzinfo=timezone.utc)
        if to_date.tzinfo is None:
            to_date = to_date.replace(tzinfo=timezone.utc)

        if from_date > to_date:
            raise AppException(
                status_code=422,
                detail="from_date must not be after to_date",
                error_code="VALIDATION_ERROR",
            )

        total = self._total_transactions(from_date, to_date)
        flagged = self._flagged_transactions(from_date, to_date)
        alerts_open = self._count_alerts("OPEN")
        alerts_resolved = self._count_alerts("RESOLVED")
        risk_dist = self._risk_distribution(from_date, to_date)
        top_factors = self._top_risk_factors(from_date, to_date)
        over_time = self._transactions_over_time(from_date, to_date)

        return DashboardResponse(
            from_date=from_date,
            to_date=to_date,
            total_transactions=total,
            flagged_transactions=flagged,
            alerts_open=alerts_open,
            alerts_resolved=alerts_resolved,
            risk_distribution=RiskDistribution(
                LOW=risk_dist.get("LOW", 0),
                MEDIUM=risk_dist.get("MEDIUM", 0),
                HIGH=risk_dist.get("HIGH", 0),
            ),
            top_risk_factors=top_factors,
            transactions_over_time=over_time,
        )

    # ------------------------------------------------------------------
    # Transaction aggregations
    # ------------------------------------------------------------------

    def _total_transactions(
        self, from_date: datetime, to_date: datetime
    ) -> int:
        """Count all transactions within the date range."""
        result = (
            self._db.query(func.count(Transaction.id))
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
            )
            .scalar()
        )
        return result or 0

    def _flagged_transactions(
        self, from_date: datetime, to_date: datetime
    ) -> int:
        """Count transactions with risk_level MEDIUM or HIGH.

        A transaction is considered *flagged* when its risk analysis
        resulted in a MEDIUM or HIGH risk level (i.e. not LOW / not
        un-analysed).
        """
        result = (
            self._db.query(func.count(Transaction.id))
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
                Transaction.risk_level.in_(["MEDIUM", "HIGH"]),
            )
            .scalar()
        )
        return result or 0

    # ------------------------------------------------------------------
    # Alert aggregations
    # ------------------------------------------------------------------

    def _count_alerts(self, status: str) -> int:
        """Count alerts with the given status (not date-bounded)."""
        result = (
            self._db.query(func.count(Alert.id))
            .filter(Alert.status == status)
            .scalar()
        )
        return result or 0

    # ------------------------------------------------------------------
    # Risk distribution
    # ------------------------------------------------------------------

    def _risk_distribution(
        self, from_date: datetime, to_date: datetime
    ) -> dict[str, int]:
        """Return ``{risk_level: count}`` for transactions in range."""
        rows = (
            self._db.query(
                Transaction.risk_level,
                func.count(Transaction.id),
            )
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
            )
            .group_by(Transaction.risk_level)
            .all()
        )
        dist: dict[str, int] = {}
        for risk_level, count in rows:
            if risk_level is not None:
                dist[risk_level] = count
        return dist

    # ------------------------------------------------------------------
    # Top risk factors
    # ------------------------------------------------------------------

    def _top_risk_factors(
        self, from_date: datetime, to_date: datetime
    ) -> list[RiskFactorCount]:
        """Extract and count risk factors from ``explanation_json``.

        Factors are collected from three sub-keys inside the JSONB column:

        * ``ml_top_factors``  → ``feature``
        * ``behaviour_signals`` → ``signal``
        * ``rules_triggered``   → ``rule``

        Results are sorted by count descending, then alphabetically for
        deterministic ordering.
        """
        explanation_rows = (
            self._db.query(Transaction.explanation_json)
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
                Transaction.explanation_json.isnot(None),
            )
            .all()
        )

        factor_counts: dict[str, int] = defaultdict(int)
        for (explanation,) in explanation_rows:
            if not isinstance(explanation, dict):
                continue

            for factor in explanation.get("ml_top_factors", []):
                name = factor.get("feature")
                if name:
                    factor_counts[name] += 1

            for signal in explanation.get("behaviour_signals", []):
                name = signal.get("signal")
                if name:
                    factor_counts[name] += 1

            for rule in explanation.get("rules_triggered", []):
                name = rule.get("rule")
                if name:
                    factor_counts[name] += 1

        # Sort: highest count first, then alphabetically for ties.
        sorted_factors = sorted(
            factor_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RiskFactorCount(factor=name, count=count)
            for name, count in sorted_factors
        ]

    # ------------------------------------------------------------------
    # Transactions over time
    # ------------------------------------------------------------------

    def _transactions_over_time(
        self, from_date: datetime, to_date: datetime
    ) -> list[TransactionsDay]:
        """Return daily total and flagged transaction counts.

        Uses ``func.date()`` (PostgreSQL ``DATE`` cast) for efficient
        SQL-side grouping.  Days with no transactions are filled with
        zero counts so the frontend receives a continuous time series.
        """
        date_col = func.date(Transaction.timestamp)
        is_flagged = Transaction.risk_level.in_(["MEDIUM", "HIGH"])

        # Total per day.
        total_rows = (
            self._db.query(date_col, func.count(Transaction.id))
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
            )
            .group_by(date_col)
            .all()
        )

        # Flagged per day.
        flagged_rows = (
            self._db.query(date_col, func.count(Transaction.id))
            .filter(
                Transaction.timestamp >= from_date,
                Transaction.timestamp <= to_date,
                is_flagged,
            )
            .group_by(date_col)
            .all()
        )

        # Build lookup dicts keyed by 'YYYY-MM-DD' string.
        total_by_day: dict[str, int] = {str(d): c for d, c in total_rows}
        flagged_by_day: dict[str, int] = {str(d): c for d, c in flagged_rows}

        # Fill every calendar day in the range (zero where absent).
        start_day = from_date.date()
        end_day = to_date.date()
        result: list[TransactionsDay] = []
        current = start_day
        while current <= end_day:
            day_str = current.isoformat()
            result.append(
                TransactionsDay(
                    date=day_str,
                    total=total_by_day.get(day_str, 0),
                    flagged=flagged_by_day.get(day_str, 0),
                )
            )
            current += timedelta(days=1)

        return result
