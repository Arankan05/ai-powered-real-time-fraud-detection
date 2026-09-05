"""Analytics router — ``GET /api/v1/analytics/dashboard``.

Provides aggregated metrics for the fraud analyst dashboard matching
``docs/api-contract.md`` §Analytics Endpoint.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from psycopg.rows import dict_row

from backend.db.alert_repository import AlertRepository
from backend.db.transaction_repository import TransactionRepository
from backend.schemas import (
    DashboardResponse,
    RiskDistribution,
    RiskFactorCount,
    TransactionsDay,
)
from backend.security.deps import require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_require_analyst = require_roles("fraud_analyst", "admin")

_pg_pool: Any | None = None
_transaction_repo: TransactionRepository | None = None
_alert_repo: AlertRepository | None = None


def set_postgres_pool(pool: Any) -> None:
    """Set the PostgreSQL connection pool (called during app startup)."""
    global _pg_pool
    _pg_pool = pool


def set_transaction_repository(repo: TransactionRepository) -> None:
    """Set the transaction repository (called during app startup)."""
    global _transaction_repo
    _transaction_repo = repo


def set_alert_repository(repo: AlertRepository) -> None:
    """Set the alert repository (called during app startup)."""
    global _alert_repo
    _alert_repo = repo


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
)
def get_dashboard(
    from_date: datetime | None = Query(None),
    to_date: datetime | None = Query(None),
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> DashboardResponse:
    """Return aggregated analytics for the fraud analyst dashboard.

    * **fraud_analyst / admin**: full access.
    * Date range defaults to the last 30 days if unspecified.
    """
    now = datetime.now(timezone.utc)

    if to_date is None:
        to_date = now
    if from_date is None:
        from_date = to_date - timedelta(days=30)

    if from_date.tzinfo is None:
        from_date = from_date.replace(tzinfo=timezone.utc)
    if to_date.tzinfo is None:
        to_date = to_date.replace(tzinfo=timezone.utc)

    if from_date > to_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from_date must not be after to_date",
        )

    if _pg_pool is not None:
        return _get_dashboard_postgres(from_date, to_date)
    else:
        return _get_dashboard_fallback(from_date, to_date)


def _get_dashboard_postgres(from_date: datetime, to_date: datetime) -> DashboardResponse:
    with _pg_pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # 1. Total transactions
            cur.execute(
                "SELECT COUNT(*) as cnt FROM transactions WHERE timestamp >= %s AND timestamp <= %s",
                (from_date, to_date),
            )
            total_tx = cur.fetchone()["cnt"]

            # 2. Flagged transactions (MEDIUM or HIGH)
            cur.execute(
                """SELECT COUNT(*) as cnt FROM transactions
                   WHERE timestamp >= %s AND timestamp <= %s AND risk_level IN ('MEDIUM', 'HIGH')""",
                (from_date, to_date),
            )
            flagged_tx = cur.fetchone()["cnt"]

            # 3. Alerts open / resolved
            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'OPEN'")
            alerts_open = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'RESOLVED'")
            alerts_resolved = cur.fetchone()["cnt"]

            # 4. Risk distribution
            cur.execute(
                "SELECT risk_level, COUNT(*) as cnt FROM transactions WHERE timestamp >= %s AND timestamp <= %s GROUP BY risk_level",
                (from_date, to_date),
            )
            risk_dist_rows = cur.fetchall()
            risk_dist_map = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for r in risk_dist_rows:
                if r["risk_level"] in risk_dist_map:
                    risk_dist_map[r["risk_level"]] = r["cnt"]

            # 5. Top risk factors
            cur.execute(
                "SELECT explanation_json FROM transactions WHERE timestamp >= %s AND timestamp <= %s AND explanation_json IS NOT NULL",
                (from_date, to_date),
            )
            expl_rows = cur.fetchall()
            factor_counts: dict[str, int] = defaultdict(int)
            for row in expl_rows:
                expl = row["explanation_json"]
                if isinstance(expl, str):
                    try:
                        expl = json.loads(expl)
                    except Exception:
                        continue
                if isinstance(expl, dict):
                    for f in expl.get("ml_top_factors", []):
                        if isinstance(f, dict) and f.get("feature"):
                            factor_counts[f["feature"]] += 1
                    for s in expl.get("behaviour_signals", []):
                        if isinstance(s, dict) and s.get("signal"):
                            factor_counts[s["signal"]] += 1
                    for r in expl.get("rules_triggered", []):
                        if isinstance(r, dict) and r.get("rule"):
                            factor_counts[r["rule"]] += 1

            sorted_factors = sorted(factor_counts.items(), key=lambda x: (-x[1], x[0]))
            top_factors = [
                RiskFactorCount(factor=k, count=v) for k, v in sorted_factors
            ]

            # 6. Transactions over time
            cur.execute(
                """
                SELECT DATE(timestamp) as tx_date, COUNT(*) as total_cnt,
                       COUNT(*) FILTER (WHERE risk_level IN ('MEDIUM', 'HIGH')) as flagged_cnt
                FROM transactions
                WHERE timestamp >= %s AND timestamp <= %s
                GROUP BY DATE(timestamp)
                ORDER BY tx_date ASC
                """,
                (from_date, to_date),
            )
            over_time_rows = cur.fetchall()
            over_time_dict = {
                str(r["tx_date"]): (r["total_cnt"], r["flagged_cnt"])
                for r in over_time_rows
            }

            start_day = from_date.date()
            end_day = to_date.date()
            over_time: list[TransactionsDay] = []
            curr = start_day
            while curr <= end_day:
                d_str = curr.isoformat()
                tot, flg = over_time_dict.get(d_str, (0, 0))
                over_time.append(TransactionsDay(date=d_str, total=tot, flagged=flg))
                curr += timedelta(days=1)

    return DashboardResponse(
        from_date=from_date,
        to_date=to_date,
        total_transactions=total_tx,
        flagged_transactions=flagged_tx,
        alerts_open=alerts_open,
        alerts_resolved=alerts_resolved,
        risk_distribution=RiskDistribution(
            LOW=risk_dist_map["LOW"],
            MEDIUM=risk_dist_map["MEDIUM"],
            HIGH=risk_dist_map["HIGH"],
        ),
        top_risk_factors=top_factors,
        transactions_over_time=over_time,
    )


def _get_dashboard_fallback(from_date: datetime, to_date: datetime) -> DashboardResponse:
    # In-memory / SQLite fallback
    txs, total_count = [], 0
    if _transaction_repo is not None:
        txs, total_count = _transaction_repo.list_transactions(
            from_date=from_date, to_date=to_date, page=1, per_page=10000
        )

    alerts: list[dict[str, Any]] = []
    if _alert_repo is not None:
        alerts, _ = _alert_repo.list_alerts(page=1, per_page=10000)

    alerts_open = sum(1 for a in alerts if a.get("status") == "OPEN")
    alerts_resolved = sum(1 for a in alerts if a.get("status") == "RESOLVED")

    flagged_tx = 0
    risk_dist_map = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    factor_counts: dict[str, int] = defaultdict(int)
    over_time_dict: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for tx in txs:
        r_level = tx.get("risk_level", "LOW")
        if r_level in risk_dist_map:
            risk_dist_map[r_level] += 1
        if r_level in ("MEDIUM", "HIGH"):
            flagged_tx += 1

        tx_ts = tx.get("timestamp")
        if isinstance(tx_ts, str):
            try:
                tx_dt = datetime.fromisoformat(tx_ts)
                d_str = tx_dt.date().isoformat()
            except Exception:
                d_str = from_date.date().isoformat()
        elif isinstance(tx_ts, datetime):
            d_str = tx_ts.date().isoformat()
        else:
            d_str = from_date.date().isoformat()

        over_time_dict[d_str][0] += 1
        if r_level in ("MEDIUM", "HIGH"):
            over_time_dict[d_str][1] += 1

        expl = tx.get("explanation") or tx.get("explanation_json")
        if isinstance(expl, str):
            try:
                expl = json.loads(expl)
            except Exception:
                pass
        if isinstance(expl, dict):
            for f in expl.get("ml_top_factors", []):
                if isinstance(f, dict) and f.get("feature"):
                    factor_counts[f["feature"]] += 1
            for s in expl.get("behaviour_signals", []):
                if isinstance(s, dict) and s.get("signal"):
                    factor_counts[s["signal"]] += 1
            for r in expl.get("rules_triggered", []):
                if isinstance(r, dict) and r.get("rule"):
                    factor_counts[r["rule"]] += 1

    sorted_factors = sorted(factor_counts.items(), key=lambda x: (-x[1], x[0]))
    top_factors = [RiskFactorCount(factor=k, count=v) for k, v in sorted_factors]

    start_day = from_date.date()
    end_day = to_date.date()
    over_time: list[TransactionsDay] = []
    curr = start_day
    while curr <= end_day:
        d_str = curr.isoformat()
        tot, flg = over_time_dict[d_str]
        over_time.append(TransactionsDay(date=d_str, total=tot, flagged=flg))
        curr += timedelta(days=1)

    return DashboardResponse(
        from_date=from_date,
        to_date=to_date,
        total_transactions=total_count,
        flagged_transactions=flagged_tx,
        alerts_open=alerts_open,
        alerts_resolved=alerts_resolved,
        risk_distribution=RiskDistribution(
            LOW=risk_dist_map["LOW"],
            MEDIUM=risk_dist_map["MEDIUM"],
            HIGH=risk_dist_map["HIGH"],
        ),
        top_risk_factors=top_factors,
        transactions_over_time=over_time,
    )
