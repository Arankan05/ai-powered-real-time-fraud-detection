"""Alert repository — SQLite, PostgreSQL, and in-memory implementations.

Provides persistent storage for fraud alerts.  Alerts are created
automatically when a transaction receives ``decision == HOLD`` and
persist across application restarts (SQLite) or live only in memory
(tests/fallback).

Status transitions
------------------
Valid transitions are enforced by :meth:`update_status`:

* ``OPEN`` → ``IN_REVIEW``, ``RESOLVED``, ``DISMISSED``
* ``IN_REVIEW`` → ``RESOLVED``, ``DISMISSED``
* ``RESOLVED`` / ``DISMISSED`` → terminal (no further transitions)

When status changes to ``RESOLVED`` or ``DISMISSED``, ``resolved_at``
is set automatically.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import psycopg
from psycopg import errors as pg_errors
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from backend.db.user_repository import _coerce_uuid


# ── Status constants and transitions ──────────────────────────────────

OPEN = "OPEN"
IN_REVIEW = "IN_REVIEW"
RESOLVED = "RESOLVED"
DISMISSED = "DISMISSED"

VALID_STATUSES = frozenset({OPEN, IN_REVIEW, RESOLVED, DISMISSED})
TERMINAL_STATUSES = frozenset({RESOLVED, DISMISSED})

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    OPEN: frozenset({IN_REVIEW, RESOLVED, DISMISSED}),
    IN_REVIEW: frozenset({RESOLVED, DISMISSED}),
    RESOLVED: frozenset(),
    DISMISSED: frozenset(),
}


def is_valid_transition(current: str, target: str) -> bool:
    """Return True if the status transition is allowed."""
    return target in _VALID_TRANSITIONS.get(current, frozenset())


# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class AlertRepository(Protocol):
    """Abstract contract for alert persistence."""

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new alert. Returns the full alert dict."""
        ...

    def get_by_id(self, alert_id: str) -> dict[str, Any] | None:
        """Return a single alert by ID, or None."""
        ...

    def get_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        """Return the alert for a transaction, or None."""
        ...

    def list_alerts(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (alerts, total_count) with optional filtering."""
        ...

    def update_status(
        self,
        alert_id: str,
        *,
        new_status: str,
        notes: str | None = None,
        analyst_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Update alert status/notes. Returns updated alert or None."""
        ...


# ── In-memory implementation ─────────────────────────────────────────


class InMemoryAlertStore:
    """Volatile in-memory alert store (for tests and fallback)."""

    def __init__(self) -> None:
        self._alerts: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> dict[str, Any]:
        alert_id = kwargs.get("id") or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        alert = {
            "id": alert_id,
            "transaction_id": kwargs.get("transaction_id", str(uuid.uuid4())),
            "customer_id": kwargs.get("customer_id"),
            "amount": kwargs.get("amount"),
            "currency": kwargs.get("currency"),
            "merchant_name": kwargs.get("merchant_name"),
            "transaction_type": kwargs.get("transaction_type"),
            "timestamp": kwargs.get("timestamp"),
            "risk_score": kwargs["risk_score"],
            "risk_level": kwargs["risk_level"],
            "decision": kwargs["decision"],
            "fraud_probability": kwargs.get("fraud_probability"),
            "model_version": kwargs.get("model_version"),
            "risk_factors": kwargs.get("risk_factors"),
            "explanation_json": kwargs.get("explanation_json"),
            "status": OPEN,
            "analyst_id": None,
            "notes": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
        }
        with self._lock:
            self._alerts[alert_id] = alert
        return dict(alert)

    def get_by_id(self, alert_id: str) -> dict[str, Any] | None:
        with self._lock:
            alert = self._alerts.get(alert_id)
        return dict(alert) if alert else None

    def get_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            for alert in self._alerts.values():
                if alert["transaction_id"] == transaction_id:
                    return dict(alert)
        return None

    def list_alerts(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            results = list(self._alerts.values())

        if status:
            results = [a for a in results if a["status"] == status]
        if risk_level:
            results = [a for a in results if a["risk_level"] == risk_level]

        # Sort by created_at descending (newest first)
        results.sort(key=lambda a: a["created_at"], reverse=True)
        total = len(results)

        start = (page - 1) * per_page
        end = start + per_page
        return [dict(a) for a in results[start:end]], total

    def update_status(
        self,
        alert_id: str,
        *,
        new_status: str,
        notes: str | None = None,
        analyst_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            # Allow same-status updates (for notes-only changes)
            if new_status != alert["status"] and not is_valid_transition(alert["status"], new_status):
                return None  # caller should check return
            alert["status"] = new_status
            if notes is not None:
                alert["notes"] = notes
            if analyst_id is not None and alert["analyst_id"] is None:
                alert["analyst_id"] = analyst_id
            if new_status in TERMINAL_STATUSES:
                alert["resolved_at"] = datetime.now(timezone.utc).isoformat()
            alert["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(alert)


# ── SQLite implementation ─────────────────────────────────────────────


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS alerts (
    id               TEXT PRIMARY KEY,
    transaction_id   TEXT NOT NULL,
    customer_id      TEXT,
    amount           REAL,
    currency         TEXT,
    merchant_name    TEXT,
    transaction_type TEXT,
    timestamp        INTEGER,
    risk_score       INTEGER NOT NULL,
    risk_level       TEXT NOT NULL DEFAULT 'HIGH',
    decision         TEXT NOT NULL DEFAULT 'HOLD',
    fraud_probability REAL,
    model_version    TEXT,
    risk_factors     TEXT,
    explanation_json TEXT,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    analyst_id       TEXT,
    notes            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    resolved_at      TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_alerts_transaction_id ON alerts(transaction_id);",
    "CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status);",
    "CREATE INDEX IF NOT EXISTS ix_alerts_risk_level ON alerts(risk_level);",
]


class SQLiteAlertRepository:
    """Persistent SQLite-backed alert repository.

    Uses WAL journal mode for concurrent read/write safety,
    following the same pattern as ``ml/features/history.py``.
    """

    def __init__(self, db_path: str | Path = "data/alerts.db") -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            self._conn.execute(idx_sql)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def create(self, **kwargs: Any) -> dict[str, Any]:
        alert_id = kwargs.get("id") or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        risk_factors = kwargs.get("risk_factors")
        explanation_json = kwargs.get("explanation_json")

        row = {
            "id": alert_id,
            "transaction_id": kwargs.get("transaction_id", str(uuid.uuid4())),
            "customer_id": kwargs.get("customer_id"),
            "amount": kwargs.get("amount"),
            "currency": kwargs.get("currency"),
            "merchant_name": kwargs.get("merchant_name"),
            "transaction_type": kwargs.get("transaction_type"),
            "timestamp": kwargs.get("timestamp"),
            "risk_score": kwargs["risk_score"],
            "risk_level": kwargs["risk_level"],
            "decision": kwargs["decision"],
            "fraud_probability": kwargs.get("fraud_probability"),
            "model_version": kwargs.get("model_version"),
            "risk_factors": json.dumps(risk_factors) if risk_factors else None,
            "explanation_json": json.dumps(explanation_json) if explanation_json else None,
            "status": OPEN,
            "analyst_id": None,
            "notes": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
        }

        with self._lock:
            self._conn.execute(
                """\
                INSERT INTO alerts (
                    id, transaction_id, customer_id, amount, currency,
                    merchant_name, transaction_type, timestamp, risk_score,
                    risk_level, decision, fraud_probability, model_version,
                    risk_factors, explanation_json, status, analyst_id, notes,
                    created_at, updated_at, resolved_at
                ) VALUES (
                    :id, :transaction_id, :customer_id, :amount, :currency,
                    :merchant_name, :transaction_type, :timestamp, :risk_score,
                    :risk_level, :decision, :fraud_probability, :model_version,
                    :risk_factors, :explanation_json, :status, :analyst_id, :notes,
                    :created_at, :updated_at, :resolved_at
                )""",
                row,
            )
            self._conn.commit()

        return self._deserialise(row)

    def get_by_id(self, alert_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            )
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE transaction_id = ?",
                (transaction_id,),
            )
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_alerts(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        if risk_level:
            clauses.append("risk_level = ?")
            params.append(risk_level)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM alerts {where}", params
            ).fetchone()[0]

            offset = (page - 1) * per_page
            cur = self._conn.execute(
                f"SELECT * FROM alerts {where} ORDER BY created_at DESC "
                f"LIMIT ? OFFSET ?",
                [*params, per_page, offset],
            )
            rows = cur.fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def update_status(
        self,
        alert_id: str,
        *,
        new_status: str,
        notes: str | None = None,
        analyst_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None

            current_status = row["status"]
            # Allow same-status updates (for notes-only changes)
            if new_status != current_status and not is_valid_transition(current_status, new_status):
                return None

            now = datetime.now(timezone.utc).isoformat()
            resolved_at = now if new_status in TERMINAL_STATUSES else row["resolved_at"]
            updated_analyst = row["analyst_id"]
            if analyst_id is not None and updated_analyst is None:
                updated_analyst = analyst_id
            updated_notes = notes if notes is not None else row["notes"]

            self._conn.execute(
                """\
                UPDATE alerts SET
                    status = ?, notes = ?, analyst_id = ?,
                    updated_at = ?, resolved_at = ?
                WHERE id = ?""",
                (new_status, updated_notes, updated_analyst, now, resolved_at, alert_id),
            )
            self._conn.commit()

            # Re-fetch the updated row
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            )
            updated_row = cur.fetchone()

        return self._row_to_dict(updated_row) if updated_row else None

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        return SQLiteAlertRepository._deserialise(d)

    @staticmethod
    def _deserialise(d: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON-encoded fields back to Python objects."""
        result = dict(d)
        for field in ("risk_factors", "explanation_json"):
            val = result.get(field)
            if isinstance(val, str):
                try:
                    result[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return result


# ── PostgreSQL implementation ─────────────────────────────────────────


class PostgresAlertRepository:
    """PostgreSQL-backed alert repository (production persistence).

    Shares a :class:`psycopg_pool.ConnectionPool` with the user
    repository.  ``risk_factors`` and ``explanation_json`` are stored
    as ``JSONB`` (round-tripped transparently by :mod:`psycopg`).

    Constructing a repository automatically ensures the backing schema
    exists via :func:`backend.db.postgres.init_schema` — idempotent.

    A ``UNIQUE`` index on ``transaction_id`` (``uq_alerts_transaction_id``)
    is the database-level guard against duplicate alerts per transaction;
    the router-level ``get_by_transaction_id`` pre-check remains the
    primary path (same as the SQLite implementation).
    """

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        from backend.db.postgres import init_schema  # local import avoids cycle
        init_schema(pool)

    # ── AlertRepository Protocol ────────────────────────────────────────

    def create(self, **kwargs: Any) -> dict[str, Any]:
        alert_id = kwargs.get("id") or str(uuid.uuid4())
        transaction_id = kwargs.get("transaction_id") or str(uuid.uuid4())
        customer_id = kwargs.get("customer_id")
        now = datetime.now(timezone.utc).isoformat()

        risk_factors = kwargs.get("risk_factors")
        explanation_json = kwargs.get("explanation_json")

        row = {
            "id": uuid.UUID(str(alert_id)),
            "transaction_id": uuid.UUID(str(transaction_id)),
            "customer_id": uuid.UUID(str(customer_id)) if customer_id else None,
            "amount": kwargs.get("amount"),
            "currency": kwargs.get("currency"),
            "merchant_name": kwargs.get("merchant_name"),
            "transaction_type": kwargs.get("transaction_type"),
            "timestamp": kwargs.get("timestamp"),
            "risk_score": kwargs["risk_score"],
            "risk_level": kwargs["risk_level"],
            "decision": kwargs["decision"],
            "fraud_probability": kwargs.get("fraud_probability"),
            "model_version": kwargs.get("model_version"),
            "risk_factors": Json(risk_factors) if risk_factors else None,
            "explanation_json": Json(explanation_json) if explanation_json else None,
            "status": OPEN,
            "analyst_id": None,
            "notes": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
        }
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """\
                        INSERT INTO alerts (
                            id, transaction_id, customer_id, amount, currency,
                            merchant_name, transaction_type, timestamp, risk_score,
                            risk_level, decision, fraud_probability, model_version,
                            risk_factors, explanation_json, status, analyst_id, notes,
                            created_at, updated_at, resolved_at
                        ) VALUES (
                            %(id)s, %(transaction_id)s, %(customer_id)s, %(amount)s,
                            %(currency)s, %(merchant_name)s, %(transaction_type)s,
                            %(timestamp)s, %(risk_score)s, %(risk_level)s, %(decision)s,
                            %(fraud_probability)s, %(model_version)s, %(risk_factors)s,
                            %(explanation_json)s, %(status)s, %(analyst_id)s, %(notes)s,
                            %(created_at)s, %(updated_at)s, %(resolved_at)s
                        )""",
                        row,
                    )
        except pg_errors.UniqueViolation:
            # Duplicate transaction_id (race): fall back to lookup
            existing = self.get_by_transaction_id(str(transaction_id))
            if existing is not None:
                return existing
            raise

        # Return a dict with the original Python types (matches SQLite behaviour):
        # JSONB fields stay as Python objects; no ``_deserialise`` needed.
        return {
            "id": str(alert_id),
            "transaction_id": str(transaction_id),
            "customer_id": str(customer_id) if customer_id else None,
            "amount": kwargs.get("amount"),
            "currency": kwargs.get("currency"),
            "merchant_name": kwargs.get("merchant_name"),
            "transaction_type": kwargs.get("transaction_type"),
            "timestamp": kwargs.get("timestamp"),
            "risk_score": kwargs["risk_score"],
            "risk_level": kwargs["risk_level"],
            "decision": kwargs["decision"],
            "fraud_probability": kwargs.get("fraud_probability"),
            "model_version": kwargs.get("model_version"),
            "risk_factors": risk_factors if risk_factors else None,
            "explanation_json": explanation_json if explanation_json else None,
            "status": OPEN,
            "analyst_id": None,
            "notes": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
        }

    def get_by_id(self, alert_id: str) -> dict[str, Any] | None:
        aid = _coerce_uuid(alert_id)
        if aid is None:
            return None
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM alerts WHERE id = %s", (aid,))
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        tid = _coerce_uuid(transaction_id)
        if tid is None:
            return None
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM alerts WHERE transaction_id = %s", (tid,)
                )
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_alerts(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = %s")
            params.append(status)
        if risk_level:
            clauses.append("risk_level = %s")
            params.append(risk_level)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SELECT COUNT(*) FROM alerts {where}", params)
                total_row = cur.fetchone()
                total = int(total_row["count"] if isinstance(total_row, dict) else total_row[0])

                offset = (page - 1) * per_page
                cur.execute(
                    f"SELECT * FROM alerts {where} ORDER BY created_at DESC "
                    f"LIMIT %s OFFSET %s",
                    [*params, per_page, offset],
                )
                rows = cur.fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def update_status(
        self,
        alert_id: str,
        *,
        new_status: str,
        notes: str | None = None,
        analyst_id: str | None = None,
    ) -> dict[str, Any] | None:
        aid = _coerce_uuid(alert_id)
        if aid is None:
            return None
        analyst_uuid = _coerce_uuid(analyst_id)

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM alerts WHERE id = %s", (aid,))
                row = cur.fetchone()
                if row is None:
                    return None

                current_status = row["status"]
                if new_status != current_status and not is_valid_transition(current_status, new_status):
                    return None

                now = datetime.now(timezone.utc).isoformat()
                resolved_at = now if new_status in TERMINAL_STATUSES else row["resolved_at"]
                updated_analyst = row["analyst_id"]
                if analyst_uuid is not None and updated_analyst is None:
                    updated_analyst = analyst_uuid
                updated_notes = notes if notes is not None else row["notes"]

                cur.execute(
                    """\
                    UPDATE alerts SET
                        status = %s, notes = %s, analyst_id = %s,
                        updated_at = %s, resolved_at = %s
                    WHERE id = %s""",
                    (new_status, updated_notes, updated_analyst, now, resolved_at, aid),
                )
                cur.execute("SELECT * FROM alerts WHERE id = %s", (aid,))
                updated_row = cur.fetchone()

        return self._row_to_dict(updated_row) if updated_row else None

    # ── Internal helpers ──────────────────────────────────────────────

    def close(self) -> None:
        """Close the shared connection pool (safe to call more than once)."""
        try:
            self._pool.close()
        except Exception:
            pass

    @staticmethod
    def _row_to_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Convert psycopg row types to the dict shape used by routers.

        UUID fields are stringified, timestamp fields are ISO-formatted,
        and JSONB fields are already Python objects (no deserialisation
        needed, unlike the SQLite implementation).
        """
        result = dict(d)
        for field in ("id", "transaction_id", "customer_id", "analyst_id"):
            v = result.get(field)
            if v is not None:
                result[field] = str(v)
        for field in ("created_at", "updated_at", "resolved_at"):
            v = result.get(field)
            if v is not None and hasattr(v, "isoformat"):
                result[field] = v.isoformat()
        return result
