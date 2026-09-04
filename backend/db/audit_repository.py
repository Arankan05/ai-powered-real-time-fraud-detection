"""Audit repository — append-only fraud decision audit trail.

Step 45: Production-ready audit logging for fraud decisions.  Every
important decision-related event is recorded so that historical fraud
decisions can be reconstructed later.

Event types
-----------
* ``DECISION_MADE`` — ML prediction completed successfully
* ``ML_FAILURE`` — ML service was unavailable, timed out, or errored
* ``ALERT_CREATED`` — fraud alert created for a HOLD decision
* ``ALERT_STATE_CHANGED`` — analyst changed alert status
* ``OUTCOME_RECORDED`` — fraud outcome feedback recorded

Append-only design
------------------
The repository only exposes a ``create`` (append) operation and a
``list_by_transaction`` read operation.  There is **no** ``update``
or ``delete`` — old audit events are immutable.

Implementations
~~~~~~~~~~~~~~~
* :class:`InMemoryAuditStore` — volatile dict-backed store for tests.
* :class:`PostgresAuditRepository` — production PostgreSQL store.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from backend.db.user_repository import _coerce_uuid

logger = logging.getLogger(__name__)

# ── Event-type constants ───────────────────────────────────────────────

DECISION_MADE = "DECISION_MADE"
ML_FAILURE = "ML_FAILURE"
ALERT_CREATED = "ALERT_CREATED"
ALERT_STATE_CHANGED = "ALERT_STATE_CHANGED"
OUTCOME_RECORDED = "OUTCOME_RECORDED"

VALID_EVENT_TYPES = frozenset({
    DECISION_MADE,
    ML_FAILURE,
    ALERT_CREATED,
    ALERT_STATE_CHANGED,
    OUTCOME_RECORDED,
})

# ── Bounding constants ────────────────────────────────────────────────

MAX_EXPLANATION_ITEMS = 5
MAX_EXPLANATION_FIELD_LEN = 200
MAX_RULE_ITEMS = 10
MAX_FAILURE_CATEGORY_LEN = 50


# ── Bounded summary helpers ────────────────────────────────────────────


def build_explanation_summary(
    explanation: Any | None,
) -> dict[str, Any] | None:
    """Build a bounded explanation summary from an MLExplanation-like object.

    Returns a dict with at most ``MAX_EXPLANATION_ITEMS`` top factors
    and ``MAX_RULE_ITEMS`` triggered rules, with all strings truncated
    to ``MAX_EXPLANATION_FIELD_LEN``.  Returns ``None`` when no
    explanation is available.
    """
    if explanation is None:
        return None

    summary: dict[str, Any] = {}

    # ml_top_factors
    factors = getattr(explanation, "ml_top_factors", None)
    if factors is None and isinstance(explanation, dict):
        factors = explanation.get("ml_top_factors")
    if factors:
        bounded_factors = []
        for f in factors[:MAX_EXPLANATION_ITEMS]:
            if hasattr(f, "feature"):
                name = (f.feature or "")[:MAX_EXPLANATION_FIELD_LEN]
                importance = f.importance
            elif isinstance(f, dict):
                name = str(f.get("feature", ""))[:MAX_EXPLANATION_FIELD_LEN]
                importance = f.get("importance", 0)
            else:
                continue
            bounded_factors.append({"feature": name, "importance": importance})
        summary["ml_top_factors"] = bounded_factors

    # rules_triggered
    rules = getattr(explanation, "rules_triggered", None)
    if rules is None and isinstance(explanation, dict):
        rules = explanation.get("rules_triggered")
    if rules:
        bounded_rules = []
        for r in rules[:MAX_RULE_ITEMS]:
            if hasattr(r, "rule"):
                rule_name = (r.rule or "")[:MAX_EXPLANATION_FIELD_LEN]
                contribution = r.contribution
            elif isinstance(r, dict):
                rule_name = str(r.get("rule", ""))[:MAX_EXPLANATION_FIELD_LEN]
                contribution = r.get("contribution", 0)
            else:
                continue
            bounded_rules.append({"rule": rule_name, "contribution": contribution})
        summary["rules_triggered"] = bounded_rules

    return summary if summary else None


def build_rule_signal_summary(
    risk_factors: list[str] | None,
    explanation: Any | None,
) -> dict[str, Any] | None:
    """Build a bounded rule-signal summary.

    Combines risk_factors (list of strings) and triggered rules into a
    compact dict.  Returns ``None`` when nothing is available.
    """
    result: dict[str, Any] = {}

    if risk_factors:
        result["risk_factors"] = [
            str(rf)[:MAX_EXPLANATION_FIELD_LEN]
            for rf in risk_factors[:MAX_RULE_ITEMS]
        ]

    if explanation is not None:
        rules = getattr(explanation, "rules_triggered", None)
        if rules is None and isinstance(explanation, dict):
            rules = explanation.get("rules_triggered")
        if rules:
            bounded = []
            for r in rules[:MAX_RULE_ITEMS]:
                if hasattr(r, "rule"):
                    bounded.append(str(r.rule)[:MAX_EXPLANATION_FIELD_LEN])
                elif isinstance(r, dict):
                    bounded.append(str(r.get("rule", ""))[:MAX_EXPLANATION_FIELD_LEN])
            if bounded:
                result["triggered_rules"] = bounded

    return result if result else None


def normalize_failure_category(category: str | None) -> str:
    """Normalise a failure category string to a bounded, safe value."""
    if not category:
        return "unknown"
    return category[:MAX_FAILURE_CATEGORY_LEN].strip() or "unknown"


# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class AuditRepository(Protocol):
    """Abstract contract for audit persistence (append-only)."""

    def create(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        event_type: str,
        decision: str | None = None,
        risk_score: int | None = None,
        risk_level: str | None = None,
        fraud_probability: float | None = None,
        model_version: str | None = None,
        explanation_summary: dict[str, Any] | None = None,
        rule_signal_summary: dict[str, Any] | None = None,
        failure_category: str | None = None,
        actor_id: str | None = None,
        actor_role: str | None = None,
        previous_state: str | None = None,
        new_state: str | None = None,
        alert_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an audit event.  Returns the created audit record dict."""
        ...

    def list_by_transaction(
        self,
        transaction_id: str,
    ) -> list[dict[str, Any]]:
        """Return all audit events for a transaction, ordered by created_at ASC."""
        ...

    def count_by_transaction_and_event(
        self,
        transaction_id: str,
        event_type: str,
    ) -> int:
        """Count audit events of a given type for a transaction."""
        ...


# ── In-memory implementation ─────────────────────────────────────────


class InMemoryAuditStore:
    """Volatile in-memory audit store (for tests)."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        event_type: str,
        decision: str | None = None,
        risk_score: int | None = None,
        risk_level: str | None = None,
        fraud_probability: float | None = None,
        model_version: str | None = None,
        explanation_summary: dict[str, Any] | None = None,
        rule_signal_summary: dict[str, Any] | None = None,
        failure_category: str | None = None,
        actor_id: str | None = None,
        actor_role: str | None = None,
        previous_state: str | None = None,
        new_state: str | None = None,
        alert_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        event = {
            "audit_id": str(uuid.uuid4()),
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "event_type": event_type,
            "decision": decision,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "model_version": model_version,
            "explanation_summary": explanation_summary,
            "rule_signal_summary": rule_signal_summary,
            "failure_category": failure_category,
            "actor_id": actor_id,
            "actor_role": actor_role,
            "previous_state": previous_state,
            "new_state": new_state,
            "alert_id": alert_id,
            "metadata": metadata,
            "created_at": now,
        }
        with self._lock:
            self._events.append(event)
        return dict(event)

    def list_by_transaction(
        self,
        transaction_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = [
                dict(e) for e in self._events
                if e["transaction_id"] == transaction_id
            ]
        results.sort(key=lambda e: e["created_at"])
        return results

    def count_by_transaction_and_event(
        self,
        transaction_id: str,
        event_type: str,
    ) -> int:
        with self._lock:
            return sum(
                1 for e in self._events
                if e["transaction_id"] == transaction_id
                and e["event_type"] == event_type
            )

    def reset(self) -> None:
        """Clear all events (tests only)."""
        with self._lock:
            self._events.clear()


# ── PostgreSQL implementation ────────────────────────────────────────


class PostgresAuditRepository:
    """PostgreSQL-backed append-only audit repository.

    The ``fraud_decision_audit`` table is created idempotently at
    construction time.  There are no UPDATE or DELETE paths — the
    table is append-only by design.
    """

    _SCHEMA_STATEMENTS: tuple[str, ...] = (
        """\
CREATE TABLE IF NOT EXISTS fraud_decision_audit (
    audit_id             UUID PRIMARY KEY,
    transaction_id       UUID NOT NULL,
    customer_id          UUID NOT NULL,
    event_type           VARCHAR(30) NOT NULL
                         CHECK (event_type IN (
                             'DECISION_MADE', 'ML_FAILURE',
                             'ALERT_CREATED', 'ALERT_STATE_CHANGED',
                             'OUTCOME_RECORDED')),
    decision             VARCHAR(20),
    risk_score           INTEGER,
    risk_level           VARCHAR(10),
    fraud_probability    DOUBLE PRECISION,
    model_version        VARCHAR(100),
    explanation_summary  JSONB,
    rule_signal_summary   JSONB,
    failure_category     VARCHAR(50),
    actor_id             UUID,
    actor_role           VARCHAR(20),
    previous_state       VARCHAR(20),
    new_state            VARCHAR(20),
    alert_id             UUID,
    metadata             JSONB,
    created_at           TIMESTAMPTZ NOT NULL
)""",
        "CREATE INDEX IF NOT EXISTS ix_audit_transaction_id ON fraud_decision_audit (transaction_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_customer_id ON fraud_decision_audit (customer_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_created_at ON fraud_decision_audit (created_at)",
        "CREATE INDEX IF NOT EXISTS ix_audit_event_type ON fraud_decision_audit (event_type)",
        # Prevent duplicate DECISION_MADE events per transaction (the
        # primary idempotency guard — replays must not duplicate audits).
        """\
CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_decision_per_txn
    ON fraud_decision_audit (transaction_id, event_type)
    WHERE event_type = 'DECISION_MADE'""",
    )

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in self._SCHEMA_STATEMENTS:
                    cur.execute(stmt)

    def create(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        event_type: str,
        decision: str | None = None,
        risk_score: int | None = None,
        risk_level: str | None = None,
        fraud_probability: float | None = None,
        model_version: str | None = None,
        explanation_summary: dict[str, Any] | None = None,
        rule_signal_summary: dict[str, Any] | None = None,
        failure_category: str | None = None,
        actor_id: str | None = None,
        actor_role: str | None = None,
        previous_state: str | None = None,
        new_state: str | None = None,
        alert_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from psycopg import errors as pg_errors
        from psycopg.types.json import Json

        audit_id = uuid.uuid4()
        now = datetime.now(timezone.utc).isoformat()

        row = {
            "audit_id": audit_id,
            "transaction_id": uuid.UUID(str(transaction_id)),
            "customer_id": uuid.UUID(str(customer_id)),
            "event_type": event_type,
            "decision": decision,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "model_version": model_version,
            "explanation_summary": Json(explanation_summary) if explanation_summary else None,
            "rule_signal_summary": Json(rule_signal_summary) if rule_signal_summary else None,
            "failure_category": normalize_failure_category(failure_category) if failure_category else None,
            "actor_id": uuid.UUID(str(actor_id)) if actor_id else None,
            "actor_role": actor_role,
            "previous_state": previous_state,
            "new_state": new_state,
            "alert_id": uuid.UUID(str(alert_id)) if alert_id else None,
            "metadata": Json(metadata) if metadata else None,
            "created_at": now,
        }
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """\
                        INSERT INTO fraud_decision_audit (
                            audit_id, transaction_id, customer_id, event_type,
                            decision, risk_score, risk_level, fraud_probability,
                            model_version, explanation_summary, rule_signal_summary,
                            failure_category, actor_id, actor_role,
                            previous_state, new_state, alert_id, metadata,
                            created_at
                        ) VALUES (
                            %(audit_id)s, %(transaction_id)s, %(customer_id)s, %(event_type)s,
                            %(decision)s, %(risk_score)s, %(risk_level)s, %(fraud_probability)s,
                            %(model_version)s, %(explanation_summary)s, %(rule_signal_summary)s,
                            %(failure_category)s, %(actor_id)s, %(actor_role)s,
                            %(previous_state)s, %(new_state)s, %(alert_id)s, %(metadata)s,
                            %(created_at)s
                        )""",
                        row,
                    )
        except pg_errors.UniqueViolation:
            # Duplicate DECISION_MADE for same transaction — this is the
            # idempotency guard.  Log and return a sentinel so the caller
            # knows no new event was created.
            logger.info(
                "Audit idempotency guard: duplicate %s for transaction %s",
                event_type, transaction_id,
            )
            return {"duplicate": True, "transaction_id": transaction_id}

        return {
            "audit_id": str(audit_id),
            "transaction_id": str(transaction_id),
            "customer_id": str(customer_id),
            "event_type": event_type,
            "decision": decision,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "model_version": model_version,
            "explanation_summary": explanation_summary,
            "rule_signal_summary": rule_signal_summary,
            "failure_category": failure_category,
            "actor_id": str(actor_id) if actor_id else None,
            "actor_role": actor_role,
            "previous_state": previous_state,
            "new_state": new_state,
            "alert_id": str(alert_id) if alert_id else None,
            "metadata": metadata,
            "created_at": now,
        }

    def list_by_transaction(
        self,
        transaction_id: str,
    ) -> list[dict[str, Any]]:
        tid = _coerce_uuid(transaction_id)
        if tid is None:
            return []
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM fraud_decision_audit "
                    "WHERE transaction_id = %s "
                    "ORDER BY created_at ASC",
                    (tid,),
                )
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_by_transaction_and_event(
        self,
        transaction_id: str,
        event_type: str,
    ) -> int:
        tid = _coerce_uuid(transaction_id)
        if tid is None:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM fraud_decision_audit "
                    "WHERE transaction_id = %s AND event_type = %s",
                    (tid, event_type),
                )
                row = cur.fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Convert psycopg dict_row to the standard audit dict shape."""
        result = dict(d)
        for field in ("audit_id", "transaction_id", "customer_id", "actor_id", "alert_id"):
            v = result.get(field)
            if v is not None:
                result[field] = str(v)
        v = result.get("created_at")
        if v is not None and hasattr(v, "isoformat"):
            result["created_at"] = v.isoformat()
        return result
