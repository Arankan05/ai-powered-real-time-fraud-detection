"""Promotion governance repository — Step 50.

Production-safe governance records for model promotion decisions.
Tracks the lifecycle of a candidate model promotion from gate decision
through human approval/rejection to eventual activation.

Governance states
-----------------
* ``PENDING`` — gate decision recorded, awaiting human review
* ``APPROVED`` — reviewer approved; ready for Step 46 activation
* ``REJECTED`` — reviewer rejected the promotion
* ``PROMOTED`` — operator confirmed Step 46 activation completed

Valid transitions::

    PENDING → APPROVED
    PENDING → REJECTED
    APPROVED → PROMOTED

All other transitions are rejected.

Append-only audit
-----------------
Every governance event (create, approve, reject, mark-promoted) is
recorded in the existing fraud-decision audit trail (Step 45).

Security
--------
* Actor identity always comes from the JWT — never from the request
  payload.
* Customers cannot create, approve, or reject promotions.
* Only ``fraud_analyst`` and ``admin`` roles may interact with
  governance records.
* Model identity (checksum, version, threshold) is taken from the
  verified Step 48 gate decision — never from client input.
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

__all__ = [
    "STATUS_PENDING",
    "STATUS_APPROVED",
    "STATUS_REJECTED",
    "STATUS_PROMOTED",
    "VALID_STATUSES",
    "TERMINAL_STATUSES",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "AUDIT_PROMOTION_CREATED",
    "AUDIT_PROMOTION_APPROVED",
    "AUDIT_PROMOTION_REJECTED",
    "AUDIT_PROMOTION_MARKED_PROMOTED",
    "ACTIVATION_NONE",
    "ACTIVATION_TOKEN_ISSUED",
    "ACTIVATION_CONSUMED",
    "MAX_COMMENT_LENGTH",
    "PromotionGovernanceRepository",
    "InMemoryPromotionGovernanceStore",
    "PostgresPromotionGovernanceRepository",
    "DuplicatePromotionError",
    "InvalidTransitionError",
    "PromotionNotFoundError",
]


# ── Status constants ──────────────────────────────────────────────────

STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_PROMOTED = "PROMOTED"

VALID_STATUSES = frozenset({STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED, STATUS_PROMOTED})
TERMINAL_STATUSES = frozenset({STATUS_REJECTED, STATUS_PROMOTED})

VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_PENDING: frozenset({STATUS_APPROVED, STATUS_REJECTED}),
    STATUS_APPROVED: frozenset({STATUS_PROMOTED}),
    STATUS_REJECTED: frozenset(),
    STATUS_PROMOTED: frozenset(),
}


def is_valid_transition(current: str, target: str) -> bool:
    """Return True if the governance status transition is allowed."""
    return target in VALID_TRANSITIONS.get(current, frozenset())


# ── Audit event types ─────────────────────────────────────────────────

AUDIT_PROMOTION_CREATED = "PROMOTION_CREATED"
AUDIT_PROMOTION_APPROVED = "PROMOTION_APPROVED"
AUDIT_PROMOTION_REJECTED = "PROMOTION_REJECTED"
AUDIT_PROMOTION_MARKED_PROMOTED = "PROMOTION_MARKED_PROMOTED"

# ── Activation status constants (Step 51) ─────────────────────────────

ACTIVATION_NONE = "NONE"
ACTIVATION_TOKEN_ISSUED = "TOKEN_ISSUED"
ACTIVATION_CONSUMED = "CONSUMED"

# ── Bounding constants ────────────────────────────────────────────────

MAX_COMMENT_LENGTH = 500
MAX_MODEL_VERSION_LENGTH = 100
MAX_CHECKSUM_LENGTH = 128
MAX_SCHEMA_VERSION_LENGTH = 20


# ── Exceptions ────────────────────────────────────────────────────────


class DuplicatePromotionError(Exception):
    """A governance record already exists for this gate decision."""


class InvalidTransitionError(Exception):
    """The requested governance status transition is not allowed."""


class PromotionNotFoundError(Exception):
    """The requested promotion governance record was not found."""


# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class PromotionGovernanceRepository(Protocol):
    """Abstract contract for promotion governance persistence."""

    def create(
        self,
        *,
        gate_decision: str,
        candidate_model_name: str,
        candidate_model_version: str,
        candidate_checksum: str,
        candidate_schema_version: str,
        candidate_n_features: int,
        production_model_name: str,
        production_model_version: str,
        production_checksum: str,
        production_schema_version: str,
        production_n_features: int,
        gate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a PENDING governance record from a gate decision."""
        ...

    def get_by_id(self, promotion_id: str) -> dict[str, Any] | None:
        """Return a single governance record by ID, or None."""
        ...

    def list_records(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List governance records with optional status filter."""
        ...

    def count_records(self, *, status: str | None = None) -> int:
        """Count governance records with optional status filter."""
        ...

    def approve(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Transition PENDING → APPROVED."""
        ...

    def reject(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Transition PENDING → REJECTED."""
        ...

    def mark_promoted(
        self,
        promotion_id: str,
        *,
        actor_id: str,
        actor_role: str,
    ) -> dict[str, Any]:
        """Transition APPROVED → PROMOTED."""
        ...


# ── Helpers ───────────────────────────────────────────────────────────


def _truncate(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    return value[:max_len]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── In-memory implementation ─────────────────────────────────────────


class InMemoryPromotionGovernanceStore:
    """Volatile in-memory governance store (for tests)."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        gate_decision: str,
        candidate_model_name: str,
        candidate_model_version: str,
        candidate_checksum: str,
        candidate_schema_version: str,
        candidate_n_features: int,
        production_model_name: str,
        production_model_version: str,
        production_checksum: str,
        production_schema_version: str,
        production_n_features: int,
        gate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        promotion_id = str(uuid.uuid4())
        now = _now_iso()

        # Idempotency: check for duplicate candidate+decision
        with self._lock:
            for rec in self._records.values():
                if (
                    rec["candidate_model_version"] == candidate_model_version
                    and rec["candidate_checksum"] == candidate_checksum
                    and rec["gate_decision"] == gate_decision
                ):
                    raise DuplicatePromotionError(
                        "Governance record already exists for this gate decision"
                    )

            record = {
                "promotion_id": promotion_id,
                "gate_decision": gate_decision,
                "governance_status": STATUS_PENDING,
                "candidate_model_name": _truncate(candidate_model_name, MAX_MODEL_VERSION_LENGTH),
                "candidate_model_version": _truncate(candidate_model_version, MAX_MODEL_VERSION_LENGTH),
                "candidate_checksum": _truncate(candidate_checksum, MAX_CHECKSUM_LENGTH),
                "candidate_schema_version": _truncate(candidate_schema_version, MAX_SCHEMA_VERSION_LENGTH),
                "candidate_n_features": candidate_n_features,
                "production_model_name": _truncate(production_model_name, MAX_MODEL_VERSION_LENGTH),
                "production_model_version": _truncate(production_model_version, MAX_MODEL_VERSION_LENGTH),
                "production_checksum": _truncate(production_checksum, MAX_CHECKSUM_LENGTH),
                "production_schema_version": _truncate(production_schema_version, MAX_SCHEMA_VERSION_LENGTH),
                "production_n_features": production_n_features,
                "gate_report": gate_report,
                "reviewer_id": None,
                "reviewer_role": None,
                "reviewed_at": None,
                "approval_comment": None,
                "rejection_reason": None,
                "execution_status": None,
                "promoted_by": None,
                "promoted_at": None,
                "activation_status": ACTIVATION_NONE,
                "activation_token_issued_at": None,
                "activation_token_expires_at": None,
                "activation_consumed_at": None,
                "activation_actor_id": None,
                "created_at": now,
                "updated_at": now,
            }
            self._records[promotion_id] = record
            return dict(record)

    def get_by_id(self, promotion_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._records.get(promotion_id)
            return dict(rec) if rec else None

    def list_records(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records.values())
        if status:
            records = [r for r in records if r["governance_status"] == status]
        records.sort(key=lambda r: r["created_at"], reverse=True)
        return [dict(r) for r in records[offset : offset + limit]]

    def count_records(self, *, status: str | None = None) -> int:
        with self._lock:
            records = list(self._records.values())
        if status:
            return sum(1 for r in records if r["governance_status"] == status)
        return len(records)

    def approve(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(promotion_id)
            if rec is None:
                raise PromotionNotFoundError(f"Promotion {promotion_id} not found")
            if not is_valid_transition(rec["governance_status"], STATUS_APPROVED):
                raise InvalidTransitionError(
                    f"Cannot transition from {rec['governance_status']} to {STATUS_APPROVED}"
                )
            # Idempotency: if already approved by same reviewer, return existing
            if rec["reviewer_id"] == reviewer_id and rec["governance_status"] == STATUS_APPROVED:
                return dict(rec)

            rec["governance_status"] = STATUS_APPROVED
            rec["reviewer_id"] = reviewer_id
            rec["reviewer_role"] = reviewer_role
            rec["reviewed_at"] = _now_iso()
            rec["approval_comment"] = _truncate(comment, MAX_COMMENT_LENGTH)
            rec["updated_at"] = _now_iso()
            return dict(rec)

    def reject(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(promotion_id)
            if rec is None:
                raise PromotionNotFoundError(f"Promotion {promotion_id} not found")
            if not is_valid_transition(rec["governance_status"], STATUS_REJECTED):
                raise InvalidTransitionError(
                    f"Cannot transition from {rec['governance_status']} to {STATUS_REJECTED}"
                )
            rec["governance_status"] = STATUS_REJECTED
            rec["reviewer_id"] = reviewer_id
            rec["reviewer_role"] = reviewer_role
            rec["reviewed_at"] = _now_iso()
            rec["rejection_reason"] = _truncate(reason, MAX_COMMENT_LENGTH)
            rec["updated_at"] = _now_iso()
            return dict(rec)

    def mark_promoted(
        self,
        promotion_id: str,
        *,
        actor_id: str,
        actor_role: str,
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(promotion_id)
            if rec is None:
                raise PromotionNotFoundError(f"Promotion {promotion_id} not found")
            if not is_valid_transition(rec["governance_status"], STATUS_PROMOTED):
                raise InvalidTransitionError(
                    f"Cannot transition from {rec['governance_status']} to {STATUS_PROMOTED}"
                )
            rec["governance_status"] = STATUS_PROMOTED
            rec["promoted_by"] = actor_id
            rec["promoted_at"] = _now_iso()
            rec["execution_status"] = "ACTIVATED_VIA_STEP_46"
            rec["updated_at"] = _now_iso()
            return dict(rec)

    def reset(self) -> None:
        """Clear all records (tests only)."""
        with self._lock:
            self._records.clear()

    def update_activation_status(
        self,
        promotion_id: str,
        *,
        activation_status: str,
        token_issued_at: str | None = None,
        token_expires_at: str | None = None,
        consumed_at: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        """Update the activation status of a governance record."""
        with self._lock:
            rec = self._records.get(promotion_id)
            if rec is None:
                raise PromotionNotFoundError(f"Promotion {promotion_id} not found")
            rec["activation_status"] = activation_status
            if token_issued_at is not None:
                rec["activation_token_issued_at"] = token_issued_at
            if token_expires_at is not None:
                rec["activation_token_expires_at"] = token_expires_at
            if consumed_at is not None:
                rec["activation_consumed_at"] = consumed_at
            if actor_id is not None:
                rec["activation_actor_id"] = actor_id
            rec["updated_at"] = _now_iso()
            return dict(rec)

    def try_transition_activation_status(
        self,
        promotion_id: str,
        *,
        expected_status: str,
        new_status: str,
        consumed_at: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Atomically transition activation status (compare-and-set).

        Returns the updated record on success, or ``None`` if the
        current status does not match ``expected_status``.
        """
        with self._lock:
            rec = self._records.get(promotion_id)
            if rec is None:
                raise PromotionNotFoundError(f"Promotion {promotion_id} not found")
            if rec["activation_status"] != expected_status:
                return None  # CAS failed
            rec["activation_status"] = new_status
            if consumed_at is not None:
                rec["activation_consumed_at"] = consumed_at
            if actor_id is not None:
                rec["activation_actor_id"] = actor_id
            rec["updated_at"] = _now_iso()
            return dict(rec)


# ── PostgreSQL implementation ────────────────────────────────────────


class PostgresPromotionGovernanceRepository:
    """PostgreSQL-backed promotion governance repository.

    The ``promotion_governance`` table is created idempotently at
    construction time.
    """

    _SCHEMA_STATEMENTS: tuple[str, ...] = (
        """\
CREATE TABLE IF NOT EXISTS promotion_governance (
    promotion_id               UUID PRIMARY KEY,
    gate_decision              VARCHAR(10) NOT NULL
                               CHECK (gate_decision IN ('APPROVED', 'REJECTED')),
    governance_status          VARCHAR(10) NOT NULL DEFAULT 'PENDING'
                               CHECK (governance_status IN (
                                   'PENDING', 'APPROVED', 'REJECTED', 'PROMOTED')),
    candidate_model_name       VARCHAR(100) NOT NULL,
    candidate_model_version    VARCHAR(100) NOT NULL,
    candidate_checksum         VARCHAR(128) NOT NULL,
    candidate_schema_version   VARCHAR(20) NOT NULL,
    candidate_n_features       INTEGER NOT NULL,
    production_model_name      VARCHAR(100) NOT NULL,
    production_model_version   VARCHAR(100) NOT NULL,
    production_checksum        VARCHAR(128) NOT NULL,
    production_schema_version  VARCHAR(20) NOT NULL,
    production_n_features      INTEGER NOT NULL,
    gate_report                JSONB,
    reviewer_id                UUID,
    reviewer_role              VARCHAR(20),
    reviewed_at                TIMESTAMPTZ,
    approval_comment           VARCHAR(500),
    rejection_reason           VARCHAR(500),
    execution_status           VARCHAR(50),
    promoted_by                UUID,
    promoted_at                TIMESTAMPTZ,
    activation_status          VARCHAR(20) NOT NULL DEFAULT 'NONE'
                               CHECK (activation_status IN (
                                   'NONE', 'TOKEN_ISSUED', 'CONSUMED')),
    activation_token_issued_at TIMESTAMPTZ,
    activation_token_expires_at TIMESTAMPTZ,
    activation_consumed_at     TIMESTAMPTZ,
    activation_actor_id        UUID,
    created_at                 TIMESTAMPTZ NOT NULL,
    updated_at                 TIMESTAMPTZ NOT NULL
)""",
        "CREATE INDEX IF NOT EXISTS ix_promo_gov_status ON promotion_governance (governance_status)",
        "CREATE INDEX IF NOT EXISTS ix_promo_gov_candidate ON promotion_governance (candidate_model_version, candidate_checksum)",
        "CREATE INDEX IF NOT EXISTS ix_promo_gov_created_at ON promotion_governance (created_at DESC)",
        # Idempotency: prevent duplicate governance records for the same
        # candidate version + checksum + gate decision combination.
        """\
CREATE UNIQUE INDEX IF NOT EXISTS uq_promo_gov_candidate_decision
    ON promotion_governance (candidate_model_version, candidate_checksum, gate_decision)""",
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
        gate_decision: str,
        candidate_model_name: str,
        candidate_model_version: str,
        candidate_checksum: str,
        candidate_schema_version: str,
        candidate_n_features: int,
        production_model_name: str,
        production_model_version: str,
        production_checksum: str,
        production_schema_version: str,
        production_n_features: int,
        gate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from psycopg import errors as pg_errors
        from psycopg.types.json import Json

        promotion_id = uuid.uuid4()
        now = _now_iso()

        row = {
            "promotion_id": promotion_id,
            "gate_decision": gate_decision,
            "governance_status": STATUS_PENDING,
            "candidate_model_name": _truncate(candidate_model_name, MAX_MODEL_VERSION_LENGTH),
            "candidate_model_version": _truncate(candidate_model_version, MAX_MODEL_VERSION_LENGTH),
            "candidate_checksum": _truncate(candidate_checksum, MAX_CHECKSUM_LENGTH),
            "candidate_schema_version": _truncate(candidate_schema_version, MAX_SCHEMA_VERSION_LENGTH),
            "candidate_n_features": candidate_n_features,
            "production_model_name": _truncate(production_model_name, MAX_MODEL_VERSION_LENGTH),
            "production_model_version": _truncate(production_model_version, MAX_MODEL_VERSION_LENGTH),
            "production_checksum": _truncate(production_checksum, MAX_CHECKSUM_LENGTH),
            "production_schema_version": _truncate(production_schema_version, MAX_SCHEMA_VERSION_LENGTH),
            "production_n_features": production_n_features,
            "gate_report": Json(gate_report) if gate_report else None,
            "created_at": now,
            "updated_at": now,
        }

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """\
                        INSERT INTO promotion_governance (
                            promotion_id, gate_decision, governance_status,
                            candidate_model_name, candidate_model_version,
                            candidate_checksum, candidate_schema_version,
                            candidate_n_features,
                            production_model_name, production_model_version,
                            production_checksum, production_schema_version,
                            production_n_features,
                            gate_report, created_at, updated_at
                        ) VALUES (
                            %(promotion_id)s, %(gate_decision)s, %(governance_status)s,
                            %(candidate_model_name)s, %(candidate_model_version)s,
                            %(candidate_checksum)s, %(candidate_schema_version)s,
                            %(candidate_n_features)s,
                            %(production_model_name)s, %(production_model_version)s,
                            %(production_checksum)s, %(production_schema_version)s,
                            %(production_n_features)s,
                            %(gate_report)s, %(created_at)s, %(updated_at)s
                        )""",
                        row,
                    )
        except pg_errors.UniqueViolation:
            raise DuplicatePromotionError(
                "Governance record already exists for this gate decision"
            ) from None

        return {
            "promotion_id": str(promotion_id),
            "gate_decision": gate_decision,
            "governance_status": STATUS_PENDING,
            "candidate_model_name": row["candidate_model_name"],
            "candidate_model_version": row["candidate_model_version"],
            "candidate_checksum": row["candidate_checksum"],
            "candidate_schema_version": row["candidate_schema_version"],
            "candidate_n_features": candidate_n_features,
            "production_model_name": row["production_model_name"],
            "production_model_version": row["production_model_version"],
            "production_checksum": row["production_checksum"],
            "production_schema_version": row["production_schema_version"],
            "production_n_features": production_n_features,
            "gate_report": gate_report,
            "reviewer_id": None,
            "reviewer_role": None,
            "reviewed_at": None,
            "approval_comment": None,
            "rejection_reason": None,
            "execution_status": None,
            "promoted_by": None,
            "promoted_at": None,
            "activation_status": ACTIVATION_NONE,
            "activation_token_issued_at": None,
            "activation_token_expires_at": None,
            "activation_consumed_at": None,
            "activation_actor_id": None,
            "created_at": now,
            "updated_at": now,
        }

    def get_by_id(self, promotion_id: str) -> dict[str, Any] | None:
        pid = _coerce_uuid(promotion_id)
        if pid is None:
            return None
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM promotion_governance WHERE promotion_id = %s",
                    (pid,),
                )
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_records(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                if status:
                    cur.execute(
                        "SELECT * FROM promotion_governance "
                        "WHERE governance_status = %s "
                        "ORDER BY created_at DESC "
                        "LIMIT %s OFFSET %s",
                        (status, limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM promotion_governance "
                        "ORDER BY created_at DESC "
                        "LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_records(self, *, status: str | None = None) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if status:
                    cur.execute(
                        "SELECT COUNT(*) FROM promotion_governance WHERE governance_status = %s",
                        (status,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM promotion_governance")
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def approve(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        return self._transition(
            promotion_id,
            target_status=STATUS_APPROVED,
            actor_id=reviewer_id,
            actor_role=reviewer_role,
            comment=_truncate(comment, MAX_COMMENT_LENGTH),
            reason=None,
        )

    def reject(
        self,
        promotion_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._transition(
            promotion_id,
            target_status=STATUS_REJECTED,
            actor_id=reviewer_id,
            actor_role=reviewer_role,
            comment=None,
            reason=_truncate(reason, MAX_COMMENT_LENGTH),
        )

    def mark_promoted(
        self,
        promotion_id: str,
        *,
        actor_id: str,
        actor_role: str,
    ) -> dict[str, Any]:
        return self._transition(
            promotion_id,
            target_status=STATUS_PROMOTED,
            actor_id=actor_id,
            actor_role=actor_role,
            comment=None,
            reason=None,
            mark_promoted=True,
        )

    def _transition(
        self,
        promotion_id: str,
        *,
        target_status: str,
        actor_id: str,
        actor_role: str,
        comment: str | None = None,
        reason: str | None = None,
        mark_promoted: bool = False,
    ) -> dict[str, Any]:
        pid = _coerce_uuid(promotion_id)
        if pid is None:
            raise PromotionNotFoundError(f"Invalid promotion ID: {promotion_id}")

        now = _now_iso()
        reviewer_uuid = _coerce_uuid(actor_id)

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # Read current state with FOR UPDATE to prevent races
                cur.execute(
                    "SELECT * FROM promotion_governance WHERE promotion_id = %s FOR UPDATE",
                    (pid,),
                )
                row = cur.fetchone()
                if row is None:
                    raise PromotionNotFoundError(f"Promotion {promotion_id} not found")

                current_status = row["governance_status"]
                if not is_valid_transition(current_status, target_status):
                    raise InvalidTransitionError(
                        f"Cannot transition from {current_status} to {target_status}"
                    )

                # Build UPDATE
                updates: dict[str, Any] = {
                    "governance_status": target_status,
                    "reviewer_id": reviewer_uuid,
                    "reviewer_role": reviewer_role,
                    "reviewed_at": now,
                    "updated_at": now,
                }
                if comment is not None:
                    updates["approval_comment"] = comment
                if reason is not None:
                    updates["rejection_reason"] = reason
                if mark_promoted:
                    updates["promoted_by"] = reviewer_uuid
                    updates["promoted_at"] = now
                    updates["execution_status"] = "ACTIVATED_VIA_STEP_46"

                set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
                updates["promotion_id"] = pid
                cur.execute(
                    f"UPDATE promotion_governance SET {set_clause} WHERE promotion_id = %(promotion_id)s",
                    updates,
                )

            # Re-read the updated row
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM promotion_governance WHERE promotion_id = %s",
                    (pid,),
                )
                updated = cur.fetchone()

        return self._row_to_dict(updated) if updated else None

    def update_activation_status(
        self,
        promotion_id: str,
        *,
        activation_status: str,
        token_issued_at: str | None = None,
        token_expires_at: str | None = None,
        consumed_at: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        """Update the activation status of a governance record."""
        pid = _coerce_uuid(promotion_id)
        if pid is None:
            raise PromotionNotFoundError(f"Invalid promotion ID: {promotion_id}")

        now = _now_iso()
        actor_uuid = _coerce_uuid(actor_id) if actor_id else None

        updates: dict[str, Any] = {
            "activation_status": activation_status,
            "updated_at": now,
        }
        if token_issued_at is not None:
            updates["activation_token_issued_at"] = token_issued_at
        if token_expires_at is not None:
            updates["activation_token_expires_at"] = token_expires_at
        if consumed_at is not None:
            updates["activation_consumed_at"] = consumed_at
        if actor_uuid is not None:
            updates["activation_actor_id"] = actor_uuid

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
                updates["promotion_id"] = pid
                cur.execute(
                    f"UPDATE promotion_governance SET {set_clause} WHERE promotion_id = %(promotion_id)s",
                    updates,
                )
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM promotion_governance WHERE promotion_id = %s",
                    (pid,),
                )
                updated = cur.fetchone()

        return self._row_to_dict(updated) if updated else None

    def try_transition_activation_status(
        self,
        promotion_id: str,
        *,
        expected_status: str,
        new_status: str,
        consumed_at: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Atomically transition activation status (compare-and-set).

        Uses ``WHERE activation_status = expected_status`` for
        transaction-safe CAS. Returns the updated record on success,
        or ``None`` if the current status does not match.
        """
        pid = _coerce_uuid(promotion_id)
        if pid is None:
            raise PromotionNotFoundError(f"Invalid promotion ID: {promotion_id}")

        now = _now_iso()
        actor_uuid = _coerce_uuid(actor_id) if actor_id else None

        updates: dict[str, Any] = {
            "activation_status": new_status,
            "updated_at": now,
        }
        if consumed_at is not None:
            updates["activation_consumed_at"] = consumed_at
        if actor_uuid is not None:
            updates["activation_actor_id"] = actor_uuid

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
                updates["promotion_id"] = pid
                updates["expected_status"] = expected_status
                cur.execute(
                    f"UPDATE promotion_governance SET {set_clause} "
                    f"WHERE promotion_id = %(promotion_id)s "
                    f"AND activation_status = %(expected_status)s",
                    updates,
                )
                if cur.rowcount == 0:
                    return None  # CAS failed
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM promotion_governance WHERE promotion_id = %s",
                    (pid,),
                )
                updated = cur.fetchone()

        return self._row_to_dict(updated) if updated else None

    @staticmethod
    def _row_to_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Convert psycopg dict_row to the standard governance dict shape."""
        result = dict(d)
        for field in ("promotion_id", "reviewer_id", "promoted_by", "activation_actor_id"):
            v = result.get(field)
            if v is not None:
                result[field] = str(v)
        for field in ("created_at", "updated_at", "reviewed_at", "promoted_at",
                      "activation_token_issued_at", "activation_token_expires_at",
                      "activation_consumed_at"):
            v = result.get(field)
            if v is not None and hasattr(v, "isoformat"):
                result[field] = v.isoformat()
        return result
