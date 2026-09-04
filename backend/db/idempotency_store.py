"""Idempotency store — prevents duplicate transaction processing.

Step 44: Production-ready idempotency for the transaction creation
endpoint.  Clients can supply an ``Idempotency-Key`` HTTP header; the
same authenticated customer using the same key will never create
multiple transactions.

Implementations
---------------
* :class:`InMemoryIdempotencyStore` — thread-safe dict-backed store for
  tests and lightweight local development.
* :class:`PostgresIdempotencyStore` — database-backed store with a
  UNIQUE constraint on ``(customer_id, idempotency_key)`` that guards
  against race conditions at the database level.

The store follows the project's repository-protocol pattern: the router
depends on the :class:`IdempotencyStore` protocol, not a concrete
implementation.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Record ─────────────────────────────────────────────────────────────


class IdempotencyRecord:
    """A stored idempotency state."""

    __slots__ = (
        "customer_id",
        "idempotency_key",
        "status",
        "transaction_id",
        "response_json",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        *,
        customer_id: str,
        idempotency_key: str,
        status: str = "processing",
        transaction_id: str | None = None,
        response_json: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.customer_id = customer_id
        self.idempotency_key = idempotency_key
        self.status = status  # "processing" | "completed" | "failed"
        self.transaction_id = transaction_id
        self.response_json = response_json
        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now


# ── Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class IdempotencyStore(Protocol):
    """Protocol that all idempotency store implementations must satisfy."""

    def try_reserve(
        self, customer_id: str, idempotency_key: str
    ) -> IdempotencyRecord | None:
        """Atomically reserve an idempotency key for *customer_id*.

        Returns
        -------
        IdempotencyRecord or None
            ``None`` if a new "processing" record was created (caller
            should proceed).  An :class:`IdempotencyRecord` if one
            already exists for this (customer_id, idempotency_key).
        """
        ...

    def mark_completed(
        self,
        customer_id: str,
        idempotency_key: str,
        transaction_id: str,
        response_data: dict[str, Any],
    ) -> None:
        """Mark a reserved key as completed and cache the response."""
        ...

    def mark_failed(
        self, customer_id: str, idempotency_key: str
    ) -> None:
        """Mark a reserved key as failed (allows future retry)."""
        ...

    def reset(self) -> None:
        """Clear all records (tests only)."""
        ...


# ── In-memory implementation ──────────────────────────────────────────


class InMemoryIdempotencyStore:
    """Thread-safe dict-backed idempotency store.

    Suitable for tests and single-process local development.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Composite key: (customer_id, idempotency_key) → record
        self._store: dict[tuple[str, str], IdempotencyRecord] = {}

    def try_reserve(
        self, customer_id: str, idempotency_key: str
    ) -> IdempotencyRecord | None:
        with self._lock:
            key = (customer_id, idempotency_key)
            existing = self._store.get(key)
            if existing is not None:
                return existing
            self._store[key] = IdempotencyRecord(
                customer_id=customer_id,
                idempotency_key=idempotency_key,
                status="processing",
            )
            return None  # new reservation — proceed

    def mark_completed(
        self,
        customer_id: str,
        idempotency_key: str,
        transaction_id: str,
        response_data: dict[str, Any],
    ) -> None:
        with self._lock:
            key = (customer_id, idempotency_key)
            record = self._store.get(key)
            if record is not None:
                record.status = "completed"
                record.transaction_id = transaction_id
                record.response_json = json.dumps(
                    response_data, default=str
                )
                record.updated_at = datetime.now(timezone.utc)

    def mark_failed(
        self, customer_id: str, idempotency_key: str
    ) -> None:
        with self._lock:
            key = (customer_id, idempotency_key)
            record = self._store.get(key)
            if record is not None:
                record.status = "failed"
                record.updated_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        with self._lock:
            self._store.clear()


# ── PostgreSQL implementation ─────────────────────────────────────────


class PostgresIdempotencyStore:
    """Database-backed idempotency store using PostgreSQL.

    Uses a UNIQUE constraint on ``(customer_id, idempotency_key)``
    to guard against race conditions at the database level.

    The schema is created idempotently by :func:`init_schema`.
    """

    _SCHEMA_STATEMENTS: tuple[str, ...] = (
        """\
CREATE TABLE IF NOT EXISTS idempotency_keys (
    id              UUID PRIMARY KEY,
    customer_id     UUID NOT NULL,
    idempotency_key VARCHAR(255) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'processing',
    transaction_id  UUID,
    response_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
)""",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_idempotency_customer_key ON idempotency_keys (customer_id, idempotency_key)",
        "CREATE INDEX IF NOT EXISTS ix_idempotency_keys_created_at ON idempotency_keys (created_at)",
    )

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in self._SCHEMA_STATEMENTS:
                    cur.execute(stmt)

    def try_reserve(
        self, customer_id: str, idempotency_key: str
    ) -> IdempotencyRecord | None:
        import uuid as _uuid

        from psycopg import errors

        now = datetime.now(timezone.utc)
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # First check if record exists
                    cur.execute(
                        "SELECT customer_id, idempotency_key, status, "
                        "transaction_id, response_json, created_at, updated_at "
                        "FROM idempotency_keys "
                        "WHERE customer_id = %s AND idempotency_key = %s",
                        (customer_id, idempotency_key),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        return IdempotencyRecord(
                            customer_id=str(row[0]),
                            idempotency_key=str(row[1]),
                            status=row[2],
                            transaction_id=str(row[3]) if row[3] else None,
                            response_json=row[4] if row[4] else None,
                            created_at=row[5],
                            updated_at=row[6],
                        )
                    # Insert new record — UNIQUE constraint is the
                    # race-condition guard
                    cur.execute(
                        "INSERT INTO idempotency_keys "
                        "(id, customer_id, idempotency_key, status, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            _uuid.uuid4(),
                            customer_id,
                            idempotency_key,
                            "processing",
                            now,
                            now,
                        ),
                    )
                    return None  # new reservation — proceed
        except errors.UniqueViolation:
            # Race condition: another request inserted the same key
            # between our SELECT and INSERT.  Re-read the record.
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT customer_id, idempotency_key, status, "
                        "transaction_id, response_json, created_at, updated_at "
                        "FROM idempotency_keys "
                        "WHERE customer_id = %s AND idempotency_key = %s",
                        (customer_id, idempotency_key),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        return IdempotencyRecord(
                            customer_id=str(row[0]),
                            idempotency_key=str(row[1]),
                            status=row[2],
                            transaction_id=str(row[3]) if row[3] else None,
                            response_json=row[4] if row[4] else None,
                            created_at=row[5],
                            updated_at=row[6],
                        )
            # Extremely unlikely: record was deleted between violation and re-read
            return None

    def mark_completed(
        self,
        customer_id: str,
        idempotency_key: str,
        transaction_id: str,
        response_data: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE idempotency_keys SET status = %s, "
                    "transaction_id = %s, response_json = %s, "
                    "updated_at = %s "
                    "WHERE customer_id = %s AND idempotency_key = %s",
                    (
                        "completed",
                        transaction_id,
                        json.dumps(response_data, default=str),
                        now,
                        customer_id,
                        idempotency_key,
                    ),
                )

    def mark_failed(
        self, customer_id: str, idempotency_key: str
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE idempotency_keys SET status = %s, "
                    "updated_at = %s "
                    "WHERE customer_id = %s AND idempotency_key = %s",
                    ("failed", now, customer_id, idempotency_key),
                )

    def reset(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM idempotency_keys")
