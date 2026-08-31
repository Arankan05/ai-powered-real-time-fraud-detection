"""Customer history repository for real-time fraud features.

Provides:

* :class:`TransactionRecord` — typed dataclass for a single historical
  transaction.
* :class:`CustomerHistoryRepository` — abstract protocol defining the
  repository contract.
* :class:`InMemoryHistoryStore` — volatile in-memory implementation
  (fast, loses data on restart).
* :class:`SQLiteHistoryRepository` — persistent SQLite-backed
  implementation (survives restarts, safe for multiple workers via
  WAL journal mode).
* :func:`record_transaction` — helper that creates a history record
  from a raw transaction dict after a successful prediction.

Design rationale
----------------
The ML / Fraud Intelligence Service is a standalone process that needs
to persist customer transaction history across restarts.  The project
plans PostgreSQL for the full backend (see ``docs/database-design.md``
and ``docker-compose.yml``), but that infrastructure is not yet
implemented.

For the ML service, **SQLite** (Python stdlib ``sqlite3``) is the
smallest appropriate persistence layer:

* Zero external dependencies (stdlib only).
* No separate database server required.
* WAL journal mode enables concurrent reads with a single writer.
* Can be replaced by the planned PostgreSQL backend later by
  implementing :class:`CustomerHistoryRepository` and swapping the
  singleton in ``ml/api/app.py``.

Leakage protections
-----------------
* Transactions are recorded **after** each successful prediction.
* Lookups return only entries with ``timestamp < current_timestamp``
  (temporal safety — never uses future data).
* ``is_fraud`` is stored as ``0`` at prediction time (the true label
  is unknown).  :meth:`record_outcome` allows updating later.

Replacing with PostgreSQL
-------------------------
1. Implement :class:`CustomerHistoryRepository` with SQL queries
   against the planned ``transactions`` table.
2. Replace the :data:`history_store` singleton in ``ml/api/app.py``.
3. No changes to :mod:`ml.features.engineer` or the ML model needed.

Thread safety
-------------
Both implementations use :class:`threading.Lock` so they are safe for
concurrent access from FastAPI worker threads.

Usage::

    from ml.features.history import history_store, record_transaction

    # After a successful prediction:
    record_transaction(history_store, raw_data)

    # Retrieve history for a new transaction:
    history = history_store.get("customer_123", before_timestamp=90000)
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ── Typed record ─────────────────────────────────────────────────────


@dataclasses.dataclass
class TransactionRecord:
    """A single historical transaction record.

    All fields except *timestamp* and *amount* are optional and default
    to safe cold-start values.
    """

    timestamp: int = 0
    """TransactionDT in seconds from the dataset reference epoch."""

    amount: float = 0.0
    """TransactionAmt."""

    product_cd: str | None = None
    """ProductCD (W / X / Y / Z / S)."""

    addr1: int | None = None
    """Region code (integer)."""

    addr2: int | None = None
    """Country code (integer)."""

    device_type: str | None = None
    """DeviceType from the identity table."""

    id_19: str | None = None
    """Identity field id_19."""

    id_20: str | None = None
    """Identity field id_20."""

    has_identity_data: int = 0
    """Whether identity-table data exists for this transaction (0/1)."""

    is_fraud: int = 0
    """Fraud label — ``0`` at prediction time (unknown), updated later."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for storage and serialisation)."""
        return dataclasses.asdict(self)


# ── Abstract repository protocol ────────────────────────────────────


@runtime_checkable
class CustomerHistoryRepository(Protocol):
    """Contract that any history backend must satisfy.

    Implement this protocol to replace the in-memory store with a
    database, Redis cache, or remote service.  The feature-engineering
    pipeline depends only on :meth:`get` and :meth:`add`.
    """

    def get(
        self,
        customer_id: str,
        *,
        before_timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical transactions for *customer_id*.

        Only entries with ``timestamp < before_timestamp`` should be
        returned (temporal safety).  Results sorted by timestamp
        ascending.  Empty list when no history exists.
        """
        ...

    def add(self, customer_id: str, record: dict[str, Any]) -> None:
        """Record a transaction for *customer_id*."""
        ...


# ── In-memory implementation ────────────────────────────────────────


class InMemoryHistoryStore:
    """Thread-safe, in-memory history store keyed by customer ID.

    Implements :class:`CustomerHistoryRepository`.

    Args:
        max_per_customer: Maximum entries kept per customer.  Oldest
                          entries are evicted when the limit is reached.
    """

    def __init__(self, max_per_customer: int = 1_000) -> None:
        self._max = max_per_customer
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ── Mutations ────────────────────────────────────────────────────

    def add(self, customer_id: str, record: dict[str, Any] | TransactionRecord) -> None:
        """Record a transaction for *customer_id*.

        Accepts both plain dicts and :class:`TransactionRecord`
        instances.  A defensive copy is always stored.

        If the list exceeds ``max_per_customer``, the oldest entry is
        removed (FIFO eviction).
        """
        if isinstance(record, TransactionRecord):
            record = record.to_dict()
        with self._lock:
            entries = self._data.setdefault(customer_id, [])
            entries.append(dict(record))  # defensive copy
            if len(entries) > self._max:
                self._data[customer_id] = entries[-self._max:]

    def record_outcome(
        self,
        customer_id: str,
        timestamp: int,
        is_fraud: int,
    ) -> bool:
        """Update the ``is_fraud`` label for a previously recorded transaction.

        Matches by *customer_id* and *timestamp*.  Returns ``True`` if
        the record was found and updated, ``False`` otherwise.
        """
        with self._lock:
            entries = self._data.get(customer_id, [])
            for entry in reversed(entries):
                if entry.get("timestamp") == timestamp:
                    entry["is_fraud"] = int(is_fraud)
                    return True
        return False

    # ── Reads ────────────────────────────────────────────────────────

    def get(
        self,
        customer_id: str,
        *,
        before_timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical transactions for *customer_id*.

        Args:
            customer_id: Customer identifier.
            before_timestamp: If given, only return entries with
                              ``timestamp < before_timestamp``
                              (temporal safety).

        Returns:
            List of history records (dicts), sorted by timestamp
            ascending.  Empty list if the customer has no history.
        """
        with self._lock:
            entries = self._data.get(customer_id, [])
            if not entries:
                return []
            result = list(entries)  # defensive copy

        if before_timestamp is not None:
            result = [e for e in result if e.get("timestamp", 0) < before_timestamp]

        # Sort by timestamp ascending (for temporal ordering)
        result.sort(key=lambda e: e.get("timestamp", 0))
        return result

    def customer_count(self, customer_id: str) -> int:
        """Return the number of stored entries for *customer_id*."""
        with self._lock:
            return len(self._data.get(customer_id, []))

    def total_count(self) -> int:
        """Return total entries across all customers."""
        with self._lock:
            return sum(len(v) for v in self._data.values())

    def clear(self) -> None:
        """Remove all stored history (useful in tests)."""
        with self._lock:
            self._data.clear()


# Backward-compatible alias used by existing code and tests.
TransactionHistoryStore = InMemoryHistoryStore


# ── SQLite-backed implementation ────────────────────────────────────


class SQLiteHistoryRepository:
    """Persistent history store backed by SQLite.

    Implements :class:`CustomerHistoryRepository`.  Uses WAL journal
    mode for better concurrent read performance and creates the table
    automatically on first use.

    Thread safety is achieved via :class:`threading.Lock` — a single
    shared connection (``check_same_thread=False``) is reused across
    all operations.

    Args:
        db_path: Path to the SQLite database file.  Parent directories
                 are created automatically.
        max_per_customer: Maximum entries kept per customer.  Oldest
                          entries are evicted when the limit is reached.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS transaction_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT    NOT NULL,
            timestamp  INTEGER NOT NULL,
            amount     REAL    NOT NULL DEFAULT 0.0,
            product_cd TEXT,
            addr1      INTEGER,
            addr2      INTEGER,
            device_type TEXT,
            id_19      TEXT,
            id_20      TEXT,
            has_identity_data INTEGER NOT NULL DEFAULT 0,
            is_fraud   INTEGER NOT NULL DEFAULT 0
        )
    """
    _INDEX = """
        CREATE INDEX IF NOT EXISTS ix_history_customer_ts
        ON transaction_history (customer_id, timestamp)
    """

    def __init__(
        self,
        db_path: str | Path = "data/ml_history.db",
        *,
        max_per_customer: int = 1_000,
    ) -> None:
        self._max = max_per_customer
        self._lock = threading.Lock()

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode: concurrent readers, single writer, survives crashes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self._SCHEMA)
        self._conn.execute(self._INDEX)
        self._conn.commit()

    # ── Mutations ────────────────────────────────────────────────────

    def add(self, customer_id: str, record: dict[str, Any] | TransactionRecord) -> None:
        """Record a transaction for *customer_id*.

        Accepts both plain dicts and :class:`TransactionRecord`.
        """
        if isinstance(record, TransactionRecord):
            record = record.to_dict()

        with self._lock:
            self._conn.execute(
                "INSERT INTO transaction_history "
                "(customer_id, timestamp, amount, product_cd, addr1, "
                "addr2, device_type, id_19, id_20, has_identity_data, "
                "is_fraud) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(customer_id),
                    int(record.get("timestamp", 0)),
                    float(record.get("amount", 0.0)),
                    record.get("product_cd"),
                    record.get("addr1"),
                    record.get("addr2"),
                    record.get("device_type"),
                    record.get("id_19"),
                    record.get("id_20"),
                    int(record.get("has_identity_data", 0)),
                    int(record.get("is_fraud", 0)),
                ),
            )
            self._conn.commit()

            # FIFO eviction when per-customer cap is exceeded.
            count = self._conn.execute(
                "SELECT COUNT(*) FROM transaction_history WHERE customer_id = ?",
                (str(customer_id),),
            ).fetchone()[0]
            if count > self._max:
                overflow = count - self._max
                self._conn.execute(
                    "DELETE FROM transaction_history WHERE id IN ("
                    "  SELECT id FROM transaction_history"
                    "  WHERE customer_id = ?"
                    "  ORDER BY timestamp ASC, id ASC"
                    "  LIMIT ?"
                    ")",
                    (str(customer_id), overflow),
                )
                self._conn.commit()

    def record_outcome(
        self,
        customer_id: str,
        timestamp: int,
        is_fraud: int,
    ) -> bool:
        """Update ``is_fraud`` for a previously recorded transaction.

        Matches by *customer_id* and *timestamp*.  Returns ``True`` if
        at least one record was updated.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE transaction_history SET is_fraud = ? "
                "WHERE customer_id = ? AND timestamp = ?",
                (int(is_fraud), str(customer_id), int(timestamp)),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ── Reads ────────────────────────────────────────────────────────

    def get(
        self,
        customer_id: str,
        *,
        before_timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical transactions for *customer_id*.

        Only entries with ``timestamp < before_timestamp`` are returned
        (temporal safety).  Results sorted by timestamp ascending.
        """
        with self._lock:
            if before_timestamp is not None:
                rows = self._conn.execute(
                    "SELECT timestamp, amount, product_cd, addr1, addr2, "
                    "device_type, id_19, id_20, has_identity_data, is_fraud "
                    "FROM transaction_history "
                    "WHERE customer_id = ? AND timestamp < ? "
                    "ORDER BY timestamp ASC, id ASC",
                    (str(customer_id), int(before_timestamp)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT timestamp, amount, product_cd, addr1, addr2, "
                    "device_type, id_19, id_20, has_identity_data, is_fraud "
                    "FROM transaction_history "
                    "WHERE customer_id = ? "
                    "ORDER BY timestamp ASC, id ASC",
                    (str(customer_id),),
                ).fetchall()

        return [dict(row) for row in rows]

    def customer_count(self, customer_id: str) -> int:
        """Return the number of stored entries for *customer_id*."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM transaction_history WHERE customer_id = ?",
                (str(customer_id),),
            ).fetchone()[0]

    def total_count(self) -> int:
        """Return total entries across all customers."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM transaction_history"
            ).fetchone()[0]

    def clear(self) -> None:
        """Remove all stored history (useful in tests)."""
        with self._lock:
            self._conn.execute("DELETE FROM transaction_history")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


# ── Recording helper ────────────────────────────────────────────────


def record_transaction(
    store: CustomerHistoryRepository,
    raw: dict[str, Any],
    *,
    customer_id: str | None = None,
) -> None:
    """Record a raw transaction in the history store.

    Extracts the fields needed by the historical feature pipeline and
    stores them under the resolved customer identifier.

    This function is intentionally **best-effort** — callers should
    wrap it in try/except so a recording failure never blocks
    prediction.

    Args:
        store: History repository instance.
        raw: Raw transaction dict (e.g. ``request.model_dump()``).
        customer_id: Pre-resolved customer ID.  If ``None``, uses
                     ``raw["customer_id"]`` or ``raw["device_fingerprint"]``.
    """
    if customer_id is None:
        from ml.features.engineer import _resolve_customer_id

        customer_id = _resolve_customer_id(raw)

    def _int(key: str, default: int) -> int:
        v = raw.get(key)
        return default if v is None else int(v)

    def _str(key: str, default: str | None) -> str | None:
        v = raw.get(key)
        return default if v is None else str(v)

    pc = _str("ProductCD", None) or _str("merchant_category", "W")
    # DeviceType (dataset convention) with device_type fallback (backend convention)
    dt = _str("DeviceType", None) or _str("device_type", None)

    store.add(
        customer_id,
        TransactionRecord(
            timestamp=_int("timestamp", 0),
            amount=float(raw.get("amount", 0.0)),
            product_cd=pc,
            addr1=_int("addr1", -1),
            addr2=_int("addr2", -1),
            device_type=dt,
            id_19=_str("id_19", None),
            id_20=_str("id_20", None),
            has_identity_data=_int("has_identity_data", 0),
            is_fraud=0,  # unknown at prediction time
        ),
    )


# ── Module-level singleton ──────────────────────────────────────────

history_store: InMemoryHistoryStore | SQLiteHistoryRepository = InMemoryHistoryStore()
"""Global history store instance shared across the ML service.

Defaults to :class:`InMemoryHistoryStore`.  The ML API lifespan handler
in ``ml/api/app.py`` replaces this with :class:`SQLiteHistoryRepository`
when the SQLite database is writable.
"""
