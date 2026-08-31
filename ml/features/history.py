"""Customer history repository for real-time fraud features.

Provides:

* :class:`TransactionRecord` — typed dataclass for a single historical
  transaction.
* :class:`CustomerHistoryRepository` — abstract protocol defining the
  repository contract (for future database replacement).
* :class:`InMemoryHistoryStore` — default in-memory implementation of
  the protocol.
* :func:`record_transaction` — helper that creates a history record
  from a raw transaction dict after a successful prediction.

Design rationale
----------------
No database persistence layer exists yet in the project.  This module
provides the simplest implementation that fits the architecture:

* Transactions are recorded after each successful prediction.
* Lookups return only entries with ``timestamp < current_timestamp``
  (temporal safety — never uses future data).
* ``is_fraud`` is stored as ``0`` at prediction time (the true label
  is unknown).  An optional :meth:`~InMemoryHistoryStore.record_outcome`
  method allows updating the label later when investigation results
  are available.
* The store is capped at ``max_per_customer`` entries per customer
  (FIFO eviction) to bound memory usage.

Replacing with a database
-------------------------
To swap the in-memory store for PostgreSQL (or another backend):

1. Implement :class:`CustomerHistoryRepository` with DB queries.
2. Replace the module-level :data:`history_store` singleton in
   ``ml/api/app.py`` with the DB-backed instance.
3. No changes to :mod:`ml.features.engineer` or the ML model are
   needed — the feature-engineering pipeline only depends on the
   protocol's ``get()`` / ``add()`` contract.

Thread safety
-------------
All mutations and reads in :class:`InMemoryHistoryStore` go through a
:class:`threading.Lock` so the store is safe for concurrent access
from FastAPI worker threads.

Usage::

    from ml.features.history import history_store, record_transaction

    # After a successful prediction:
    record_transaction(history_store, raw_data)

    # Retrieve history for a new transaction:
    history = history_store.get("customer_123", before_timestamp=90000)
"""

from __future__ import annotations

import dataclasses
import threading
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

    store.add(
        customer_id,
        TransactionRecord(
            timestamp=_int("timestamp", 0),
            amount=float(raw.get("amount", 0.0)),
            product_cd=pc,
            addr1=_int("addr1", -1),
            addr2=_int("addr2", -1),
            device_type=_str("DeviceType", None),
            id_19=_str("id_19", None),
            id_20=_str("id_20", None),
            has_identity_data=_int("has_identity_data", 0),
            is_fraud=0,  # unknown at prediction time
        ),
    )


# ── Module-level singleton ──────────────────────────────────────────

history_store: InMemoryHistoryStore = InMemoryHistoryStore()
"""Global history store instance shared across the ML service.

Replace this with a database-backed :class:`CustomerHistoryRepository`
implementation when persistent storage becomes available.
"""
