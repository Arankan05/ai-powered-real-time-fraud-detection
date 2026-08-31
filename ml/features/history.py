"""In-memory transaction history store for customer-level features.

Provides :class:`TransactionHistoryStore` — a thread-safe, in-memory
repository of past transactions keyed by customer identifier.  Used by
:func:`ml.features.engineer.engineer_features_for_inference` to supply
real historical context for feature engineering instead of cold-start
defaults.

Design rationale
----------------
No database persistence layer exists yet in the project.  This module
provides the simplest implementation that fits the architecture:

* Transactions are recorded after each successful prediction.
* Lookups return only entries with ``timestamp < current_timestamp``
  (temporal safety — never uses future data).
* ``is_fraud`` is stored as ``0`` at prediction time (the true label
  is unknown).  An optional :meth:`record_outcome` method allows
  updating the label later when investigation results are available.
* The store is capped at ``max_per_customer`` entries per customer
  (FIFO eviction) to bound memory usage.

This module is intentionally lightweight and dependency-free beyond
the standard library.  When a database becomes available, this class
can be replaced with a DB-backed implementation without changing the
calling code.

Thread safety
-------------
All mutations and reads go through a :class:`threading.Lock` so the
store is safe for concurrent access from the FastAPI worker threads.

Usage::

    from ml.features.history import history_store

    # After a successful prediction:
    history_store.add("customer_123", {
        "timestamp": 86400,
        "amount": 150.0,
        "product_cd": "W",
        "addr1": 100,
        "addr2": 200,
        "device_type": "mobile",
        "id_19": "val1",
        "id_20": "val2",
    })

    # Retrieve history for a new transaction:
    history = history_store.get("customer_123", before_timestamp=90000)
"""

from __future__ import annotations

import threading
from typing import Any


# ── Record type ───────────────────────────────────────────────────────

HistoryRecord = dict[str, Any]
"""A single historical transaction record.

Expected keys:
    ``timestamp``   — int, TransactionDT (seconds from reference)
    ``amount``      — float, TransactionAmt
    ``product_cd``  — str | None, ProductCD
    ``addr1``       — int | None, region code
    ``addr2``       — int | None, country code
    ``device_type`` — str | None, DeviceType
    ``id_19``       — str | None, identity field
    ``id_20``       — str | None, identity field
    ``is_fraud``    — int, 0 or 1 (0 at prediction time)
"""


# ── Store ─────────────────────────────────────────────────────────────


class TransactionHistoryStore:
    """Thread-safe, in-memory transaction history keyed by customer ID.

    Args:
        max_per_customer: Maximum entries kept per customer.  Oldest
                          entries are evicted when the limit is reached.
    """

    def __init__(self, max_per_customer: int = 1_000) -> None:
        self._max = max_per_customer
        self._data: dict[str, list[HistoryRecord]] = {}
        self._lock = threading.Lock()

    # ── Mutations ────────────────────────────────────────────────────

    def add(self, customer_id: str, record: HistoryRecord) -> None:
        """Record a transaction for *customer_id*.

        The record is appended to the customer's history list.  If the
        list exceeds ``max_per_customer``, the oldest entry is removed.
        """
        with self._lock:
            entries = self._data.setdefault(customer_id, [])
            entries.append(dict(record))  # defensive copy
            if len(entries) > self._max:
                # Evict oldest
                self._data[customer_id] = entries[-self._max :]

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
    ) -> list[HistoryRecord]:
        """Return historical transactions for *customer_id*.

        Args:
            customer_id: Customer identifier.
            before_timestamp: If given, only return entries with
                              ``timestamp < before_timestamp``
                              (temporal safety).

        Returns:
            List of history records, sorted by timestamp ascending.
            Empty list if the customer has no history.
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


# ── Module-level singleton ────────────────────────────────────────────

history_store = TransactionHistoryStore()
"""Global history store instance shared across the ML service."""
