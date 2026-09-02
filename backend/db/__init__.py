"""Database persistence layer for the backend.

Currently provides:

* :class:`AlertRepository` — abstract protocol for alert persistence.
* :class:`SQLiteAlertRepository` — persistent SQLite-backed implementation.
* :class:`InMemoryAlertStore` — volatile in-memory implementation (tests).

Design rationale
----------------
The backend plans PostgreSQL for full production use (see
``docs/database-design.md``).  That infrastructure is not yet
implemented.  Following the same pattern as ``ml/features/history.py``,
SQLite is the smallest appropriate persistence layer for the alert
system at this stage.

Replacing with PostgreSQL
-------------------------
1. Implement :class:`AlertRepository` with SQL queries against the
   planned ``alerts`` table.
2. Replace the singleton in ``backend/app.py``.
3. No changes to routers or schemas needed.
"""
