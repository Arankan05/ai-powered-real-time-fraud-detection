"""Database persistence layer for the backend.

Provides:

* :class:`~backend.db.alert_repository.AlertRepository` — abstract
  protocol for alert persistence.
* :class:`~backend.db.alert_repository.InMemoryAlertStore` — volatile
  in-memory implementation (tests).
* :class:`~backend.db.alert_repository.SQLiteAlertRepository` —
  persistent SQLite-backed implementation (local development opt-in).
* :class:`~backend.db.alert_repository.PostgresAlertRepository` —
  PostgreSQL / Supabase-backed implementation (production default).

* :class:`~backend.db.user_repository.UserRepository` — abstract
  protocol for user persistence.
* :class:`~backend.db.user_repository.InMemoryUserStore` — volatile
  in-memory implementation (tests).
* :class:`~backend.db.user_repository.SQLiteUserRepository` —
  persistent SQLite-backed implementation (local development opt-in).
* :class:`~backend.db.user_repository.PostgresUserRepository` —
  PostgreSQL / Supabase-backed implementation (production default).

* :mod:`backend.db.postgres` — shared PostgreSQL connection layer
  (connection pool, schema initialisation, sanitised errors).
* :mod:`backend.db.seed_users` — development seeding tool for
  ``fraud_analyst`` / ``admin`` accounts.

Production selection
--------------------
The ``PERSISTENCE_BACKEND`` environment variable chooses the storage
backend (``postgres`` by default).  See
``docs/database-design.md`` for the architectural rationale and
``.env.example`` for the connection variables.

ML feature history
------------------
The ML / Fraud Intelligence Service maintains its own SQLite-backed
transaction-history repository in ``ml/features/history.py``.  That
store is intentionally left on SQLite (zero external dependencies,
single-process service) — see its module docstring for the
replacement path.
"""
