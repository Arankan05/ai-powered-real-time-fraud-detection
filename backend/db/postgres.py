"""PostgreSQL connection layer for the backend persistence repositories.

Shared infrastructure for the PostgreSQL / Supabase persistence phase
(Step 40):

* :func:`create_pool` — build a validated, fail-fast connection pool
  from the application settings (``POSTGRES_*`` environment variables).
* :func:`init_schema` — idempotently create the ``users`` and ``alerts``
  tables and their indexes (the project's schema initialisation
  mechanism until an Alembic environment is introduced).
* :class:`PostgresConfigError` / :class:`PostgresConnectionError` —
  errors whose messages are safe to log or surface to operators: they
  never contain the password or the full connection string.

Design rationale
----------------
The backend's :class:`~backend.db.user_repository.UserRepository` and
:class:`~backend.db.alert_repository.AlertRepository` protocols are
storage-agnostic — they accept any implementation with matching method
signatures.  The PostgreSQL implementations here use
:mod:`psycopg` 3 (binary build, sync mode) because:

* Sync API matches the existing repository pattern (single
  ``threading.Lock`` + connection in the SQLite repos; raw parameterised
  SQL).
* Native JSONB, UUID, and TIMESTAMPTZ adaptation keeps the wire
  protocol clean without bespoke serialisation layers.
* ``sslmode`` in the ``conninfo`` string supports both the local
  ``docker-compose`` PostgreSQL (``prefer``) and Supabase (``require``)
  with no code change.
* ``psycopg_pool`` handles thread-safe connection checkout cleanly —
  one pool is shared between the user and alert repositories.

Replacing SQLite
----------------
1. Set ``PERSISTENCE_BACKEND=postgres`` (the default).
2. Provide ``POSTGRES_HOST`` / ``POSTGRES_PORT`` / ``POSTGRES_DB`` /
   ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` (and ``POSTGRES_SSL_MODE``
   for managed services).
3. ``backend/app.py`` constructs the pool and hands it to
   :class:`PostgresUserRepository` / :class:`PostgresAlertRepository`,
   which automatically run :func:`init_schema` on first construction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from psycopg import conninfo as pg_conninfo
from psycopg_pool import ConnectionPool

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Default timeout (seconds) for the initial pool.open() check.
CONNECT_TIMEOUT_SECONDS: float = 10.0

# Regex that scrubs any stray ``password=…`` fragments from libpq /
# psycopg error messages before the message is logged or returned.
_PASSWORD_RE = re.compile(r"password=[^\s]*", re.IGNORECASE)


# ── Exceptions ────────────────────────────────────────────────────────


class PostgresConfigError(RuntimeError):
    """PostgreSQL configuration is missing or malformed.

    The message is safe to log / surface: it never contains the
    password or the full connection string.
    """


class PostgresConnectionError(RuntimeError):
    """PostgreSQL server is unreachable or rejected the connection.

    The message is safe to log / surface: it never contains the
    password or the full connection string.
    """


# ── Schema DDL ────────────────────────────────────────────────────────
# Idempotent CREATE statements executed by :func:`init_schema`.
# These statements are the project's schema-initialisation mechanism
# (see ``docs/database-design.md``).


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # ── users ─────────────────────────────────────────────────────────
    """\
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY,
    email         VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'customer'
                  CHECK (role IN ('customer', 'fraud_analyst', 'admin')),
    first_name    VARCHAR(100),
    last_name     VARCHAR(100),
    phone         VARCHAR(30),
    date_of_birth DATE,
    address       VARCHAR(255),
    customer_id   UUID,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL
)""",
    # Case-insensitive email uniqueness (replaces SQLite's ``COLLATE NOCASE``).
    # Stored values are lowercased by the repository; the index guards against
    # mixed-case inserts from any other writer.
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_ci ON users (lower(email))",
    "CREATE INDEX IF NOT EXISTS ix_users_role ON users (role)",

    # ── alerts ────────────────────────────────────────────────────────
    """\
CREATE TABLE IF NOT EXISTS alerts (
    id                UUID PRIMARY KEY,
    transaction_id    UUID NOT NULL,
    customer_id       UUID,
    amount            DOUBLE PRECISION,
    currency          VARCHAR(3),
    merchant_name     VARCHAR(255),
    transaction_type  VARCHAR(20),
    timestamp         BIGINT,
    risk_score        INTEGER NOT NULL,
    risk_level        VARCHAR(10) NOT NULL DEFAULT 'HIGH',
    decision          VARCHAR(10) NOT NULL DEFAULT 'HOLD',
    fraud_probability DOUBLE PRECISION,
    model_version     VARCHAR(100),
    risk_factors      JSONB,
    explanation_json  JSONB,
    status            VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    analyst_id        UUID REFERENCES users(id),
    notes             TEXT,
    created_at        TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL,
    resolved_at       TIMESTAMPTZ
)""",
    # One alert per transaction (router-level protection is the primary
    # guard; the unique index is the secondary / race-condition guard).
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_alerts_transaction_id ON alerts (transaction_id)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts (status)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_risk_level ON alerts (risk_level)",
    # Default ``ORDER BY created_at DESC`` in ``list_alerts`` — the index
    # avoids a filesort on every page fetch.
    "CREATE INDEX IF NOT EXISTS ix_alerts_created_at ON alerts (created_at DESC)",

    # ── promotion_governance (Step 50) ─────────────────────────────
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
    created_at                 TIMESTAMPTZ NOT NULL,
    updated_at                 TIMESTAMPTZ NOT NULL
)""",
    "CREATE INDEX IF NOT EXISTS ix_promo_gov_status ON promotion_governance (governance_status)",
    "CREATE INDEX IF NOT EXISTS ix_promo_gov_candidate ON promotion_governance (candidate_model_version, candidate_checksum)",
    "CREATE INDEX IF NOT EXISTS ix_promo_gov_created_at ON promotion_governance (created_at DESC)",
    """\
CREATE UNIQUE INDEX IF NOT EXISTS uq_promo_gov_candidate_decision
    ON promotion_governance (candidate_model_version, candidate_checksum, gate_decision)""",
)


# ── Connection pool ───────────────────────────────────────────────────


def _require_setting(settings: Settings, name: str) -> str:
    value = getattr(settings, name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PostgresConfigError(
            f"{name} is not configured; set it in the environment or .env"
        )
    return value


def _sanitise(exc: Exception, *, host: str, port: int, dbname: str) -> str:
    """Build a log-safe description of a connection failure.

    Includes host/port/db (non-credentials, useful for operators) and
    the exception class, but never the password or connection string.
    """
    detail = str(exc).strip() or type(exc).__name__
    detail = _PASSWORD_RE.sub("password=***", detail)
    # Truncate extremely long messages (some drivers embed the query).
    if len(detail) > 400:
        detail = detail[:397] + "..."
    return f"connection to PostgreSQL at {host}:{port}/{dbname} failed ({type(exc).__name__}: {detail})"


def create_pool(
    settings: Settings | None = None,
    *,
    min_size: int = 1,
    max_size: int = 5,
    timeout: float = CONNECT_TIMEOUT_SECONDS,
    open_pool: bool = True,
    pool_kwargs: dict[str, Any] | None = None,
) -> ConnectionPool:
    """Build a PostgreSQL connection pool from the application settings.

    Raises
    ------
    PostgresConfigError
        If ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` are missing or the
        pool cannot be constructed.
    PostgresConnectionError
        If the server cannot be reached within ``timeout`` seconds
        (only when ``open_pool`` is ``True``).

    Notes
    -----
    The password and connection string are never logged or included in
    any error message raised by this function.
    """
    settings = settings or get_settings()

    user = _require_setting(settings, "POSTGRES_USER")
    password = _require_setting(settings, "POSTGRES_PASSWORD")
    host = settings.POSTGRES_HOST or "localhost"
    port = int(settings.POSTGRES_PORT)
    dbname = settings.POSTGRES_DB or "fraud_detection"
    ssl_mode = (settings.POSTGRES_SSL_MODE or "").strip()

    conn_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
        "connect_timeout": max(int(timeout), 1),
    }
    if ssl_mode:
        conn_kwargs["sslmode"] = ssl_mode

    try:
        conninfo = pg_conninfo.make_conninfo(**{k: v for k, v in conn_kwargs.items() if v is not None})
    except Exception as exc:
        raise PostgresConfigError(
            f"invalid PostgreSQL connection parameters: {type(exc).__name__}"
        ) from None

    kwargs = dict(pool_kwargs or {})
    try:
        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_size,
            max_size=max_size,
            name="fraud-detection-pool",
            timeout=timeout,
            open=False,
            **kwargs,
        )
    except Exception as exc:
        raise PostgresConfigError(
            f"invalid pool configuration: {type(exc).__name__}: {_PASSWORD_RE.sub('password=***', str(exc))}"
        ) from None

    if open_pool:
        try:
            pool.open(wait=True, timeout=timeout)
        except Exception as exc:
            # Best-effort cleanup of pool worker threads.
            try:
                pool.close()
            except Exception:
                pass
            raise PostgresConnectionError(_sanitise(exc, host=host, port=port, dbname=dbname)) from None

        # Quick sanity ping — a pool with ``min_size=0`` would ``open``
        # successfully without actually connecting; this forces a real
        # connection to surface bad credentials / unreachable hosts
        # deterministically.
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception as exc:
            try:
                pool.close()
            except Exception:
                pass
            raise PostgresConnectionError(_sanitise(exc, host=host, port=port, dbname=dbname)) from None

    return pool


def init_schema(pool: ConnectionPool) -> None:
    """Create the ``users`` and ``alerts`` tables (idempotent).

    Uses the current ``search_path`` on the pool's connections, so
    callers can target an isolated schema for tests by setting
    ``options="-c search_path=<schema>"`` on the pool.
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for stmt in _SCHEMA_STATEMENTS:
                cur.execute(stmt)
    # pool.connection() context commits on clean exit.
