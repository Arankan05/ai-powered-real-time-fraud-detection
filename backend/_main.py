"""FastAPI application entry-point.

Initialises the ML / Fraud Intelligence Service client, the alert
repository, the user repository (JWT authentication), and registers
the authentication, transaction, and alert routers.

Persistence backend
-------------------
Controlled by the ``PERSISTENCE_BACKEND`` environment variable
(default ``postgres``):

* ``postgres`` — the production backend.  The application fails fast
  at startup when the PostgreSQL / Supabase server is unreachable or
  ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` are not configured
  (sanitised error, no credentials leaked).
* ``sqlite``   — lightweight local development without a database
  server; alerts and users live in the files pointed to by
  ``ALERT_DB_PATH`` / ``USER_DB_PATH``.

Internal roles (``fraud_analyst`` / ``admin``) can be provisioned for
development with::

    python -m backend.db.seed_users

Run locally::

    uvicorn backend._main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.config import get_settings
from backend.db.alert_repository import SQLiteAlertRepository
from backend.db.user_repository import SQLiteUserRepository
from backend.routers.alerts import router as alerts_router, set_alert_repository
from backend.routers.auth import router as auth_router
from backend.routers.transactions import (
    router as transactions_router,
    set_alert_repository as set_txn_alert_repo,
    set_ml_client,
)
from backend.security.deps import set_user_repository
from backend.services.ml_client import MLServiceClient

logger = logging.getLogger(__name__)

settings = get_settings()

# Module-level repository references — populated by lifespan()
_alert_repo = None
_user_repo = None
_pg_pool = None  # set when PERSISTENCE_BACKEND=postgres


def _init_sqlite() -> None:
    """Set up the lightweight SQLite-backed repositories."""
    global _alert_repo, _user_repo
    try:
        repo = SQLiteAlertRepository(db_path=settings.ALERT_DB_PATH)
        _alert_repo = repo
        set_alert_repository(repo)
        set_txn_alert_repo(repo)
        logger.info("Alert store: SQLite (%s)", settings.ALERT_DB_PATH)
    except Exception as exc:
        logger.warning("SQLite alert store unavailable (%s); alerts disabled", exc)

    try:
        users = SQLiteUserRepository(db_path=settings.USER_DB_PATH)
        _user_repo = users
        set_user_repository(users)
        logger.info("User store: SQLite (%s)", settings.USER_DB_PATH)
    except Exception as exc:
        logger.warning("SQLite user store unavailable (%s); authentication disabled", exc)


def _init_postgres() -> None:
    """Set up the PostgreSQL-backed repositories (fail-fast).

    On any configuration or connection error, a :class:`RuntimeError`
    with a *sanitised* message (no credentials, no full connection
    string) is raised — uvicorn logs it and exits without ever serving
    requests.
    """
    global _alert_repo, _user_repo, _pg_pool

    # Imported lazily so that SQLite-only runs never pull in psycopg
    # (useful for slim CI images that only run the Step-39 test suite).
    from backend.db.alert_repository import PostgresAlertRepository
    from backend.db.postgres import (
        PostgresConfigError,
        PostgresConnectionError,
        create_pool,
    )
    from backend.db.user_repository import PostgresUserRepository

    try:
        _pg_pool = create_pool(settings)
    except (PostgresConfigError, PostgresConnectionError) as exc:
        # Message is already sanitised — no password or connection string.
        raise RuntimeError(
            f"PostgreSQL persistence is required but unavailable: {exc}"
        ) from None

    _alert_repo = PostgresAlertRepository(_pg_pool)
    _user_repo = PostgresUserRepository(_pg_pool)
    set_alert_repository(_alert_repo)
    set_txn_alert_repo(_alert_repo)
    set_user_repository(_user_repo)
    logger.info(
        "Persistence: PostgreSQL (host=%s port=%s db=%s)",
        settings.POSTGRES_HOST, settings.POSTGRES_PORT, settings.POSTGRES_DB,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: initialise ML client, persistence backend, and alert store."""
    # ML service client
    client = MLServiceClient(
        base_url=settings.ML_SERVICE_URL,
        timeout=float(settings.ML_REQUEST_TIMEOUT_SECONDS),
    )
    set_ml_client(client)
    logger.info("ML client configured (timeout %ds)", settings.ML_REQUEST_TIMEOUT_SECONDS)

    backend_name = (settings.PERSISTENCE_BACKEND or "postgres").strip().lower()
    if backend_name == "postgres":
        _init_postgres()
    elif backend_name == "sqlite":
        _init_sqlite()
    else:
        raise RuntimeError(
            f"Unknown PERSISTENCE_BACKEND {backend_name!r}: expected 'postgres' or 'sqlite'"
        )

    yield

    # Shutdown: release resources cleanly (pool close is idempotent).
    if _pg_pool is not None:
        try:
            _pg_pool.close()
        except Exception:
            pass
    else:
        for repo in (_alert_repo, _user_repo):
            if repo is not None and hasattr(repo, "close"):
                try:
                    repo.close()
                except Exception:
                    pass


app = FastAPI(
    title="Fraud Detection API",
    description="Backend for the AI-Powered Fraud Detection System.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(transactions_router)
app.include_router(alerts_router)
