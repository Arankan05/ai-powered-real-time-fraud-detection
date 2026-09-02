"""FastAPI application entry-point.

Initialises the ML / Fraud Intelligence Service client, the alert
repository, and registers the transaction and alert routers.
Authentication, database (PostgreSQL), and other modules will be
added by the backend developer (Developer A).

Run locally::

    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.config import get_settings
from backend.db.alert_repository import SQLiteAlertRepository
from backend.routers.alerts import router as alerts_router, set_alert_repository
from backend.routers.transactions import (
    router as transactions_router,
    set_alert_repository as set_txn_alert_repo,
    set_ml_client,
)
from backend.services.ml_client import MLServiceClient

logger = logging.getLogger(__name__)

settings = get_settings()

# Module-level alert repository reference
_alert_repo: SQLiteAlertRepository | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: initialise ML client and alert store.  Shutdown: release."""
    global _alert_repo

    # ML service client
    client = MLServiceClient(
        base_url=settings.ML_SERVICE_URL,
        timeout=float(settings.ML_REQUEST_TIMEOUT_SECONDS),
    )
    set_ml_client(client)
    logger.info("ML client configured (timeout %ds)", settings.ML_REQUEST_TIMEOUT_SECONDS)

    # Alert persistence
    try:
        repo = SQLiteAlertRepository(db_path=settings.ALERT_DB_PATH)
        _alert_repo = repo
        set_alert_repository(repo)
        set_txn_alert_repo(repo)
        logger.info("Alert store: SQLite (%s)", settings.ALERT_DB_PATH)
    except Exception as exc:
        logger.warning("SQLite alert store unavailable (%s); alerts disabled", exc)

    yield

    # Shutdown
    if _alert_repo is not None:
        _alert_repo.close()


app = FastAPI(
    title="Fraud Detection API",
    description="Backend for the AI-Powered Fraud Detection System.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(transactions_router)
app.include_router(alerts_router)
