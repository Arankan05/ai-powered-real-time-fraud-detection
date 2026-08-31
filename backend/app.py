"""FastAPI application entry-point.

Initialises the ML / Fraud Intelligence Service client and registers
the transaction router.  Authentication, database, and other modules
will be added by the backend developer (Developer A).

Run locally::

    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.config import get_settings
from backend.routers.transactions import router as transactions_router, set_ml_client
from backend.services.ml_client import MLServiceClient

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: initialise ML service client.  Shutdown: release."""
    client = MLServiceClient(
        base_url=settings.ML_SERVICE_URL,
        timeout=float(settings.ML_REQUEST_TIMEOUT_SECONDS),
    )
    set_ml_client(client)
    logger.info("ML client configured (timeout %ds)", settings.ML_REQUEST_TIMEOUT_SECONDS)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Backend for the AI-Powered Fraud Detection System.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(transactions_router)
