"""Health-check endpoint.

Reports backend availability **and** database connectivity.

* **HTTP 200** when all configured services are healthy.
* **HTTP 503** when any configured service is degraded.

The ML service is reported as ``not_configured`` until it is integrated
in a later task.
"""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)) -> JSONResponse:
    """Return backend and service health status."""

    # -- Database connectivity --
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        db_status = "disconnected"

    # -- ML service (not yet integrated) --
    ml_status = "not_configured"

    # -- Overall status --
    all_healthy = db_status == "connected"
    status = "healthy" if all_healthy else "degraded"
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "version": "0.1.0",
            "services": {
                "database": {"status": db_status},
                "ml_service": {"status": ml_status},
            },
        },
    )
