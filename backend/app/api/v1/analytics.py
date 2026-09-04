"""Analytics endpoint.

* ``GET /analytics/dashboard`` — **Auth (fraud_analyst, admin)** — Aggregated
  metrics for the fraud analyst dashboard.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.services.analytics import AnalyticsService
from app.services.auth import require_role

router = APIRouter()


@router.get("/dashboard")
async def dashboard(
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    current_user: User = Depends(require_role("fraud_analyst", "admin")),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Return aggregated analytics for the fraud analyst dashboard."""
    service = AnalyticsService(db)
    result = service.get_dashboard(from_date=from_date, to_date=to_date)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
