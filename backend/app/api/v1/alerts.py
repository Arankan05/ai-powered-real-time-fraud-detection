"""Alert endpoints.

* ``GET   /alerts``      — **Auth (fraud_analyst, admin)** — List alerts.
* ``GET   /alerts/{id}`` — **Auth (fraud_analyst, admin)** — Alert detail.
* ``PATCH /alerts/{id}`` — **Auth (fraud_analyst, admin)** — Update alert.
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.alert import AlertUpdateRequest
from app.services.alert import AlertService
from app.services.auth import require_role

router = APIRouter()


@router.get("")
async def list_alerts(
    page: int = 1,
    per_page: int = 20,
    status: Literal["OPEN", "IN_REVIEW", "RESOLVED", "DISMISSED"] | None = None,
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] | None = None,
    current_user: User = Depends(require_role("fraud_analyst", "admin")),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """List alerts with pagination and filters."""
    service = AlertService(db)
    result = service.list_alerts(
        current_user,
        page=page,
        per_page=per_page,
        status=status,
        risk_level=risk_level,
    )
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/{alert_id}")
async def get_alert(
    alert_id: UUID,
    current_user: User = Depends(require_role("fraud_analyst", "admin")),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Get a single alert with full transaction detail."""
    service = AlertService(db)
    result = service.get_alert(alert_id, current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.patch("/{alert_id}")
async def update_alert(
    alert_id: UUID,
    body: AlertUpdateRequest,
    current_user: User = Depends(require_role("fraud_analyst", "admin")),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Update alert status and/or notes."""
    service = AlertService(db)
    result = service.update_alert(alert_id, body, current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
