"""Fraud check endpoint.

* ``POST /fraud/check`` — **Auth (fraud_analyst, admin)** — Run fraud analysis
  without persisting.  Used by analysts to test scenarios.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.transaction import FraudCheckRequest
from app.services.auth import require_role
from app.services.transaction import TransactionService

router = APIRouter()


@router.post("/check")
async def fraud_check(
    body: FraudCheckRequest,
    current_user: User = Depends(require_role("fraud_analyst", "admin")),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Run fraud analysis on a payload without creating a transaction."""
    service = TransactionService(db)
    result = service.fraud_check(body, current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
