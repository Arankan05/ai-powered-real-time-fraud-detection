"""Transaction endpoints.

* ``POST /transactions``       — **Auth (customer)** — Create + fraud detection.
* ``GET  /transactions``       — **Auth** — List (customers: own; analysts/admins: all).
* ``GET  /transactions/{id}``  — **Auth** — Detail (customers: own; analysts/admins: any).
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.transaction import (
    TransactionCreateRequest,
    TransactionQueryParams,
)
from app.services.auth import get_current_user
from app.services.transaction import TransactionService

router = APIRouter()


@router.post("", status_code=201)
async def create_transaction(
    body: TransactionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Submit a transaction with automatic fraud detection."""
    service = TransactionService(db)
    result = service.create_transaction(body, current_user)
    return JSONResponse(
        status_code=201,
        content=result.model_dump(mode="json"),
    )


@router.get("")
async def list_transactions(
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    risk_level: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """List transactions with pagination and filters."""
    from datetime import datetime

    params = TransactionQueryParams(
        page=page,
        per_page=per_page,
        status=status,
        risk_level=risk_level,
        from_date=datetime.fromisoformat(from_date) if from_date else None,
        to_date=datetime.fromisoformat(to_date) if to_date else None,
    )

    service = TransactionService(db)
    result = service.list_transactions(current_user, params)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Get a single transaction with full details."""
    service = TransactionService(db)
    result = service.get_transaction(transaction_id, current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
