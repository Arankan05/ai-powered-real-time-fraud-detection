"""Customer endpoints.

* ``GET /customers/me``             — **Auth (customer)** — Own profile.
* ``GET /customers/{id}``           — **Auth** — Customers see own; analysts/admins see any.
* ``GET /customers/{id}/transactions`` — **Auth** — Customer transaction history.
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.transaction import TransactionQueryParams
from app.services.auth import get_current_user
from app.services.customer import CustomerService
from app.services.transaction import TransactionService

router = APIRouter()


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Get the current customer's own profile."""
    service = CustomerService(db)
    result = service.get_me(current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/{customer_id}")
async def get_customer(
    customer_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Get a customer profile by ID."""
    service = CustomerService(db)
    result = service.get_by_id(customer_id, current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/{customer_id}/transactions")
async def get_customer_transactions(
    customer_id: UUID,
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    risk_level: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Get transaction history for a specific customer."""
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
    result = service.list_customer_transactions(customer_id, current_user, params)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
