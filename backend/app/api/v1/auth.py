"""Authentication endpoints.

All routes match ``docs/api-contract.md`` exactly.

* ``POST /auth/register``  — **Public** — Create customer account.
* ``POST /auth/login``     — **Public** — Authenticate.
* ``POST /auth/refresh``   — **Auth required** — Refresh tokens.
* ``GET  /auth/me``        — **Auth required** — Current user profile.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.auth import (
    LoginRequest,
    MeResponse,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from app.services.auth import AuthService, get_current_user
from app.models.user import User

router = APIRouter()


@router.post("/register", status_code=201)
async def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Create a new user account with ``customer`` role."""
    service = AuthService(db)
    result: RegisterResponse = service.register(body)
    return JSONResponse(
        status_code=201,
        content=result.model_dump(mode="json"),
    )


@router.post("/login")
async def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Authenticate and receive JWT tokens."""
    service = AuthService(db)
    result: TokenResponse = service.login(body)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.post("/refresh")
async def refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Refresh an expiring access token."""
    service = AuthService(db)
    result: TokenResponse = service.refresh(body)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/me")
async def me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Return the currently authenticated user's profile."""
    service = AuthService(db)
    result: MeResponse = service.get_me(current_user)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )
