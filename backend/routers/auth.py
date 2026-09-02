"""Authentication router — register, login, refresh, and me endpoints.

Implements the ``/api/v1/auth/*`` endpoints from the API contract:

* ``POST /api/v1/auth/register`` — public; always creates a ``customer``
  account (roles cannot be chosen by clients).
* ``POST /api/v1/auth/login`` — public; issues a JWT access + refresh
  token pair.
* ``POST /api/v1/auth/refresh`` — exchanges a valid refresh token for a
  new token pair.
* ``GET /api/v1/auth/me`` — returns the authenticated user's profile.

Security notes
--------------
* Login returns the same error for unknown emails and wrong passwords
  (no account enumeration).
* Passwords and password hashes are never returned in any response.
* Internal roles (``fraud_analyst``, ``admin``) are provisioned with
  ``python -m backend.db.seed_users``, never through public
  registration.
"""

from __future__ import annotations

import logging
from typing import Any

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, status

from backend.config import get_settings
from backend.schemas import (
    LoginRequest,
    MeResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.security.deps import get_current_user, get_user_repository
from backend.security.jwt_utils import (
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.security.passwords import verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["auth"])


def _normalise_email(email: str) -> str:
    """Lower-case and strip an email for case-insensitive storage."""
    return email.strip().lower()


def _issue_tokens(user: dict[str, Any]) -> TokenResponse:
    """Build a fresh access + refresh token pair for a user."""
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user_id=user["id"], role=user["role"]),
        refresh_token=create_refresh_token(user_id=user["id"], role=user["role"]),
        token_type="bearer",
        expires_in=settings.BACKEND_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(request: RegisterRequest) -> UserResponse:
    """Register a new user account with the ``customer`` role.

    Any ``role`` field supplied by the client is ignored — privilege
    escalation through registration is not possible.
    """
    repo = get_user_repository()
    email = _normalise_email(request.email)

    if repo.email_exists(email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = repo.create_user(
        email=email,
        password=request.password,
        role="customer",
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone,
        date_of_birth=request.date_of_birth,
        address=request.address,
    )
    logger.info("User registered: id=%s role=customer", user["id"])

    return UserResponse(
        id=user["id"],
        email=user["email"],
        first_name=user.get("first_name"),
        last_name=user.get("last_name"),
        role=user["role"],
        customer_id=user.get("customer_id"),
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate with email + password and receive JWT tokens."""
    repo = get_user_repository()
    user = repo.get_by_email(_normalise_email(request.email))

    # Same error for unknown email and wrong password — prevents
    # account enumeration.
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive.",
        )

    logger.info("User login: id=%s", user["id"])
    return _issue_tokens(user)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest) -> TokenResponse:
    """Exchange a valid refresh token for a new token pair."""
    repo = get_user_repository()

    try:
        payload = decode_token(request.refresh_token)
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    if payload.get("type") != REFRESH_TOKEN_TYPE:
        # Access tokens must not be used as refresh tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = repo.get_by_id(payload.get("sub", ""))
    if user is None or not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    return _issue_tokens(user)


@router.get("/auth/me", response_model=MeResponse)
async def me(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> MeResponse:
    """Return the currently authenticated user's profile."""
    return MeResponse(
        id=current_user["id"],
        email=current_user["email"],
        first_name=current_user.get("first_name"),
        last_name=current_user.get("last_name"),
        role=current_user["role"],
        customer_id=current_user.get("customer_id"),
        is_active=bool(current_user.get("is_active", False)),
        created_at=current_user.get("created_at", ""),
    )
