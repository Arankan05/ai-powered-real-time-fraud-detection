"""Authentication business logic, service layer, and FastAPI dependencies.

Contains:

* ``AuthService`` — register, login, refresh, get-current-user.
* FastAPI dependencies — ``get_current_user``, ``require_role``.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import ForbiddenException, UnauthorizedException
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.models.user import User
from app.repositories.auth import UserRepository
from app.schemas.auth import (
    LoginRequest,
    MeResponse,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# Seconds reported in every token response (matches BACKEND_ACCESS_TOKEN_EXPIRE_MINUTES)
_EXPIRES_IN = settings.backend.access_token_expire_minutes * 60


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AuthService:
    """Orchestrates the authentication flow."""

    def __init__(self, db: Session) -> None:
        self._repo = UserRepository(db)

    # -- Register ---------------------------------------------------------

    def register(self, data: RegisterRequest) -> RegisterResponse:
        """Create a new customer account (role = 'customer')."""
        try:
            data.validate_password()
        except ValueError as exc:
            from app.core.errors import AppException

            raise AppException(
                status_code=400,
                detail=str(exc),
                error_code="INVALID_PASSWORD",
            )

        if self._repo.get_user_by_email(data.email):
            from app.core.errors import AppException

            raise AppException(
                status_code=409,
                detail="A user with this email already exists",
                error_code="EMAIL_EXISTS",
            )

        pw_hash = hash_password(data.password)
        user = self._repo.create_customer_and_user(
            email=data.email,
            password_hash=pw_hash,
            first_name=data.first_name,
            last_name=data.last_name,
            phone=data.phone,
            date_of_birth=data.date_of_birth,
            address=data.address,
        )

        return RegisterResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=user.role,
            customer_id=user.customer_id,
        )

    # -- Login ------------------------------------------------------------

    def login(self, data: LoginRequest) -> TokenResponse:
        """Authenticate with email/password and return JWT tokens."""
        user = self._repo.get_user_by_email(data.email)

        if user is None or not verify_password(data.password, user.password_hash):
            raise UnauthorizedException(detail="Invalid credentials")

        if not user.is_active:
            raise ForbiddenException(detail="Account is inactive")

        access = create_access_token(user.id, user.role)
        refresh = create_refresh_token(user.id)

        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            token_type="bearer",
            expires_in=_EXPIRES_IN,
        )

    # -- Refresh ----------------------------------------------------------

    def refresh(self, data: RefreshRequest) -> TokenResponse:
        """Validate a refresh token and issue new tokens."""
        try:
            payload = decode_refresh_token(data.refresh_token)
        except TokenError as exc:
            raise UnauthorizedException(detail=str(exc))

        user_id = UUID(payload["sub"])
        user = self._repo.get_user_by_id(user_id)

        if user is None:
            raise UnauthorizedException(detail="User not found")
        if not user.is_active:
            raise ForbiddenException(detail="Account is inactive")

        access = create_access_token(user.id, user.role)
        refresh = create_refresh_token(user.id)

        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            token_type="bearer",
            expires_in=_EXPIRES_IN,
        )

    # -- Me ---------------------------------------------------------------

    def get_me(self, user: User) -> MeResponse:
        """Return the current user's profile."""
        return MeResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=user.role,
            customer_id=user.customer_id,
            is_active=user.is_active,
            created_at=user.created_at,
        )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Dependency: extract and validate the Bearer token, return the user.

    Raises ``401 Unauthorized`` for missing / invalid / expired tokens.
    """
    if credentials is None:
        raise UnauthorizedException(detail="Authentication required")

    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        raise UnauthorizedException(detail=str(exc))

    user_id = UUID(payload["sub"])
    repo = UserRepository(db)
    user = repo.get_user_by_id(user_id)

    if user is None:
        raise UnauthorizedException(detail="User not found")
    if not user.is_active:
        raise ForbiddenException(detail="Account is inactive")

    return user


def require_role(*roles: str):
    """Factory that returns a dependency enforcing *roles*.

    Usage::

        @router.get("/admin", dependencies=[Depends(require_role("admin"))])
        def admin_view(): ...

    The authenticated user must hold **at least one** of the listed roles.
    """

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise ForbiddenException(detail="Insufficient permissions")
        return current_user

    return _check
