"""Reusable FastAPI authentication and authorisation dependencies.

Provides:

* :func:`set_user_repository` / :func:`get_user_repository` — module-level
  repository injection (mirrors the alert-repository pattern used by the
  routers).
* :func:`get_current_user` — validates the ``Authorization: Bearer``
  header, decodes the JWT, loads the user and rejects unknown or
  inactive accounts.
* :func:`require_roles` — dependency factory enforcing role-based
  authorisation on top of :func:`get_current_user`.

JWT validation lives in one place — routers must never decode tokens
themselves.

Status-code conventions:

* ``401`` — missing header, wrong scheme, malformed/expired/invalid
  token, or token referencing a user that no longer exists.
* ``403`` — valid authentication but insufficient permission (wrong
  role, or inactive account).
"""

from __future__ import annotations

from typing import Any, Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.security.jwt_utils import ACCESS_TOKEN_TYPE, decode_token

_bearer_scheme = HTTPBearer(auto_error=False)

# Module-level user repository — set at app startup
_user_repo: Any = None


def set_user_repository(repo: Any) -> None:
    """Set the user repository (called during app startup and tests)."""
    global _user_repo
    _user_repo = repo


def get_user_repository() -> Any:
    """Return the active user repository."""
    if _user_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not configured.",
        )
    return _user_repo


def _unauthorized(detail: str) -> HTTPException:
    """Build a 401 error carrying the WWW-Authenticate challenge."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """Validate the Bearer token and return the active user record.

    Raises 401 for missing/malformed/expired tokens and for tokens that
    reference a user that no longer exists; 403 for inactive accounts.
    """
    if credentials is None:
        # Covers a missing Authorization header and non-Bearer schemes
        raise _unauthorized("Not authenticated. Provide a Bearer token.")

    try:
        payload = decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired.")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid authentication token.")

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        # Refresh tokens must not be used as access tokens
        raise _unauthorized("Invalid authentication token.")

    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("Invalid authentication token.")

    repo = get_user_repository()
    user = repo.get_by_id(user_id)
    if user is None:
        raise _unauthorized("Invalid authentication token.")

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive.",
        )

    return user


def require_roles(*allowed_roles: str) -> Callable[..., Any]:
    """Dependency factory that enforces role-based authorisation.

    Usage::

        @router.get("/alerts")
        def list_alerts(
            current_user: dict = Depends(require_roles("fraud_analyst", "admin")),
        ): ...
    """

    async def role_checker(
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this operation.",
            )
        return current_user

    return role_checker
