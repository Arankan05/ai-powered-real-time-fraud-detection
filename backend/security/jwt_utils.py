"""JWT creation and validation (PyJWT).

Access tokens are short-lived (30 minutes by default, matching the
API contract's ``expires_in: 1800``) and carry only the claims needed
for authorisation: ``sub`` (user ID), ``role``, ``type``, ``iat`` and
``exp``.  Refresh tokens use the same claims with ``type: refresh``
and a longer expiry.

All secrets and expiry windows come from :mod:`backend.config`
(environment-driven).  The development default secret must be replaced
in production via the ``BACKEND_SECRET_KEY`` environment variable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.config import get_settings

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


def create_access_token(
    *,
    user_id: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT access token."""
    settings = get_settings()
    minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.BACKEND_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.BACKEND_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    *,
    user_id: str,
    role: str,
    expires_days: int | None = None,
) -> str:
    """Create a signed JWT refresh token."""
    settings = get_settings()
    days = (
        expires_days
        if expires_days is not None
        else settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": REFRESH_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(days=days),
    }
    return jwt.encode(payload, settings.BACKEND_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT.

    Signature, expiry and required-claims checks are enforced by
    PyJWT.  Raises ``jwt.InvalidTokenError`` (or a subclass such as
    ``jwt.ExpiredSignatureError``) for any invalid, expired or
    malformed token.
    """
    settings = get_settings()
    return jwt.decode(
        token,
        settings.BACKEND_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
