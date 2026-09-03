"""Password hashing and JWT token utilities.

* Passwords are hashed with **bcrypt** as specified in ``docs/architecture.md``.
* JWT access/refresh tokens are created with **PyJWT**.
* The signing key comes from ``BACKEND_SECRET_KEY`` — never hard-coded.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from app.config import settings

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return ``True`` when *plain* matches the bcrypt *hashed* value."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


_PASSWORD_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,128}$")


def validate_password_strength(password: str) -> bool:
    """Return ``True`` when *password* meets the documented strength rules.

    Rules (from ``api-contract.md``):
    * Minimum 8 characters, maximum 128 characters.
    * At least 1 uppercase letter.
    * At least 1 lowercase letter.
    * At least 1 digit.
    """
    return _PASSWORD_RE.match(password) is not None


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"
_REFRESH_EXPIRE_DAYS = 7


def create_access_token(
    user_id: UUID,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    delta = expires_delta or timedelta(
        minutes=settings.backend.access_token_expire_minutes,
    )
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + delta,
    }
    return jwt.encode(payload, settings.backend.secret_key, algorithm=_ALGORITHM)


def create_refresh_token(user_id: UUID) -> str:
    """Create a signed JWT refresh token (longer-lived)."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=_REFRESH_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.backend.secret_key, algorithm=_ALGORITHM)


class TokenError(Exception):
    """Raised when a token is invalid or expired."""


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Returns the payload dict on success.

    Raises ``TokenError`` for any validation failure (expired, malformed,
    wrong type, bad signature).
    """
    try:
        payload = jwt.decode(
            token,
            settings.backend.secret_key,
            algorithms=[_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise TokenError("Token has expired")
    except jwt.InvalidTokenError:
        raise TokenError("Invalid token")

    if "sub" not in payload or "type" not in payload:
        raise TokenError("Token is missing required claims")

    return payload


def decode_access_token(token: str) -> dict:
    """Decode an access token, ensuring ``type == 'access'``."""
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise TokenError("Token is not an access token")
    return payload


def decode_refresh_token(token: str) -> dict:
    """Decode a refresh token, ensuring ``type == 'refresh'``."""
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise TokenError("Token is not a refresh token")
    return payload
