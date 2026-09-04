"""Pydantic schemas for authentication endpoints.

Request/response shapes match ``docs/api-contract.md`` exactly.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from app.core.security import validate_password_strength


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """POST /api/v1/auth/register request body."""

    email: EmailStr = Field(..., max_length=255, description="Valid email, max 255 chars")
    password: str = Field(..., min_length=8, max_length=128, description="Min 8 chars, 1 upper, 1 lower, 1 digit")
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    phone: str | None = Field(None, max_length=30)
    date_of_birth: date | None = None
    address: str | None = None

    def validate_password(self) -> None:
        """Raise ``ValueError`` when the password fails strength rules."""
        if not validate_password_strength(self.password):
            raise ValueError(
                "Password must be 8-128 chars with at least 1 uppercase, "
                "1 lowercase, and 1 digit."
            )


class LoginRequest(BaseModel):
    """POST /api/v1/auth/login request body."""

    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    """POST /api/v1/auth/refresh request body."""

    refresh_token: str


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class RegisterResponse(BaseModel):
    """POST /api/v1/auth/register success response (201)."""

    id: UUID
    email: str
    first_name: str
    last_name: str
    role: str
    customer_id: UUID | None = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """POST /api/v1/auth/login and /auth/refresh success response (200)."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    """GET /api/v1/auth/me success response (200)."""

    id: UUID
    email: str
    first_name: str
    last_name: str
    role: str
    customer_id: UUID | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
