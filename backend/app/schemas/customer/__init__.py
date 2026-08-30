"""Pydantic schemas for customer endpoints.

Response shapes match ``docs/api-contract.md`` exactly.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel


class CustomerResponse(BaseModel):
    """GET /api/v1/customers/me and GET /api/v1/customers/{id} response."""

    id: UUID
    first_name: str
    last_name: str
    phone: str | None = None
    address: str | None = None
    date_of_birth: date | None = None
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}
