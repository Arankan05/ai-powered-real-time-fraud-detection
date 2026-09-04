"""Merchant identity and categorisation model.

Referenced by ``transactions.merchant_id``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False, default="LOW", server_default="LOW")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # -- Relationships --
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="merchant")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH')",
            name="ck_merchants_risk_level",
        ),
        Index("ix_merchants_name", "name"),
        Index("ix_merchants_category_code", "category_code"),
    )

    def __repr__(self) -> str:
        return f"<Merchant {self.id} name={self.name!r}>"
