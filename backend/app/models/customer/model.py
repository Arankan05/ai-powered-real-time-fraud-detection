"""Customer business/profile model.

Stores customer identity information used for behavioural baselines
and transaction association.  Referenced by ``users``, ``transactions``,
and ``customer_devices``.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import Boolean, Date, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    # -- Relationships --
    user: Mapped["User | None"] = relationship(back_populates="customer", uselist=False)  # noqa: F821
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="customer")  # noqa: F821
    devices: Mapped[list["CustomerDevice"]] = relationship(back_populates="customer")  # noqa: F821

    # -- Indexes --
    __table_args__ = (
        Index("ix_customers_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Customer {self.id}>"
