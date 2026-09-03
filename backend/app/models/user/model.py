"""Authentication and authorisation model.

Every person who can log into the system has a record here.
Links to ``customers`` for customer-role users (nullable FK).
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import (
    Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index,
    String, func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="customer", server_default="customer")
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    customer_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # -- Relationships --
    customer: Mapped["Customer | None"] = relationship(back_populates="user")  # noqa: F821
    alerts: Mapped[list["Alert"]] = relationship(back_populates="analyst")  # noqa: F821
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="actor")  # noqa: F821
    trained_models: Mapped[list["ModelMetadata"]] = relationship(back_populates="trainer")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        CheckConstraint(
            "role IN ('customer', 'fraud_analyst', 'admin')",
            name="ck_users_role",
        ),
        Index("ix_users_email", "email", unique=True),
        Index("ix_users_role", "role"),
        Index("ix_users_customer_id", "customer_id"),
    )

    def __repr__(self) -> str:
        return f"<User {self.id} email={self.email!r}>"
