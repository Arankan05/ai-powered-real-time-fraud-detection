"""Customer device model.

Known devices associated with a customer.  Used by the behaviour
engine for new-device detection.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, Index,
    String, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class CustomerDevice(Base):
    __tablename__ = "customer_devices"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    customer_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    device_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False)
    device_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    # -- Relationships --
    customer: Mapped["Customer"] = relationship(back_populates="devices")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        CheckConstraint(
            "device_type IN ('mobile', 'desktop', 'pos')",
            name="ck_customer_devices_device_type",
        ),
        UniqueConstraint(
            "customer_id", "device_fingerprint",
            name="uq_customer_devices_customer_fingerprint",
        ),
        Index("ix_customer_devices_customer_id", "customer_id"),
    )

    def __repr__(self) -> str:
        return f"<CustomerDevice {self.id} fingerprint={self.device_fingerprint!r}>"
