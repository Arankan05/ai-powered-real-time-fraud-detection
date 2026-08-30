"""Alert model.

Created automatically when a transaction receives a HIGH risk decision.
Captures a snapshot of fraud analysis fields at creation time.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint, DateTime, ForeignKey, Index, Integer,
    String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    transaction_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("transactions.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False, default="HIGH", server_default="HIGH")
    decision: Mapped[str] = mapped_column(String(10), nullable=False, default="HOLD", server_default="HOLD")
    explanation_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    analyst_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="OPEN", server_default="OPEN")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # -- Relationships --
    transaction: Mapped["Transaction"] = relationship(back_populates="alert")  # noqa: F821
    analyst: Mapped["User | None"] = relationship(back_populates="alerts")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_alerts_risk_score"),
        CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_alerts_risk_level"),
        CheckConstraint("decision IN ('APPROVE', 'VERIFY', 'HOLD')", name="ck_alerts_decision"),
        CheckConstraint(
            "status IN ('OPEN', 'IN_REVIEW', 'RESOLVED', 'DISMISSED')",
            name="ck_alerts_status",
        ),
        Index("ix_alerts_transaction_id", "transaction_id"),
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_analyst_id", "analyst_id"),
    )

    def __repr__(self) -> str:
        return f"<Alert {self.id} status={self.status!r}>"
