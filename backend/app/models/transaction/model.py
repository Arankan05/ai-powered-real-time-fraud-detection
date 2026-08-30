"""Transaction model.

Stores every transaction submitted through the system, including
fraud analysis results (ml_score, behaviour_score, rule_score,
risk_score, risk_level, decision, explanation_json, model_version).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint, DateTime, ForeignKey, Index, Integer,
    Numeric, String, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )

    # -- Customer / Merchant relationships --
    customer_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    merchant_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("merchants.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )

    # -- Original transaction information (immutable after creation) --
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD", server_default="USD")
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    location_country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    location_city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    device_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    device_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING",
    )

    # -- Fraud analysis outputs --
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(10), nullable=True)
    ml_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    behaviour_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rule_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explanation_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # -- Model version traceability (FK to model_metadata.model_version) --
    model_version: Mapped[str | None] = mapped_column(
        String(20),
        ForeignKey("model_metadata.model_version", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # -- Relationships --
    customer: Mapped["Customer"] = relationship(back_populates="transactions")  # noqa: F821
    merchant: Mapped["Merchant | None"] = relationship(back_populates="transactions")  # noqa: F821
    alert: Mapped["Alert | None"] = relationship(back_populates="transaction", uselist=False)  # noqa: F821
    model: Mapped["ModelMetadata | None"] = relationship(back_populates="scored_transactions")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        # CHECK constraints
        CheckConstraint("amount > 0", name="ck_transactions_amount"),
        CheckConstraint("LENGTH(currency) = 3", name="ck_transactions_currency"),
        CheckConstraint(
            "transaction_type IN ('purchase', 'transfer', 'withdrawal')",
            name="ck_transactions_transaction_type",
        ),
        CheckConstraint(
            "device_type IN ('mobile', 'desktop', 'pos')",
            name="ck_transactions_device_type",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'COMPLETED', 'FAILED')",
            name="ck_transactions_status",
        ),
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH')",
            name="ck_transactions_risk_level",
        ),
        CheckConstraint(
            "decision IN ('APPROVE', 'VERIFY', 'HOLD')",
            name="ck_transactions_decision",
        ),
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_transactions_risk_score"),
        CheckConstraint("ml_score >= 0 AND ml_score <= 100", name="ck_transactions_ml_score"),
        CheckConstraint("behaviour_score >= 0 AND behaviour_score <= 100", name="ck_transactions_behaviour_score"),
        CheckConstraint("rule_score >= 0 AND rule_score <= 100", name="ck_transactions_rule_score"),
        # Indexes
        Index("ix_transactions_customer_id", "customer_id"),
        Index("ix_transactions_merchant_id", "merchant_id"),
        Index("ix_transactions_timestamp", "timestamp"),
        Index("ix_transactions_status", "status"),
        Index("ix_transactions_risk_level", "risk_level"),
        Index("ix_transactions_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return f"<Transaction {self.id} amount={self.amount} status={self.status!r}>"
