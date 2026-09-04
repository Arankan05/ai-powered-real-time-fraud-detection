"""Risk rules configuration model.

Persistent configuration for rule-based risk scoring.
Standalone table with no foreign keys — read by the ML/Fraud
Intelligence Service at runtime.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RiskRulesConfig(Base):
    __tablename__ = "risk_rules_config"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    rule_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    score_contribution: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # -- Constraints & Indexes --
    __table_args__ = (
        CheckConstraint("score_contribution >= 0 AND score_contribution <= 100", name="ck_risk_rules_config_score"),
        UniqueConstraint("rule_name", name="uq_risk_rules_config_rule_name"),
        Index("ix_risk_rules_config_rule_name", "rule_name", unique=True),
    )

    def __repr__(self) -> str:
        return f"<RiskRulesConfig {self.id} rule={self.rule_name!r}>"
