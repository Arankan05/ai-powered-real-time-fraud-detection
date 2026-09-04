"""ML model metadata model.

Tracks trained ML models for reproducibility and versioning.
``model_version`` is UNIQUE and referenced by ``transactions.model_version``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid(),
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    training_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    feature_list: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    trained_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )

    # -- Relationships --
    trainer: Mapped["User | None"] = relationship(back_populates="trained_models")  # noqa: F821
    scored_transactions: Mapped[list["Transaction"]] = relationship(back_populates="model")  # noqa: F821

    # -- Constraints & Indexes --
    __table_args__ = (
        UniqueConstraint("model_version", name="uq_model_metadata_model_version"),
        Index("ix_model_metadata_version", "model_version", unique=True),
    )

    def __repr__(self) -> str:
        return f"<ModelMetadata {self.id} version={self.model_version!r}>"
