"""CompatibilityResult model for layering, structural, and visual compatibility evaluation."""

from typing import Optional
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class CompatibilityResult(Base):
    """Persistence record for garment pair compatibility evaluations."""

    __tablename__ = "compatibility_results"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("comp"),
    )
    garment_a: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("garments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    garment_b: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("garments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    compatibility_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )  # "LAYERING", "STRUCTURAL", "VISUAL"
    decision: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )  # "COMPATIBLE", "INCOMPATIBLE", "REVIEW_REQUIRED"

    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
