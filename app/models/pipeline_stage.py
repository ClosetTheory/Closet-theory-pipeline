"""PipelineStageRun model for recording stage executions and idempotency."""

from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class PipelineStageRun(Base):
    """Audit log and state machine transition entry for each pipeline stage execution."""

    __tablename__ = "pipeline_stage_runs"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("run"),
    )
    garment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("garments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    input_refs: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_refs: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    algorithm_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    error: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
