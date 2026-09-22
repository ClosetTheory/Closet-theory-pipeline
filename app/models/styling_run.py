"""Durable progress checkpoint for an in-flight styling pipeline run.

`StylingRequest` (app/models/styling.py) is only ever written once, at the very end of
`StylingOrchestrator.run()`, after all 10 stages finish -- it's the record of a *finished*
recommendation, and nothing about that changes here. A run that's still going (or that crashed
mid-run) has no footprint there at all, which is exactly why a tab switch during generation
loses everything: there is nothing in the database yet to resume from.

This is a separate table, not a column added to `styling_requests`, for the same reason
`PersonaOutfitReview` is a separate table rather than a reshaped `stylist_reviews`: this project
has no migrations, `create_all` can only create tables it doesn't see yet, and altering an
existing table's columns isn't something that mechanism can do safely. Progress tracking is
also a genuinely different access pattern from a finished request -- many small updates against
one row, versus one write ever -- so it doesn't belong on the same model even if migrations did
exist.

Written entirely by the streaming API layer's on_stage callback (app/api/v1/styling.py), never
by StylingOrchestrator itself -- so nothing about the orchestrator's 10 stage functions changes
to support this; they already call `self.on_stage(entry)` after every stage, and this table is
just a new subscriber to that existing hook.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now

# RUNNING -> SUCCEEDED | FAILED, or RUNNING -> ABANDONED (a newer request for the same scope
# superseded it, or it aged out un-flipped after a process crash -- see the lazy staleness
# check in GET /wardrobe/styling/runs/active).
RUN_STATUSES = ("RUNNING", "SUCCEEDED", "FAILED", "ABANDONED")


class StylingRunProgress(Base):
    __tablename__ = "styling_run_progress"
    __table_args__ = (
        Index("ix_styling_run_progress_scope_status", "tenant_id", "member_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("srun"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Denormalised, same reasoning as PersonaOutfitReview.persona_id -- "the active run for this
    # character" without a join. Null for an ordinary member's own request.
    persona_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("personas.id", ondelete="SET NULL"), nullable=True, index=True
    )
    initiated_by_user_id: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="RUNNING")
    request_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    current_stage: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trace: Mapped[List[Any]] = mapped_column(JSON, default=list, nullable=False)
    styling_request_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("styling_requests.id", ondelete="SET NULL"), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
