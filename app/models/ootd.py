"""Models for Outfit-of-the-Day: a daily, weather-aware, persona-aware outfit pick generated
by re-running the existing Styling Pipeline with a synthesized request_text — no new
recommendation logic, just a scheduled/cached wrapper around it (see app/styling/ootd.py)."""

from datetime import date as date_type
from typing import Any, Dict, Optional
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class OOTDSubscription(Base):
    """One row per (tenant, member): opt-in settings for automatic daily outfit-of-the-day
    generation (app/worker/queue.py's daily loop reads enabled=True rows). Absence of a row,
    or enabled=False, just means that member is skipped by the automatic daily run — the
    on-demand GET/POST endpoints work regardless, with location/persona passed explicitly."""

    __tablename__ = "ootd_subscriptions"
    __table_args__ = (UniqueConstraint("tenant_id", "member_id", name="uq_ootd_subscription_member"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("ootdsub"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    location: Mapped[str] = mapped_column(String(128), nullable=False)
    persona: Mapped[str] = mapped_column(String(64), nullable=False, default="office_going")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False,
    )


class OutfitOfTheDay(Base):
    """One row per (tenant, member, calendar date, location): the generated pick for that day.
    Wraps a real StylingRequest (produced by the normal /recommendations pipeline under the
    hood) rather than duplicating outfit-generation logic — this table exists purely so a
    repeated request for the same member/day/location returns the same cached result instead
    of re-running the full pipeline (and re-billing the vision/LLM calls) every time."""

    __tablename__ = "outfits_of_the_day"
    __table_args__ = (
        UniqueConstraint("tenant_id", "member_id", "for_date", "location", name="uq_ootd_member_date_location"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("ootd"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    for_date: Mapped[date_type] = mapped_column(Date, nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(128), nullable=False)
    persona: Mapped[str] = mapped_column(String(64), nullable=False)
    weather_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    styling_request_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("styling_requests.id", ondelete="CASCADE"), nullable=False,
    )
    generation_source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="on_demand",  # "on_demand" | "scheduled"
    )

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
