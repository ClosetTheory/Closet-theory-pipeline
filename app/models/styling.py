"""Models for the Styling Pipeline: requests, outfits, and outfit membership."""

from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class StylingRequest(Base):
    """Persistence record for a single styling recommendation request."""

    __tablename__ = "styling_requests"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("sreq"),
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    raw_text: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    anchor_garment_ids: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    normalized_intent: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    trace: Mapped[List[Any]] = mapped_column(JSON, default=list, nullable=False)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )


class Outfit(Base):
    """A single ranked outfit recommendation produced for a StylingRequest."""

    __tablename__ = "outfits"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("outfit"),
    )
    request_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("styling_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score_breakdown: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    compatibility_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    semantic_validation: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    visual_validation: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    generated_image_semantic_validation: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    generated_image_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("image_assets.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )


class OutfitGarment(Base):
    """Join table: which real garments (and in which role) make up an Outfit."""

    __tablename__ = "outfit_garments"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("og"),
    )
    outfit_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("outfits.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    garment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("garments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # TOP|BOTTOM|OUTERWEAR|FOOTWEAR|ONE_PIECE|ACCESSORY


class StylistReview(Base):
    """Internal-only QA record: a company stylist's like/dislike + free-text comment on a
    generated outfit, reviewed from their own account's own generated outfits (see
    app/static/review.html). Deliberately separate from OutfitVote (app/api/v1/styling.py's
    /vote endpoint) — that mechanism mutates the member's learned StyleProfile and keeps no
    durable per-vote record; this table exists purely so a reviewer's judgment on a specific
    outfit persists and can carry a comment, with no effect on ranking/learning."""

    __tablename__ = "stylist_reviews"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("sreview"),
    )
    outfit_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("outfits.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    vote: Mapped[str] = mapped_column(String(16), nullable=False)  # "like" | "dislike"
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
