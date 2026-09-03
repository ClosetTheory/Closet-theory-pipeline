"""Model for the learned per-member style preference profile: a numeric boldness scalar
plus per-value affinities for categorical attributes (colour, pattern, ...)."""

from typing import Any, Dict
from sqlalchemy import DateTime, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class StyleProfile(Base):
    """One row per (tenant, member): the learned styling preference signal, updated by
    outfit up/downvotes (see app/rules/style_profile.py for the update math)."""

    __tablename__ = "style_profiles"
    __table_args__ = (UniqueConstraint("tenant_id", "member_id", name="uq_style_profile_member"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("sprof"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    boldness_preference: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # {"colour": {"black": 0.34, "red": -0.12, ...}, "pattern": {"solid": 0.5, ...}}
    # Each value is an EMA in [-1, 1]: positive = liked when present, negative = disliked.
    attribute_affinities: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    vote_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
