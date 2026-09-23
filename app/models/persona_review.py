"""A stylist's verdict on one generated outfit for one character.

This is deliberately NOT the existing `stylist_reviews` table. That one has `outfit_id` declared
`unique=True`, which SQLAlchemy emits as a unique index, and carries no reviewer column at all —
so it can hold exactly one opinion per outfit, globally. Three stylists scoring the same outfit is
the entire point of the panel, so it cannot be used.

Reshaping it would mean `DROP INDEX` plus `ADD COLUMN` on a table that already exists in the
deployed database, and `create_all` performs neither; the schema would silently drift and every
query against the old model would start failing. There is also nothing sensible to backfill a
reviewer with for the existing rows. So `stylist_reviews`, its two endpoints and the `/review`
page are left exactly as they are, and this table takes the richer shape the panel actually needs:
a 1-5 score, per-dimension breakdown, and whether the stylist would put a real client in it.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now

# The four axes a stylist is actually asked to separate. Kept here rather than in the schema so
# the aggregate queries in the admin panel and the Pydantic validation agree on one list.
REVIEW_DIMENSIONS = ("colour_harmony", "fit_and_silhouette", "occasion_fit", "persona_fit")


class PersonaOutfitReview(Base):
    """One row per (outfit, reviewer). Upserted, so a stylist can revise a score without
    accumulating duplicates, and an admin can add their own alongside a stylist's."""

    __tablename__ = "persona_outfit_reviews"
    __table_args__ = (
        UniqueConstraint("outfit_id", "reviewer_user_id", name="uq_persona_outfit_review_reviewer"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("porev"))
    outfit_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalised from the outfit so "every review for this character" is one indexed query
    # rather than a join through outfits.
    persona_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("personas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    vote: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # "like" | "dislike"
    # {"colour_harmony": 4, "fit_and_silhouette": 5, "occasion_fit": 3, "persona_fit": 4}
    # An overall score alone can't distinguish "wrong colours" from "wrong occasion", and those
    # two failures point at completely different parts of the pipeline.
    dimension_ratings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The question that matters commercially, and the one a 1-5 score blurs: a stylist can score
    # an outfit 4 for craft and still never put a client in it.
    would_wear: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class OwnOutfitReview(Base):
    """The same five-dimension score, for an outfit generated in the reviewer's *own* account.

    An admin styling their own wardrobe wants to rank those outfits exactly as a stylist ranks a
    character's, but `persona_outfit_reviews.persona_id` is NOT NULL with a foreign key into
    `personas`, and there is no character behind an admin's own account. Relaxing that column
    would be an `ALTER TABLE` on the deployed database, which `create_all` never performs (see
    the module docstring above for the same reasoning applied to `stylist_reviews`). A sibling
    table with the persona column simply absent is the one shape that both works on a fresh
    `create_all` and leaves the character panel's data and aggregates untouched.
    """

    __tablename__ = "own_outfit_reviews"
    __table_args__ = (
        UniqueConstraint("outfit_id", "reviewer_user_id", name="uq_own_outfit_review_reviewer"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("oorev"))
    outfit_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    vote: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # "like" | "dislike"
    dimension_ratings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    would_wear: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
