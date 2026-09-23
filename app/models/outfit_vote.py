"""The behaviour ledger: one row per (outfit, voter, source) thumbs-up or thumbs-down.

Before this table existed a vote only nudged two aggregates on StyleProfile (a boldness scalar
and per-attribute-value affinities) and the vote itself was discarded — nobody recorded which
outfit, which garments, up or down, or when. Without that ledger no garment or garment *pairing*
can ever be scored, which is why Styling Stage 3 (Wardrobe Behaviour) stayed a constant 0.5.

Every signal that expresses an opinion on a generated outfit writes here through
app.styling.behavior.record_outfit_vote: the 👍/👎 on the styling page, and the 1-5 star
reviews on the review page (1-2 = down, 4-5 = up, weighted by distance from 3). Rows are
upserted per (outfit, voter, source) so re-clicking flips an opinion instead of stacking it.

`garment_ids` is snapshotted at vote time: the scorer needs the outfit's composition, and an
outfit's garments can later be swapped or deleted (OutfitGarment rows cascade away) — the vote
must keep meaning what it meant when it was cast.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now

VOTE_SOURCES = ("styling_page", "persona_review", "own_review")


class OutfitVote(Base):
    __tablename__ = "outfit_votes"
    __table_args__ = (
        UniqueConstraint("outfit_id", "voter_user_id", "source", name="uq_outfit_vote_voter_source"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("ovote"))
    outfit_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The wardrobe the outfit belongs to — the scope the behaviour model is built per. Copied from
    # the Outfit so Stage 3 can load a member's whole ledger with one indexed query.
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    voter_user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    vote: Mapped[str] = mapped_column(String(8), nullable=False)  # "up" | "down"
    # 1.0 for a thumbs click; a star review contributes |rating - 3| / 2 (0.5 or 1.0).
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    garment_ids: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    # --- the reason, when one was given (see app.rules.feedback) ---
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)  # reason chips
    # The extractor's structured reading of the comment, with the model that produced it, so a
    # re-extraction later can tell which rows came from the heuristic vs. a real model.
    feedback: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # What the scorer actually consumes: garment_id -> weight on this vote's polarity, "a|b" -> pair
    # weight, and garments that get the OPPOSITE polarity (praise inside a 👎). Empty = even split.
    garment_weights: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict, nullable=False)
    pair_weights: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict, nullable=False)
    counter_garment_ids: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    feedback_summary: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
