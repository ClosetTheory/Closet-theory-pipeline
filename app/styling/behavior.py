"""Glue between the vote ledger (DB) and the pure behaviour scorer (app.rules.wardrobe_behavior).

Two jobs: load a member's ledger into a WardrobeBehaviorModel for Stage 3, and record a vote
from any of the three feeders (styling-page thumbs, character review stars, own-outfit review
stars) with one upsert so they all shape the same model.
"""

from typing import List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outfit_vote import OutfitVote
from app.models.styling import Outfit, OutfitGarment
from app.rules.wardrobe_behavior import BehaviorVote, WardrobeBehaviorModel, build_behavior_model


async def load_behavior_model(session: AsyncSession, tenant_id: str, member_id: str) -> WardrobeBehaviorModel:
    """Builds the model on the fly from every vote on this wardrobe's outfits. Wardrobes here
    are small and a member's votes number in the hundreds at most, so recomputing per request
    is cheap and always consistent — no materialised affinity tables to drift."""
    rows = (
        await session.execute(
            select(OutfitVote).where(OutfitVote.tenant_id == tenant_id, OutfitVote.member_id == member_id)
        )
    ).scalars().all()
    votes = [
        BehaviorVote(garment_ids=tuple(r.garment_ids or ()), vote=r.vote, weight=float(r.weight or 0.0), created_at=r.created_at)
        for r in rows
    ]
    return build_behavior_model(votes)


def vote_from_rating(rating: Optional[int]) -> Tuple[Optional[str], float]:
    """Maps a 1-5 star review onto the ledger: 1-2 down, 4-5 up, weighted by distance from the
    neutral 3 (so a 1 or 5 counts like a thumbs click, a 2 or 4 half as much). A 3 is no
    opinion and removes any earlier ledger row from that reviewer."""
    if rating is None or rating == 3:
        return None, 0.0
    return ("up" if rating > 3 else "down"), abs(rating - 3) / 2.0


async def record_outfit_vote(
    session: AsyncSession,
    outfit: Outfit,
    voter_user_id: str,
    source: str,
    vote: Optional[str],
    weight: float = 1.0,
) -> Optional[OutfitVote]:
    """Upserts this voter's opinion on the outfit from this source; `vote=None` withdraws it.
    Snapshots the outfit's current garment ids. Flushes but does not commit — the caller owns
    the transaction (it is usually also updating StyleProfile or a review row)."""
    existing = (
        await session.execute(
            select(OutfitVote).where(
                OutfitVote.outfit_id == outfit.id,
                OutfitVote.voter_user_id == voter_user_id,
                OutfitVote.source == source,
            )
        )
    ).scalars().first()

    if vote is None:
        if existing is not None:
            await session.delete(existing)
            await session.flush()
        return None

    garment_ids: List[str] = list(
        (await session.execute(select(OutfitGarment.garment_id).where(OutfitGarment.outfit_id == outfit.id))).scalars().all()
    )
    if existing is None:
        existing = OutfitVote(
            outfit_id=outfit.id,
            tenant_id=outfit.tenant_id,
            member_id=outfit.member_id,
            voter_user_id=voter_user_id,
            source=source,
            vote=vote,
        )
        session.add(existing)
    existing.vote = vote
    existing.weight = weight
    existing.garment_ids = garment_ids or list(existing.garment_ids or [])
    await session.flush()
    return existing
