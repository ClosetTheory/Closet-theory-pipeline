"""Glue between the vote ledger (DB) and the pure behaviour scorer (app.rules.wardrobe_behavior).

Three jobs:

- load a member's ledger into a WardrobeBehaviorModel for Stage 3;
- record a vote from any feeder (styling-page thumbs, character review stars, own-outfit review
  stars) with one upsert — including the *reason*: reason chips and a free-text comment are
  turned into per-garment / per-pair weights (app.rules.feedback) and attribute lessons that
  sharpen the style profile's affinities;
- digest recent comments and recurring lessons into "taste notes" that Stage 2 hands to the
  prompts, so Stage 8 can fail an outfit that repeats something the member already said no to.
"""

import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.garment import Garment
from app.models.outfit_vote import OutfitVote
from app.models.style_profile import StyleProfile
from app.models.styling import Outfit, OutfitGarment
from app.providers.feedback import get_feedback_extractor_provider
from app.rules.feedback import (
    REASON_TAGS,
    FeedbackExtraction,
    FeedbackGarment,
    derive_vote_weights,
    describe_feedback,
    heuristic_extract,
)
from app.rules.style_profile import apply_attribute_lessons
from app.rules.wardrobe_behavior import BehaviorVote, WardrobeBehaviorModel, build_behavior_model

logger = logging.getLogger("pipeline")

SOURCE_LABELS = {"styling_page": "thumbs", "persona_review": "stylist review", "own_review": "own review"}


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
        BehaviorVote(
            garment_ids=tuple(r.garment_ids or ()),
            vote=r.vote,
            weight=float(r.weight or 0.0),
            created_at=r.created_at,
            garment_weights=r.garment_weights or None,
            pair_weights=r.pair_weights or None,
            counter_garment_ids=tuple(r.counter_garment_ids or ()),
        )
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


async def _outfit_feedback_garments(session: AsyncSession, outfit: Outfit) -> Tuple[List[FeedbackGarment], Dict[str, str]]:
    rows = (
        await session.execute(
            select(OutfitGarment.garment_id, OutfitGarment.role, Garment)
            .join(Garment, Garment.id == OutfitGarment.garment_id)
            .where(OutfitGarment.outfit_id == outfit.id)
        )
    ).all()
    garments: List[FeedbackGarment] = []
    labels: Dict[str, str] = {}
    for gid, role, g in rows:
        attrs = g.attributes_json or {}
        colours = tuple(str(c) for c in (attrs.get("colour") or []) if c)
        garments.append(FeedbackGarment(
            garment_id=gid, role=role or "", category=g.category or "", subcategory=g.subcategory or "",
            casual_name=str(attrs.get("casual_name") or ""), colours=colours,
        ))
        labels[gid] = str(attrs.get("casual_name") or g.subcategory or g.category or gid).lower().replace("_", " ")
    return garments, labels


async def apply_lessons_to_profile(session: AsyncSession, tenant_id: str, member_id: str, lessons: List[Dict[str, str]]) -> None:
    """A stated lesson ("no florals") nudges exactly that attribute value on the member's learned
    style profile — much sharper than the blanket nudge every vote gives all attributes of all
    garments in the outfit."""
    if not lessons:
        return
    profile = (
        await session.execute(select(StyleProfile).where(StyleProfile.tenant_id == tenant_id, StyleProfile.member_id == member_id))
    ).scalars().first()
    if profile is None:
        profile = StyleProfile(tenant_id=tenant_id, member_id=member_id)
        session.add(profile)
        await session.flush()
    profile.attribute_affinities = apply_attribute_lessons(profile.attribute_affinities or {}, lessons)
    await session.flush()


async def record_outfit_vote(
    session: AsyncSession,
    outfit: Outfit,
    voter_user_id: str,
    source: str,
    vote: Optional[str],
    weight: float = 1.0,
    comment: Optional[str] = None,
    tags: Iterable[str] = (),
) -> Optional[OutfitVote]:
    """Upserts this voter's opinion on the outfit from this source; `vote=None` withdraws it.

    With a comment and/or reason chips the vote is attributed: the extractor names the garments,
    pairings and attribute lessons the comment is about, and the row stores per-garment and
    per-pair weights the scorer then uses instead of an even split. Extraction failing, or a
    comment that names nothing recognisable, leaves the even split — never worse than a bare vote.

    Flushes but does not commit — the caller owns the transaction."""
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

    feedback_garments, labels = await _outfit_feedback_garments(session, outfit)
    garment_ids = [g.garment_id for g in feedback_garments] or list((existing.garment_ids if existing else None) or [])
    clean_tags = [t for t in dict.fromkeys(str(t) for t in tags) if t in REASON_TAGS]
    comment = (comment or "").strip() or None

    extraction: Optional[FeedbackExtraction] = None
    if comment and feedback_garments:
        try:
            extraction = await get_feedback_extractor_provider().extract(comment, vote, feedback_garments)
        except Exception as e:  # noqa: BLE001 — a failed model call must never lose the vote
            logger.warning(f"Feedback extraction failed ({e}); using keyword heuristic.")
            extraction = heuristic_extract(comment, vote, feedback_garments)

    weights = derive_vote_weights(vote, garment_ids, clean_tags, extraction)

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
    existing.garment_ids = garment_ids
    existing.comment = comment
    existing.tags = clean_tags
    existing.feedback = extraction.to_dict() if extraction is not None else None
    existing.garment_weights = weights.garment_weights
    existing.pair_weights = weights.pair_weights
    existing.counter_garment_ids = weights.counter_garment_ids
    existing.feedback_summary = describe_feedback(clean_tags, extraction, labels) or None
    await session.flush()

    if extraction is not None and extraction.lessons:
        await apply_lessons_to_profile(session, outfit.tenant_id, outfit.member_id, extraction.lessons)
    return existing


async def load_taste_notes(session: AsyncSession, tenant_id: str, member_id: str, limit: int = 6) -> List[str]:
    """Short, plain-language notes for the prompts: the most recent commented votes verbatim
    (as data) and the lessons that recur across all of them."""
    rows = (
        await session.execute(
            select(OutfitVote)
            .where(OutfitVote.tenant_id == tenant_id, OutfitVote.member_id == member_id)
            .order_by(OutfitVote.updated_at.desc())
            .limit(200)
        )
    ).scalars().all()
    if not rows:
        return []

    notes: List[str] = []
    for r in rows:
        if r.comment and len(notes) < limit:
            who = SOURCE_LABELS.get(r.source, r.source)
            verdict = "Disliked" if r.vote == "down" else "Liked"
            notes.append(f'{verdict} ({who}): "{r.comment[:160]}"')

    lesson_counts: Counter = Counter()
    for r in rows:
        for lesson in ((r.feedback or {}).get("lessons") or []):
            lesson_counts[(lesson.get("attribute"), lesson.get("value"), lesson.get("polarity"))] += 1
    for (attribute, value, polarity), count in lesson_counts.most_common(6):
        if not value:
            continue
        verb = "dislikes" if polarity == "down" else "likes"
        notes.append(f"Stated preference: {verb} {value} ({attribute})" + (f", said {count} times" if count > 1 else ""))
    return notes
