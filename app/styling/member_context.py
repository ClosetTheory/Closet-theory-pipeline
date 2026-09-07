"""Derives a real, member-specific styling context from whatever data already exists for
them — no manual "pick a persona from a list" step. Used by Outfit-of-the-Day (app/styling/
ootd.py) to ground the daily pick in this specific member's actual history and wardrobe,
not a generic assumption.

Signals pulled (a starting set — more can be added later without changing the calling
convention, since this always just returns one natural-language paragraph):
  - Most frequent occasion/formality/style_direction/gender from this member's past
    styling requests (StylingRequest.normalized_intent) — what they actually tend to ask for.
  - Dominant gender and most common subcategories across their real wardrobe (Garment rows)
    — what they actually own.
  - Learned boldness_preference (StyleProfile) — already flows into every recommendation via
    StylingContext.user_preferences regardless of request_text, so it's only mentioned here
    for a bit of extra explicit framing, not relied on as the only place it's used.
"""

from collections import Counter
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.garment import Garment
from app.models.style_profile import StyleProfile
from app.models.styling import StylingRequest

_HISTORY_LIMIT = 20  # most recent styling requests to learn from
_WARDROBE_LIMIT = 500  # plenty for a personal wardrobe; caps the query on a huge catalogue


def _most_common(values: List[str], min_count: int = 2) -> Optional[str]:
    values = [v for v in values if v]
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count >= min_count else (value if len(values) == 1 else None)


async def derive_member_context(session: AsyncSession, tenant_id: str, member_id: str) -> str:
    # 1. What this member actually tends to ask for
    history_stmt = (
        select(StylingRequest.normalized_intent)
        .where(StylingRequest.tenant_id == tenant_id, StylingRequest.member_id == member_id)
        .order_by(StylingRequest.created_at.desc())
        .limit(_HISTORY_LIMIT)
    )
    intents = [row[0] or {} for row in (await session.execute(history_stmt)).all()]
    top_occasion = _most_common([i.get("occasion") for i in intents])
    top_formality = _most_common([i.get("formality") for i in intents])
    top_style = _most_common([i.get("style_direction") for i in intents])
    top_gender_requested = _most_common([i.get("gender") for i in intents])

    # 2. What this member actually owns
    wardrobe_stmt = (
        select(Garment.gender, Garment.subcategory)
        .where(Garment.tenant_id == tenant_id, Garment.member_id == member_id, Garment.status == "COMPLETED")
        .limit(_WARDROBE_LIMIT)
    )
    wardrobe_rows = (await session.execute(wardrobe_stmt)).all()
    dominant_gender = _most_common([g for g, _ in wardrobe_rows], min_count=1)
    top_subcats = [s for s, c in Counter([sc for _, sc in wardrobe_rows if sc]).most_common(3)]

    # 3. Learned bold-vs-conventional preference (also auto-applied to ranking separately —
    # see StylingOrchestrator.run()'s Stage 2 — this is just extra explicit framing for the LLM).
    profile_stmt = select(StyleProfile).where(StyleProfile.tenant_id == tenant_id, StyleProfile.member_id == member_id)
    profile = (await session.execute(profile_stmt)).scalars().first()
    boldness_note = None
    if profile and profile.vote_count >= 3:
        boldness_note = "leans toward bolder, less conventional combinations" if profile.boldness_preference > 0.2 else (
            "prefers conventional, safely-matching combinations" if profile.boldness_preference < -0.2 else None
        )

    parts = []
    if top_occasion or top_formality:
        parts.append(
            f"This member most often styles for {top_occasion or 'general'} occasions"
            + (f" at {top_formality} formality" if top_formality else "") + "."
        )
    if top_style:
        parts.append(f"They tend to favour a {top_style.lower()} style direction.")
    if top_subcats:
        parts.append(f"Their wardrobe's most common pieces are: {', '.join(s.replace('_', ' ') for s in top_subcats)}.")
    if dominant_gender and dominant_gender != "unisex":
        parts.append(f"Their wardrobe is predominantly {dominant_gender}'s clothing.")
    elif top_gender_requested:
        parts.append(f"They usually style for {top_gender_requested}'s wear.")
    if boldness_note:
        parts.append(f"Based on past feedback, this member {boldness_note}.")

    if not parts:
        return "No styling history yet for this member — dress appropriately and comfortably for today's real weather conditions."
    return " ".join(parts)
