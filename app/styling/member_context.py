"""Derives a real, member-specific styling context from whatever data already exists for
them — no manual "pick a persona from a list" step. Used by Stage 2 of the styling pipeline and
by Outfit-of-the-Day (app/styling/ootd.py) to ground a pick in this specific member's actual
history, stated profile and wardrobe, not a generic assumption.

Signals pulled (always returned as one natural-language paragraph, so callers never change):
  - Most frequent occasion/formality/style_direction/gender from this member's past
    *member-initiated* styling requests (StylingRequest.normalized_intent) — what they actually
    tend to ask for.
  - What they have literally asked for recently, verbatim (`recent_asks`) — the only place a
    member states what they *want* rather than what they own.
  - Dominant gender and most common subcategories across their real wardrobe (Garment rows)
    — what they actually own.
  - Their stated profile when one exists (app.styling.member_signals): colour analysis,
    colours loved and avoided, fits, aesthetic leanings, hard constraints and today's plan.
  - Learned boldness_preference (StyleProfile) — already flows into every recommendation via
    StylingContext.user_preferences regardless of request_text, so it's only mentioned here
    for a bit of extra explicit framing, not relied on as the only place it's used.

The circularity fix. Outfit-of-the-Day synthesises its own request text *from* this paragraph,
then persists that request like any other. Left unfiltered, the history query would read back
5,000 of its own "Suggest today's outfit…" prompts (production: every one of the 5,736 requests)
and the summary would converge on whatever it said first. So requests an OOTD row points at are
excluded from the history, as is anything that starts with the OOTD prompt prefix — belt and
braces, because the OOTD row is written after the request and a crash between the two would
otherwise leak one back in.
"""

from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.garment import Garment
from app.models.ootd import OutfitOfTheDay
from app.models.style_profile import StyleProfile
from app.models.styling import StylingRequest
from app.rules.member_signals import constraint_label, palette_summary, today_plan
from app.styling.member_signals import MemberSignals

_HISTORY_LIMIT = 20  # most recent member-initiated styling requests to learn from
_RECENT_ASKS = 4     # verbatim recent requests to quote back
_ASK_MAX_CHARS = 120
_WARDROBE_LIMIT = 500  # plenty for a personal wardrobe; caps the query on a huge catalogue

# What app/styling/ootd.py's _build_request_text starts every synthesised request with.
OOTD_REQUEST_PREFIX = "Suggest today's outfit"

NO_HISTORY_NOTE = (
    "No styling history yet for this member — dress appropriately and comfortably for today's "
    "real weather conditions."
)


def _most_common(values: List[str], min_count: int = 2) -> Optional[str]:
    values = [v for v in values if v]
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count >= min_count else (value if len(values) == 1 else None)


def _member_initiated_requests(tenant_id: str, member_id: str, limit: int):
    ootd_requests = select(OutfitOfTheDay.styling_request_id).where(
        OutfitOfTheDay.tenant_id == tenant_id, OutfitOfTheDay.member_id == member_id
    )
    return (
        select(StylingRequest.raw_text, StylingRequest.normalized_intent)
        .where(
            StylingRequest.tenant_id == tenant_id,
            StylingRequest.member_id == member_id,
            StylingRequest.id.not_in(ootd_requests),
            # NULL raw_text (anchor-only requests) must survive: NOT (NULL LIKE …) is NULL.
            or_(StylingRequest.raw_text.is_(None), ~StylingRequest.raw_text.startswith(OOTD_REQUEST_PREFIX)),
        )
        .order_by(StylingRequest.created_at.desc())
        .limit(limit)
    )


async def load_recent_asks(session: AsyncSession, tenant_id: str, member_id: str, limit: int = _RECENT_ASKS) -> List[str]:
    """The member's own recent request texts, newest first, OOTD prompts excluded — the stated
    wants Stage 2 hands to the prompts and the trace as data."""
    rows = (await session.execute(_member_initiated_requests(tenant_id, member_id, limit))).all()
    asks: List[str] = []
    for raw_text, _intent in rows:
        text = " ".join(str(raw_text or "").split())
        if text:
            asks.append(text[:_ASK_MAX_CHARS] + ("…" if len(text) > _ASK_MAX_CHARS else ""))
    return asks


def _profile_sentences(signals: MemberSignals, today: Optional[date]) -> List[str]:
    parts: List[str] = []
    palette = palette_summary(signals.color_analysis)
    if palette:
        parts.append(f"Colour analysis: {palette.rstrip('.')}.")
    prefs: Dict[str, Any] = signals.preferences or {}
    colours = prefs.get("colour_preferences") or {}
    love = [str(c) for c in (colours.get("love") or [])] if isinstance(colours, dict) else []
    avoid = [str(c) for c in (colours.get("avoid") or [])] if isinstance(colours, dict) else []
    if love or avoid:
        bits = []
        if love:
            bits.append(f"loves {', '.join(love)}")
        if avoid:
            bits.append(f"avoids {', '.join(avoid)}")
        parts.append(f"Stated colour preferences: {'; '.join(bits)}.")
    fits = [str(f).replace("_", " ") for f in (prefs.get("fits_loved") or [])]
    leanings = [str(a).replace("_", " ") for a in (prefs.get("aesthetic_leanings") or [])]
    if fits or leanings:
        bits = []
        if fits:
            bits.append(f"prefers {', '.join(fits)} fits")
        if leanings:
            bits.append(f"leans {', '.join(leanings)}")
        parts.append(f"They {' and '.join(bits)}.")
    if signals.hard_constraints:
        parts.append(
            "Hard constraints, never to be broken: "
            + "; ".join(constraint_label(c) for c in signals.hard_constraints) + "."
        )
    plan = today_plan(signals.weekly_plan, today)
    if plan and (plan["tags"] or plan["note"]):
        tags = ", ".join(t.replace("_", " ") for t in plan["tags"])
        parts.append(
            "Today's plan: " + (tags if tags else "unspecified") + (f" — {plan['note']}" if plan["note"] else "") + "."
        )
    if signals.styling_notes:
        parts.append(signals.styling_notes.strip().rstrip(".") + ".")
    return parts


async def derive_member_context(
    session: AsyncSession,
    tenant_id: str,
    member_id: str,
    signals: Optional[MemberSignals] = None,
    today: Optional[date] = None,
) -> str:
    # 1. What this member actually tends to ask for — their own requests, not OOTD's
    rows = (await session.execute(_member_initiated_requests(tenant_id, member_id, _HISTORY_LIMIT))).all()
    intents = [intent or {} for _raw, intent in rows]
    top_occasion = _most_common([i.get("occasion") for i in intents])
    top_formality = _most_common([i.get("formality") for i in intents])
    top_style = _most_common([i.get("style_direction") for i in intents])
    top_gender_requested = _most_common([i.get("gender") for i in intents])
    recent_asks = []
    for raw_text, _intent in rows[:_RECENT_ASKS]:
        text = " ".join(str(raw_text or "").split())
        if text:
            recent_asks.append(text[:_ASK_MAX_CHARS] + ("…" if len(text) > _ASK_MAX_CHARS else ""))

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
    if recent_asks:
        quoted = "; ".join(f'"{a}"' for a in recent_asks)
        parts.append(f"They recently asked for: {quoted}.")
    if top_subcats:
        parts.append(f"Their wardrobe's most common pieces are: {', '.join(s.replace('_', ' ') for s in top_subcats)}.")
    if dominant_gender and dominant_gender != "unisex":
        parts.append(f"Their wardrobe is predominantly {dominant_gender}'s clothing.")
    elif top_gender_requested:
        parts.append(f"They usually style for {top_gender_requested}'s wear.")
    if boldness_note:
        parts.append(f"Based on past feedback, this member {boldness_note}.")

    # 4. What they told us about themselves
    if signals is not None and not signals.is_empty:
        parts.extend(_profile_sentences(signals, today))

    if not parts:
        return NO_HISTORY_NOTE
    return " ".join(parts)
