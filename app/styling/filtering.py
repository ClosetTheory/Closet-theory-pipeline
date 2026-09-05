"""Styling Stage 4: Attribute Candidate Filtering.

Cheap, deterministic database filtering to remove obviously irrelevant garments
before expensive retrieval/compatibility/model calls. Never sends the whole
wardrobe to an LLM/VLM.
"""

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.garment import Garment
from app.schemas.styling import StylingIntent

ACCEPTABLE_QUALITY_STATUSES = ("APPROVED", "PENDING")


async def filter_candidates(
    session: AsyncSession,
    tenant_id: str,
    member_id: str,
    intent: StylingIntent,
) -> List[Garment]:
    """Returns fully-ingested, member-scoped garments, optionally narrowed by intent."""
    stmt = select(Garment).where(
        Garment.tenant_id == tenant_id,
        Garment.member_id == member_id,
        Garment.status == "COMPLETED",
        Garment.quality_status.in_(ACCEPTABLE_QUALITY_STATUSES),
    )
    res = await session.execute(stmt)
    candidates = list(res.scalars().all())

    if intent.colors:
        wanted = {c.lower() for c in intent.colors if c.lower() != "dark" and c.lower() != "neutral"}
        if wanted:
            candidates = [
                g for g in candidates
                if not wanted.isdisjoint({c.lower() for c in (g.attributes_json or {}).get("colour", [])})
            ] or candidates  # never over-filter to zero on a soft preference

    if intent.gender:
        wanted_gender = intent.gender.lower()
        candidates = [
            g for g in candidates
            if (g.gender or "unisex").lower() in (wanted_gender, "unisex")
        ] or candidates  # never over-filter to zero — a wardrobe with incomplete gender data should still return something

    return candidates


async def get_anchor_garments(
    session: AsyncSession,
    tenant_id: str,
    member_id: str,
    anchor_garment_ids: Optional[List[str]],
) -> List[Garment]:
    """Loads and authorization-scopes anchor (preselected) garments. Raises ValueError if any are invalid."""
    if not anchor_garment_ids:
        return []

    stmt = select(Garment).where(Garment.id.in_(anchor_garment_ids))
    res = await session.execute(stmt)
    found = {g.id: g for g in res.scalars().all()}

    anchors = []
    for gid in anchor_garment_ids:
        garment = found.get(gid)
        if not garment or garment.tenant_id != tenant_id or garment.member_id != member_id:
            raise ValueError(f"Anchor garment '{gid}' not found or not accessible to this member.")
        anchors.append(garment)
    return anchors
