"""Styling Stage 4: Attribute Candidate Filtering.

Cheap, deterministic database filtering to remove obviously irrelevant garments
before expensive retrieval/compatibility/model calls. Never sends the whole
wardrobe to an LLM/VLM.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.garment import Garment
from app.rules.member_signals import garment_constraint_violations
from app.schemas.styling import StylingIntent

ACCEPTABLE_QUALITY_STATUSES = ("APPROVED", "PENDING")


@dataclass
class ConstraintFilterResult:
    kept: List[Garment]
    dropped: int = 0
    # garment_id -> the constraints it violates, for the trace card and for anchors that were
    # kept despite violating (the stylist chose them; the violation is still worth naming).
    violations: Dict[str, List[str]] = field(default_factory=dict)
    fell_back: bool = False


def apply_hard_constraints(
    candidates: List[Garment],
    hard_constraints: Iterable[str],
    exempt_ids: Optional[Set[str]] = None,
) -> ConstraintFilterResult:
    """Drops every candidate whose attributes plainly violate one of the member's hard
    constraints (app.rules.member_signals decides what "plainly" means per constraint).

    Same never-to-zero rule as the soft filters below: a wardrobe where *everything* breaks a
    constraint is a data problem to surface, not a reason to return nothing — the pool is left
    untouched and `fell_back` says so. Exempt ids (anchors) are always kept."""
    constraints = [c for c in hard_constraints if c]
    if not constraints or not candidates:
        return ConstraintFilterResult(kept=list(candidates))
    exempt = exempt_ids or set()
    kept: List[Garment] = []
    violations: Dict[str, List[str]] = {}
    for g in candidates:
        hits = garment_constraint_violations(g.attributes_json or {}, constraints)
        if hits:
            violations[g.id] = hits
        if not hits or g.id in exempt:
            kept.append(g)
    dropped = len(candidates) - len(kept)
    if not kept:
        return ConstraintFilterResult(kept=list(candidates), dropped=0, violations=violations, fell_back=True)
    return ConstraintFilterResult(kept=kept, dropped=dropped, violations=violations)


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
