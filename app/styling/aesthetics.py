"""Styling Stage 7 (ranking): holistic aesthetic scoring for candidate outfits.

Distinct from _visual_harmony() in app/styling/ranking.py (a pairwise "won't clash" average
via app/rules/visual.py) — this judges each candidate outfit ONCE, as a whole composition,
the way a stylist actually reasons about a look.
"""

import asyncio
from typing import Dict, List
from app.models.garment import Garment
from app.providers.aesthetic import get_aesthetic_provider
from app.schemas.styling import AestheticScoreResult, GarmentSummary, OutfitCandidate, StylingContext


def _to_summary(garment: Garment, role: str) -> GarmentSummary:
    return GarmentSummary(
        garment_id=garment.id,
        category=garment.category,
        subcategory=garment.subcategory,
        garment_class=garment.garment_class,
        role=role,
        attributes=garment.attributes_json,
        status=garment.status,
        quality_status=garment.quality_status,
    )


async def score_outfits_aesthetics(
    candidates: List[OutfitCandidate],
    context: StylingContext,
    garments_by_id: Dict[str, Garment],
) -> Dict[str, AestheticScoreResult]:
    """Scores every candidate concurrently (independent calls); returns outfit_id/index-keyed
    results by the same identity used elsewhere in ranking — the garment_ids tuple, since
    candidates here may not yet have an outfit_id assigned."""
    provider = get_aesthetic_provider()

    async def _score_one(candidate: OutfitCandidate) -> AestheticScoreResult:
        garments = [garments_by_id[gid] for gid in candidate.garment_ids]
        summaries = [_to_summary(g, candidate.roles.get(g.id, "")) for g in garments]
        return await provider.score_outfit(summaries, context)

    results = await asyncio.gather(*(_score_one(c) for c in candidates))
    return {_candidate_key(c): r for c, r in zip(candidates, results)}


def _candidate_key(candidate: OutfitCandidate) -> str:
    return "|".join(sorted(candidate.garment_ids))
