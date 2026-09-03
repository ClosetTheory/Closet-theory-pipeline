"""Styling Stage 8: Semantic Validation.

Validates whether each top-ranked outfit makes semantic sense for the request.
FAIL outfits are dropped in favor of the next-ranked survivor (PRD Section 16);
NEEDS_REVIEW outfits are kept but flagged. The validator never mutates garments.
"""

import asyncio
from typing import Dict, List, Tuple
from app.models.garment import Garment
from app.providers.semantic_validator import get_semantic_validator_provider
from app.schemas.styling import (
    GarmentSummary,
    OutfitCandidate,
    StylingContext,
    ValidationResult,
    ValidationStatus,
)


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


async def validate_outfits(
    ranked: List[OutfitCandidate],
    context: StylingContext,
    garments_by_id: Dict[str, Garment],
    top_k: int,
) -> Tuple[List[Tuple[OutfitCandidate, ValidationResult]], int]:
    """Returns (accepted outfits with their validation result, count dropped for FAIL)."""
    validator = get_semantic_validator_provider()

    # All candidates in the pool are validated CONCURRENTLY — each is an independent LLM
    # call, so there's no latency benefit to waiting on one before starting the next.
    async def _validate_one(outfit: OutfitCandidate) -> ValidationResult:
        garments = [garments_by_id[gid] for gid in outfit.garment_ids]
        summaries = [_to_summary(g, outfit.roles.get(g.id, "")) for g in garments]
        return await validator.validate(context, outfit, summaries)

    results = await asyncio.gather(*(_validate_one(outfit) for outfit in ranked))

    accepted: List[Tuple[OutfitCandidate, ValidationResult]] = []
    dropped = 0
    for outfit, result in zip(ranked, results):
        if len(accepted) >= top_k:
            break
        if result.status == ValidationStatus.FAIL:
            dropped += 1
            continue
        accepted.append((outfit, result))

    return accepted, dropped
