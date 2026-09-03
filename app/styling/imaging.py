"""Styling Stage 9 (Generation) + Stage 10 (Visual Gate + Semantic Gate, in parallel).

Runs only on the final, semantically-pre-validated top-k outfits (never on the full
candidate/combination set — SPEC.md Section 26/latency strategy). SPEC.md Section 36:
the Visual Gate and Semantic Gate evaluate the GENERATED result and may execute in
parallel; on failure (either gate), generation is retried up to
STYLING_IMAGE_MAX_RETRIES times before giving up on this outfit candidate.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from app.config import settings
from app.models.garment import Garment
from app.observability import logger
from app.providers.outfit_imaging import get_outfit_image_provider
from app.providers.semantic_validator import get_semantic_validator_provider
from app.providers.visual_validator import get_visual_validator_provider
from app.schemas.styling import (
    GarmentSummary,
    OutfitCandidate,
    SemanticGateResult,
    StylingContext,
    VisualGateResult,
)
from app.storage.base import StorageClient

VISUAL_GATE_PASS_THRESHOLD = 6.0


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


def gates_passed(visual: Optional[VisualGateResult], semantic: Optional[SemanticGateResult]) -> bool:
    if visual is None or semantic is None:
        return False
    return visual.score >= VISUAL_GATE_PASS_THRESHOLD and semantic.status == "PASS"


async def generate_and_run_gates(
    context: StylingContext,
    outfit: OutfitCandidate,
    garments: List[Garment],
    storage: StorageClient,
) -> Tuple[Optional[bytes], Optional[VisualGateResult], Optional[SemanticGateResult], bool]:
    """
    Generates a composite outfit image, then runs the Visual Gate and (generated-image-aware)
    Semantic Gate in parallel on the result. Retries generation on gate failure up to
    STYLING_IMAGE_MAX_RETRIES times. Returns (image_bytes, visual_gate, semantic_gate, passed).
    """
    canonical_images = []
    for garment in garments:
        image_asset = garment.canonical_image
        if not image_asset:
            continue
        try:
            canonical_images.append(await storage.get_object(image_asset.object_uri))
        except Exception as e:
            logger.warning(f"Could not load canonical image for garment {garment.id}: {e}")

    if not canonical_images:
        return None, None, None, False

    summaries = [_to_summary(g, outfit.roles.get(g.id, "")) for g in garments]
    image_provider = get_outfit_image_provider()
    visual_validator = get_visual_validator_provider()
    semantic_validator = get_semantic_validator_provider()

    attempts = max(1, settings.STYLING_IMAGE_MAX_RETRIES)
    last_image: Optional[bytes] = None
    last_visual: Optional[VisualGateResult] = None
    last_semantic: Optional[SemanticGateResult] = None

    for _attempt in range(attempts):
        generated = await image_provider.generate(summaries, canonical_images)
        if not generated:
            continue

        visual_result, semantic_result = await asyncio.gather(
            visual_validator.validate_image(generated, summaries),
            semantic_validator.validate_generated(context, outfit, summaries, generated),
        )
        last_image, last_visual, last_semantic = generated, visual_result, semantic_result

        if gates_passed(visual_result, semantic_result):
            return generated, visual_result, semantic_result, True

    # Retries exhausted without both gates passing — SPEC.md Section 27: return the
    # structured outfit without imagery rather than silently accepting a failed result.
    return None, last_visual, last_semantic, False
