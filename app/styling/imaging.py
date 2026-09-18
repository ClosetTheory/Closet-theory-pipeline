"""Styling Stage 9 (Generation) + Stage 10 (Visual Gate + Semantic Gate, in parallel).

Runs only on the final, semantically-pre-validated top-k outfits (never on the full
candidate/combination set — SPEC.md Section 26/latency strategy). SPEC.md Section 36:
the Visual Gate and Semantic Gate evaluate the GENERATED result and may execute in
parallel; on failure (either gate), generation is retried up to
STYLING_IMAGE_MAX_RETRIES times before giving up on this outfit candidate.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.persona import Persona
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


async def load_persona_portrait(
    session: AsyncSession, storage: StorageClient, tenant_id: str
) -> Optional[bytes]:
    """The portrait to render this tenant's outfits on, or None for an ordinary member.

    An evaluation character's tenant_id IS its backing account's user_id (see
    app/api/dependencies.py's ActingScope and scripts/seed_personas.py), so this is a single
    indexed no-op query for every member who isn't a character. Shared by
    StylingOrchestrator.run and swap.py's _apply_swap so a character's outfits are consistently
    themselves whether the outfit was just generated or a garment in it was swapped afterward —
    duplicating this lookup risked exactly the kind of drift where one path got it and the other
    quietly kept rendering the mannequin.
    """
    persona = (
        await session.execute(select(Persona).where(Persona.user_id == tenant_id))
    ).scalars().first()
    if not persona or not persona.portrait_image_id:
        return None
    asset = await session.get(ImageAsset, persona.portrait_image_id)
    if not asset:
        return None
    try:
        return await storage.get_object(asset.object_uri)
    except Exception as e:
        logger.warning(f"Could not load persona portrait for {tenant_id}: {e}")
        return None


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
    persona_portrait_bytes: Optional[bytes] = None,
) -> Tuple[Optional[bytes], Optional[VisualGateResult], Optional[SemanticGateResult], bool]:
    """
    Generates a composite outfit image, then runs the Visual Gate and (generated-image-aware)
    Semantic Gate in parallel on the result. Retries generation on gate failure up to
    STYLING_IMAGE_MAX_RETRIES times. Returns (image_bytes, visual_gate, semantic_gate, passed).

    `persona_portrait_bytes`: passed straight through to the image provider — see
    BaseOutfitImageProvider.generate for what it does. None for every ordinary member; the
    caller (StylingOrchestrator.run) only supplies it when tenant_id belongs to an evaluation
    character with a generated portrait.

    image_bytes is returned even when passed=False (the last attempt's generated image) so
    the caller can persist and surface rejected candidates for inspection/debugging rather
    than silently discarding what was actually generated (SPEC.md Section 27 only requires
    that a FAILED candidate not be presented as the chosen outfit — it doesn't require
    throwing away the evidence of why it failed).
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
        generated = await image_provider.generate(summaries, canonical_images, persona_portrait_bytes)
        if not generated:
            continue

        visual_result, semantic_result = await asyncio.gather(
            visual_validator.validate_image(generated, summaries),
            semantic_validator.validate_generated(context, outfit, summaries, generated),
        )
        last_image, last_visual, last_semantic = generated, visual_result, semantic_result

        if gates_passed(visual_result, semantic_result):
            return generated, visual_result, semantic_result, True

    # Retries exhausted without both gates passing — SPEC.md Section 27: the outfit is not
    # accepted as the chosen result, but the last generated attempt is still returned so it
    # can be shown as a rejected candidate rather than discarded.
    return last_image, last_visual, last_semantic, False
