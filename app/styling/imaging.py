"""Styling Stages 9-10: Outfit Image Generation + Visual Validation.

Runs only on the final, semantically-validated top-k outfits (never on the full
candidate/combination set — PRD Section 26 latency strategy). On a validation
failure, retries generation up to STYLING_IMAGE_MAX_RETRIES times; if still
unresolved, the outfit is persisted/returned without an image rather than
failing the whole request (PRD Section 27).
"""

from typing import Dict, List, Optional, Tuple
from app.config import settings
from app.models.garment import Garment
from app.observability import logger
from app.providers.outfit_imaging import get_outfit_image_provider
from app.providers.visual_validator import get_visual_validator_provider
from app.schemas.styling import GarmentSummary, ValidationResult, ValidationStatus
from app.storage.base import StorageClient


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


async def generate_and_validate_outfit_image(
    garments: List[Garment],
    roles: Dict[str, str],
    storage: StorageClient,
) -> Tuple[Optional[bytes], Optional[ValidationResult]]:
    """Generates a composite outfit image and visually validates it, retrying on mismatch."""
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
        return None, None

    summaries = [_to_summary(g, roles.get(g.id, "")) for g in garments]
    image_provider = get_outfit_image_provider()
    validator = get_visual_validator_provider()

    attempts = settings.STYLING_IMAGE_MAX_RETRIES
    last_image: Optional[bytes] = None
    last_result: Optional[ValidationResult] = None

    for _attempt in range(max(1, attempts)):
        generated = await image_provider.generate(summaries, canonical_images)
        if not generated:
            continue

        result = await validator.validate_image(generated, summaries)
        last_image, last_result = generated, result

        if result.status in (ValidationStatus.PASS, ValidationStatus.NEEDS_REVIEW):
            return generated, result

    # Retries exhausted without a clean PASS/NEEDS_REVIEW — return structured outfit without imagery.
    if last_result and last_result.status == ValidationStatus.FAIL:
        return None, last_result
    return last_image, last_result
