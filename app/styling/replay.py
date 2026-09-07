"""Builds an OutfitResult (or a full StylingRecommendationResponse) from persisted Outfit/
OutfitGarment rows.

Extracted from app/api/v1/styling.py::get_styling_request so the same "read back what was
already generated" logic can be shared with Outfit-of-the-Day's cache-hit path and the
garment-swap feature (app/styling/swap.py) without duplicating it.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.garment import Garment
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.schemas.styling import (
    OutfitResult,
    ScoreBreakdown,
    SemanticGateResult,
    StageTrace,
    StylingIntent,
    StylingRecommendationResponse,
    ValidationResult,
    VisualGateResult,
)
from app.styling.orchestrator import garment_to_summary


async def build_outfit_result(session: AsyncSession, outfit: Outfit) -> OutfitResult:
    og_res = await session.execute(select(OutfitGarment).where(OutfitGarment.outfit_id == outfit.id))
    outfit_garments = og_res.scalars().all()

    garment_ids = [og.garment_id for og in outfit_garments]
    garments_res = await session.execute(select(Garment).where(Garment.id.in_(garment_ids)))
    garments_by_id = {g.id: g for g in garments_res.scalars().all()}
    roles = {og.garment_id: og.role for og in outfit_garments}

    generated_image_url = (
        f"/api/v1/wardrobe/images/{outfit.generated_image_id}/bytes" if outfit.generated_image_id else None
    )

    return OutfitResult(
        outfit_id=outfit.id,
        rank=outfit.rank,
        garments=[
            garment_to_summary(garments_by_id[gid], roles.get(gid, ""))
            for gid in garment_ids
            if gid in garments_by_id
        ],
        roles=roles,
        scores=ScoreBreakdown.model_validate(outfit.score_breakdown or {}),
        compatibility_reason=outfit.compatibility_reason,
        semantic_validation=ValidationResult.model_validate(outfit.semantic_validation) if outfit.semantic_validation else None,
        generated_image_url=generated_image_url,
        visual_gate=VisualGateResult.model_validate(outfit.visual_validation) if outfit.visual_validation else None,
        generation_semantic_gate=SemanticGateResult.model_validate(outfit.generated_image_semantic_validation) if outfit.generated_image_semantic_validation else None,
    )


async def replay_styling_request(session: AsyncSession, styling_request: StylingRequest) -> StylingRecommendationResponse:
    stmt = select(Outfit).where(Outfit.request_id == styling_request.id).order_by(Outfit.rank.asc())
    outfits = (await session.execute(stmt)).scalars().all()

    return StylingRecommendationResponse(
        request_id=styling_request.id,
        intent=StylingIntent.model_validate(styling_request.normalized_intent or {}),
        outfits=[await build_outfit_result(session, outfit) for outfit in outfits],
        trace=[StageTrace.model_validate(t) for t in (styling_request.trace or [])],
    )
