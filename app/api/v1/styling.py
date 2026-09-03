"""Styling Pipeline API endpoints (outfit recommendation)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_db_session, get_storage
from app.models.garment import Garment
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.schemas.styling import (
    OutfitResult,
    ScoreBreakdown,
    SemanticGateResult,
    StageTrace,
    StylingIntent,
    StylingRecommendationRequest,
    StylingRecommendationResponse,
    ValidationResult,
    VisualGateResult,
)
from app.storage.base import StorageClient
from app.styling.orchestrator import StylingOrchestrator, garment_to_summary

router = APIRouter(prefix="/wardrobe/styling", tags=["Styling"])


@router.post("/recommendations", response_model=StylingRecommendationResponse)
async def get_outfit_recommendations(
    request: StylingRecommendationRequest,
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Runs the full Styling Pipeline: normalises the request, retrieves real wardrobe
    garments, checks compatibility, ranks outfits, and validates them semantically
    and visually. Never invents garments — every returned garment is a real DB row.
    """
    try:
        orchestrator = StylingOrchestrator(session, storage)
        return await orchestrator.run(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/requests/{request_id}", response_model=StylingRecommendationResponse)
async def get_styling_request(
    request_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Replays a past recommendation result from persisted Outfit/OutfitGarment rows."""
    styling_request = await session.get(StylingRequest, request_id)
    if not styling_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Styling request '{request_id}' not found")

    stmt = (
        select(Outfit)
        .where(Outfit.request_id == request_id)
        .order_by(Outfit.rank.asc())
    )
    res = await session.execute(stmt)
    outfits = res.scalars().all()

    outfit_results = []
    for outfit in outfits:
        og_res = await session.execute(select(OutfitGarment).where(OutfitGarment.outfit_id == outfit.id))
        outfit_garments = og_res.scalars().all()

        garment_ids = [og.garment_id for og in outfit_garments]
        garments_stmt = select(Garment).where(Garment.id.in_(garment_ids))
        garments_res = await session.execute(garments_stmt)
        garments_by_id = {g.id: g for g in garments_res.scalars().all()}
        roles = {og.garment_id: og.role for og in outfit_garments}

        generated_image_url = (
            f"/api/v1/wardrobe/images/{outfit.generated_image_id}/bytes" if outfit.generated_image_id else None
        )

        outfit_results.append(
            OutfitResult(
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
        )

    return StylingRecommendationResponse(
        request_id=styling_request.id,
        intent=StylingIntent.model_validate(styling_request.normalized_intent or {}),
        outfits=outfit_results,
        trace=[StageTrace.model_validate(t) for t in (styling_request.trace or [])],
    )
