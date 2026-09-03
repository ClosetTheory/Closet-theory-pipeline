"""Styling Pipeline API endpoints (outfit recommendation)."""

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user, get_db_session, get_storage
from app.models.garment import Garment
from app.models.style_profile import StyleProfile
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.models.user import User
from app.rules.style_profile import outfit_boldness, update_attribute_affinities, update_boldness_preference
from app.schemas.styling import (
    OutfitResult,
    OutfitVoteRequest,
    OutfitVoteResponse,
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
    current_user: User = Depends(get_current_user),
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
        return await orchestrator.run(request, current_user.tenant_id, current_user.member_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/recommendations/stream")
async def stream_outfit_recommendations(
    request: StylingRecommendationRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Identical pipeline to /recommendations, but streams each stage's completion as a
    Server-Sent Event the moment it actually happens, instead of the client blocking
    on one long request with no feedback until everything finishes.

    Event shapes (each a `data: <json>\\n\\n` line):
      {"type": "stage", "stage": <StageTrace>}
      {"type": "done", "result": <StylingRecommendationResponse>}
      {"type": "error", "message": str}
    """
    queue: "asyncio.Queue[tuple]" = asyncio.Queue()

    async def on_stage(entry) -> None:
        await queue.put(("stage", entry))

    async def runner() -> None:
        try:
            orchestrator = StylingOrchestrator(session, storage, on_stage=on_stage)
            result = await orchestrator.run(request, current_user.tenant_id, current_user.member_id)
            await queue.put(("done", result))
        except Exception as e:
            await queue.put(("error", str(e)))

    async def event_stream():
        task = asyncio.create_task(runner())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "stage":
                    yield f"data: {json.dumps({'type': 'stage', 'stage': payload.model_dump(mode='json')})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'type': 'done', 'result': payload.model_dump(mode='json')})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                    break
        finally:
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/requests/{request_id}", response_model=StylingRecommendationResponse)
async def get_styling_request(
    request_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Replays a past recommendation result from persisted Outfit/OutfitGarment rows."""
    styling_request = await session.get(StylingRequest, request_id)
    if not styling_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Styling request '{request_id}' not found")
    if styling_request.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This styling request belongs to another account")

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


@router.post("/outfits/{outfit_id}/vote", response_model=OutfitVoteResponse)
async def vote_outfit(
    outfit_id: str,
    request: OutfitVoteRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Upvote/downvote a previously-recommended outfit. Updates the member's learned
    StyleProfile.boldness_preference (EMA toward/away from the voted outfit's boldness) and
    per-value colour/pattern affinities (also EMA, weighted by accumulated vote count) —
    see app/rules/style_profile.py.
    """
    outfit = await session.get(Outfit, outfit_id)
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Outfit '{outfit_id}' not found")
    if outfit.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This outfit belongs to another account")

    boldness = outfit_boldness((outfit.score_breakdown or {}).get("visual_harmony", 0.7))

    og_res = await session.execute(select(OutfitGarment).where(OutfitGarment.outfit_id == outfit.id))
    garment_ids = [og.garment_id for og in og_res.scalars().all()]
    garments_res = await session.execute(select(Garment).where(Garment.id.in_(garment_ids)))
    garments_attrs = [g.attributes_json or {} for g in garments_res.scalars().all()]

    profile_stmt = select(StyleProfile).where(
        StyleProfile.tenant_id == outfit.tenant_id,
        StyleProfile.member_id == outfit.member_id,
    )
    profile = (await session.execute(profile_stmt)).scalars().first()
    if not profile:
        profile = StyleProfile(tenant_id=outfit.tenant_id, member_id=outfit.member_id)
        session.add(profile)
        await session.flush()

    profile.boldness_preference = update_boldness_preference(profile.boldness_preference, boldness, request.vote)
    profile.attribute_affinities = update_attribute_affinities(profile.attribute_affinities or {}, garments_attrs, request.vote)
    profile.vote_count += 1
    await session.commit()
    await session.refresh(profile)

    return OutfitVoteResponse(
        outfit_id=outfit_id,
        vote=request.vote,
        outfit_boldness=boldness,
        boldness_preference=profile.boldness_preference,
        vote_count=profile.vote_count,
    )
