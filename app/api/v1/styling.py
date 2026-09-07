"""Styling Pipeline API endpoints (outfit recommendation)."""

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user, get_db_session, get_storage
from app.models.garment import Garment
from app.models.ootd import OOTDSubscription
from app.models.style_profile import StyleProfile
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.models.user import User
from app.rules.style_profile import (
    confidence_for_count,
    outfit_boldness,
    update_attribute_affinities,
    update_boldness_preference,
)
from app.schemas.styling import (
    AttributeAffinityValue,
    OOTDSubscriptionRequest,
    OOTDSubscriptionResponse,
    OutfitOfTheDayRequest,
    OutfitOfTheDayResponse,
    OutfitVoteRequest,
    OutfitVoteResponse,
    StyleProfileResponse,
    StylingIntent,
    StylingRecommendationRequest,
    StylingRecommendationResponse,
)
from app.storage.base import StorageClient
from app.styling.ootd import DEFAULT_PERSONA, get_or_generate_ootd
from app.styling.orchestrator import StylingOrchestrator
from app.styling.replay import replay_styling_request

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

    return await replay_styling_request(session, styling_request)


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


@router.get("/profile", response_model=StyleProfileResponse)
async def get_style_profile(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """The authenticated user's learned styling preferences — boldness plus every tracked
    categorical attribute's per-value affinity (see app/rules/style_profile.py)."""
    profile_stmt = select(StyleProfile).where(
        StyleProfile.tenant_id == current_user.tenant_id,
        StyleProfile.member_id == current_user.member_id,
    )
    profile = (await session.execute(profile_stmt)).scalars().first()
    if not profile:
        return StyleProfileResponse(boldness_preference=0.0, vote_count=0, attribute_affinities={})

    affinities = {}
    for field, values in (profile.attribute_affinities or {}).items():
        entries = [
            AttributeAffinityValue(
                value=value,
                score=stats.get("score", 0.0),
                count=stats.get("count", 0),
                confidence=confidence_for_count(stats.get("count", 0)),
            )
            for value, stats in values.items()
        ]
        entries.sort(key=lambda e: e.count * abs(e.score), reverse=True)
        affinities[field] = entries

    return StyleProfileResponse(
        boldness_preference=profile.boldness_preference,
        vote_count=profile.vote_count,
        attribute_affinities=affinities,
    )


@router.post("/outfit-of-the-day", response_model=OutfitOfTheDayResponse)
async def generate_outfit_of_the_day(
    request: OutfitOfTheDayRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Today's weather-aware, persona-aware outfit pick (3 ranked options) — runs the exact same
    /recommendations pipeline under the hood with a synthesized request combining real current
    weather (fetched live for `location`) and `persona` context, e.g. "office_going". Returns
    the already-generated pick for today if one exists, unless force_regenerate is set.
    """
    return await get_or_generate_ootd(
        session, storage, current_user.tenant_id, current_user.member_id,
        location=request.location, persona=request.persona, force=request.force_regenerate,
        generation_source="on_demand",
    )


@router.get("/outfit-of-the-day", response_model=OutfitOfTheDayResponse)
async def read_outfit_of_the_day(
    location: str,
    persona: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Convenience GET form of POST /outfit-of-the-day (e.g. for a browser/dashboard to just
    load a URL) — always returns today's cached pick if one exists; never force-regenerates."""
    return await get_or_generate_ootd(
        session, storage, current_user.tenant_id, current_user.member_id,
        location=location, persona=persona, force=False, generation_source="on_demand",
    )


@router.put("/outfit-of-the-day/subscription", response_model=OOTDSubscriptionResponse)
async def set_ootd_subscription(
    request: OOTDSubscriptionRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Opts this member into (or out of) fully automatic daily outfit-of-the-day generation — a
    background loop (app/worker/ootd_scheduler.py) runs once a day for every enabled
    subscription and pre-generates that day's pick, so it's already there before anyone checks.
    """
    stmt = select(OOTDSubscription).where(
        OOTDSubscription.tenant_id == current_user.tenant_id,
        OOTDSubscription.member_id == current_user.member_id,
    )
    sub = (await session.execute(stmt)).scalars().first()
    if not sub:
        sub = OOTDSubscription(tenant_id=current_user.tenant_id, member_id=current_user.member_id, location=request.location)
        session.add(sub)
    sub.location = request.location
    sub.persona = request.persona or DEFAULT_PERSONA
    sub.enabled = request.enabled
    await session.commit()
    return OOTDSubscriptionResponse(location=sub.location, persona=sub.persona, enabled=sub.enabled)


@router.get("/outfit-of-the-day/subscription", response_model=OOTDSubscriptionResponse)
async def get_ootd_subscription(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """This member's current outfit-of-the-day auto-generation settings, if any."""
    stmt = select(OOTDSubscription).where(
        OOTDSubscription.tenant_id == current_user.tenant_id,
        OOTDSubscription.member_id == current_user.member_id,
    )
    sub = (await session.execute(stmt)).scalars().first()
    if not sub:
        return OOTDSubscriptionResponse(location=None, persona=None, enabled=False)
    return OOTDSubscriptionResponse(location=sub.location, persona=sub.persona, enabled=sub.enabled)
