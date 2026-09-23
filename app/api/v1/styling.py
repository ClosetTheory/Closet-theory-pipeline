"""Styling Pipeline API endpoints (outfit recommendation)."""

import asyncio
import json
from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import ActingScope, get_acting_scope, get_db_session, get_storage
from app.database import AsyncSessionLocal
from app.models.base import utc_now
from app.models.garment import Garment
from app.models.ootd import OOTDSubscription
from app.models.persona_review import OwnOutfitReview
from app.models.style_profile import StyleProfile
from app.models.styling import Outfit, OutfitGarment, StylingRequest, StylistReview
from app.models.styling_run import StylingRunProgress
from app.models.user import User
from app.rules.style_profile import (
    confidence_for_count,
    outfit_boldness,
    update_attribute_affinities,
    update_boldness_preference,
)
from app.schemas.styling import (
    AttributeAffinityValue,
    ChatSwapRequest,
    OOTDSubscriptionRequest,
    OOTDSubscriptionResponse,
    OutfitOfTheDayRequest,
    OutfitOfTheDayResponse,
    OutfitResult,
    OutfitReviewQueueItem,
    OutfitVoteRequest,
    OutfitVoteResponse,
    OwnOutfitReviewItem,
    OwnOutfitReviewResult,
    StageTrace,
    StyleProfileResponse,
    StylingIntent,
    StylingRecommendationRequest,
    StylingRecommendationResponse,
    StylingRunProgressResponse,
    StylistReviewRequest,
    StylistReviewResult,
    SwapCandidateSummary,
    SwapGarmentRequest,
)
from app.schemas.persona import PersonaOutfitReviewRequest
from app.storage.base import StorageClient
from app.styling.ootd import get_or_generate_ootd
from app.styling.orchestrator import StylingOrchestrator
from app.styling.replay import build_outfit_result, replay_styling_request
from app.styling.swap import SwapError, list_swap_candidates, swap_garment_by_chat, swap_garment_direct

router = APIRouter(prefix="/wardrobe/styling", tags=["Styling"])


@router.post("/recommendations", response_model=StylingRecommendationResponse)
async def get_outfit_recommendations(
    request: StylingRecommendationRequest,
    scope: ActingScope = Depends(get_acting_scope),
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
        return await orchestrator.run(request, scope.tenant_id, scope.member_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


STALE_RUN_MINUTES = 20  # how long a RUNNING row can go without a checkpoint before we assume
# the process that owned it crashed and lazily mark it ABANDONED (see get_active_styling_run).


def _run_to_response(row: StylingRunProgress) -> StylingRunProgressResponse:
    return StylingRunProgressResponse(
        id=row.id,
        status=row.status,
        current_stage=row.current_stage,
        trace=[StageTrace(**t) for t in (row.trace or [])],
        styling_request_id=row.styling_request_id,
        error_message=row.error_message,
        request_payload=row.request_payload or {},
    )


async def _abandon_stale_active_runs(session: AsyncSession, tenant_id: str, member_id: str) -> None:
    """No dedicated cleanup job — a RUNNING row that's gone quiet (its owning process died
    mid-run) is flipped to ABANDONED the next time anyone asks "is anything active" for that
    scope, rather than lingering forever."""
    cutoff = utc_now() - timedelta(minutes=STALE_RUN_MINUTES)
    await session.execute(
        update(StylingRunProgress)
        .where(
            StylingRunProgress.tenant_id == tenant_id,
            StylingRunProgress.member_id == member_id,
            StylingRunProgress.status == "RUNNING",
            StylingRunProgress.updated_at < cutoff,
        )
        .values(status="ABANDONED")
    )
    await session.commit()


@router.post("/recommendations/stream")
async def stream_outfit_recommendations(
    request: StylingRecommendationRequest,
    scope: ActingScope = Depends(get_acting_scope),
    storage: StorageClient = Depends(get_storage),
):
    """
    Identical pipeline to /recommendations, but streams each stage's completion as a
    Server-Sent Event the moment it actually happens, instead of the client blocking
    on one long request with no feedback until everything finishes.

    The run itself is driven on its own database session (AsyncSessionLocal(), not
    Depends(get_db_session)) and is not tied to this request's task/cancellation scope: if
    the client disconnects (e.g. a tab switch), the pipeline keeps running and keeps
    checkpointing to StylingRunProgress rather than being torn down mid-flight. A client
    can reconnect via GET /wardrobe/styling/runs/active or /runs/{run_id} and resume from
    the last checkpoint instead of losing all progress.

    Event shapes (each a `data: <json>\\n\\n` line):
      {"type": "run_started", "run_id": str}
      {"type": "stage", "stage": <StageTrace>}
      {"type": "done", "result": <StylingRecommendationResponse>}
      {"type": "error", "message": str}
    """
    queue: "asyncio.Queue[tuple]" = asyncio.Queue()
    run_session = AsyncSessionLocal()

    # A new request for this scope supersedes whatever was already running there.
    await run_session.execute(
        update(StylingRunProgress)
        .where(
            StylingRunProgress.tenant_id == scope.tenant_id,
            StylingRunProgress.member_id == scope.member_id,
            StylingRunProgress.status == "RUNNING",
        )
        .values(status="ABANDONED")
    )
    progress = StylingRunProgress(
        tenant_id=scope.tenant_id,
        member_id=scope.member_id,
        persona_id=scope.persona_id,
        initiated_by_user_id=scope.actor.id,
        request_payload=request.model_dump(mode="json"),
    )
    run_session.add(progress)
    await run_session.commit()
    run_id = progress.id

    async def checkpoint(entry) -> None:
        async with AsyncSessionLocal() as cp_session:
            row = await cp_session.get(StylingRunProgress, run_id)
            if row:
                row.current_stage = entry.stage
                row.trace = row.trace + [entry.model_dump(mode="json")]
                await cp_session.commit()

    async def on_stage(entry) -> None:
        await queue.put(("stage", entry))
        await checkpoint(entry)

    async def runner() -> None:
        try:
            orchestrator = StylingOrchestrator(run_session, storage, on_stage=on_stage)
            result = await orchestrator.run(request, scope.tenant_id, scope.member_id)
            async with AsyncSessionLocal() as cp_session:
                row = await cp_session.get(StylingRunProgress, run_id)
                if row:
                    row.status = "SUCCEEDED"
                    row.styling_request_id = result.request_id
                    await cp_session.commit()
            await queue.put(("done", result))
        except Exception as e:
            async with AsyncSessionLocal() as cp_session:
                row = await cp_session.get(StylingRunProgress, run_id)
                if row:
                    row.status = "FAILED"
                    row.error_message = str(e)
                    await cp_session.commit()
            await queue.put(("error", str(e)))
        finally:
            await run_session.close()

    async def event_stream():
        # Deliberately not awaited here or in a finally block below: the task owns its own
        # session and keeps running (and checkpointing) regardless of whether this generator
        # is still being consumed — that decoupling is the whole point of this change.
        asyncio.create_task(runner())
        yield f"data: {json.dumps({'type': 'run_started', 'run_id': run_id})}\n\n"
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Keep reverse proxies and browsers from treating a long image-generation
                # stage as an idle/dead HTTP response. SSE comments are ignored by the UI
                # but still flush bytes over the connection.
                yield ": heartbeat\n\n"
                continue
            if kind == "stage":
                yield f"data: {json.dumps({'type': 'stage', 'stage': payload.model_dump(mode='json')})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done', 'result': payload.model_dump(mode='json')})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/active", response_model=Optional[StylingRunProgressResponse])
async def get_active_styling_run(
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """The in-flight run for this scope, if any — lets a client that switched tabs mid-run
    (or just reloaded) find its way back without already knowing a request_id."""
    await _abandon_stale_active_runs(session, scope.tenant_id, scope.member_id)
    row = (
        await session.execute(
            select(StylingRunProgress)
            .where(
                StylingRunProgress.tenant_id == scope.tenant_id,
                StylingRunProgress.member_id == scope.member_id,
                StylingRunProgress.status == "RUNNING",
            )
            .order_by(StylingRunProgress.created_at.desc())
        )
    ).scalars().first()
    return _run_to_response(row) if row else None


@router.get("/runs/{run_id}", response_model=StylingRunProgressResponse)
async def get_styling_run(
    run_id: str,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Poll target for a client that's already resumed via /runs/active (or that has the id
    from this run's own `run_started` event) — the checkpointed trace plus current status."""
    row = await session.get(StylingRunProgress, run_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Styling run '{run_id}' not found")
    if row.tenant_id != scope.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This run belongs to another account")
    return _run_to_response(row)


@router.get("/requests/{request_id}", response_model=StylingRecommendationResponse)
async def get_styling_request(
    request_id: str,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Replays a past recommendation result from persisted Outfit/OutfitGarment rows."""
    styling_request = await session.get(StylingRequest, request_id)
    if not styling_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Styling request '{request_id}' not found")
    if styling_request.tenant_id != scope.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This styling request belongs to another account")

    return await replay_styling_request(session, styling_request)


@router.post("/outfits/{outfit_id}/vote", response_model=OutfitVoteResponse)
async def vote_outfit(
    outfit_id: str,
    request: OutfitVoteRequest,
    scope: ActingScope = Depends(get_acting_scope),
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
    if outfit.tenant_id != scope.tenant_id:
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


def _review_to_result(review: StylistReview) -> StylistReviewResult:
    return StylistReviewResult(
        id=review.id,
        outfit_id=review.outfit_id,
        vote=review.vote,
        comment=review.comment,
        created_at=review.created_at.isoformat(),
        updated_at=review.updated_at.isoformat(),
    )


@router.get("/review-queue", response_model=List[OutfitReviewQueueItem])
async def get_review_queue(
    limit: int = 50,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Internal QA panel (see app/static/review.html): every outfit ever generated for this
    account, most recent first, with any existing stylist review attached. Not part of the
    user-facing recommendation flow — for company stylists reviewing their own account's
    generated outfits."""
    stmt = (
        select(Outfit, StylingRequest.raw_text)
        .join(StylingRequest, StylingRequest.id == Outfit.request_id)
        .where(Outfit.tenant_id == scope.tenant_id, Outfit.member_id == scope.member_id)
        .order_by(Outfit.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()

    outfit_ids = [outfit.id for outfit, _ in rows]
    reviews_res = await session.execute(select(StylistReview).where(StylistReview.outfit_id.in_(outfit_ids)))
    reviews_by_outfit = {r.outfit_id: r for r in reviews_res.scalars().all()}

    items = []
    for outfit, raw_text in rows:
        result = await build_outfit_result(session, outfit)
        review = reviews_by_outfit.get(outfit.id)
        items.append(
            OutfitReviewQueueItem(
                outfit=result,
                request_text=raw_text,
                generated_at=outfit.created_at.isoformat(),
                review=_review_to_result(review) if review else None,
            )
        )
    return items


@router.put("/outfits/{outfit_id}/review", response_model=StylistReviewResult)
async def review_outfit(
    outfit_id: str,
    request: StylistReviewRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Records (or updates) a stylist's like/dislike + comment on one of their own account's
    generated outfits — one review per outfit, resubmitting updates it in place. Purely a QA
    record: never touches StyleProfile/ranking (see /vote for that separate mechanism)."""
    outfit = await session.get(Outfit, outfit_id)
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Outfit '{outfit_id}' not found")
    if outfit.tenant_id != scope.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This outfit belongs to another account")

    review = (await session.execute(select(StylistReview).where(StylistReview.outfit_id == outfit_id))).scalars().first()
    if review:
        review.vote = request.vote
        review.comment = request.comment
    else:
        review = StylistReview(
            outfit_id=outfit_id,
            tenant_id=outfit.tenant_id,
            member_id=outfit.member_id,
            vote=request.vote,
            comment=request.comment,
        )
        session.add(review)

    await session.commit()
    await session.refresh(review)
    return _review_to_result(review)


def _own_review_to_result(review: OwnOutfitReview, reviewer_name: Optional[str] = None) -> OwnOutfitReviewResult:
    return OwnOutfitReviewResult(
        id=review.id,
        outfit_id=review.outfit_id,
        reviewer_user_id=review.reviewer_user_id,
        reviewer_name=reviewer_name,
        rating=review.rating,
        vote=review.vote,
        dimension_ratings=review.dimension_ratings or {},
        comment=review.comment,
        would_wear=review.would_wear,
        tags=review.tags or [],
        created_at=review.created_at.isoformat(),
        updated_at=review.updated_at.isoformat(),
    )


@router.get("/outfits/reviewable", response_model=List[OwnOutfitReviewItem])
async def list_own_reviewable_outfits(
    limit: int = 50,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Every outfit generated in the acting account, newest first, with the caller's own
    five-dimension score and anyone else's alongside — the same rubric and item shape the
    character panel uses (/personas/{id}/outfits), for an admin ranking their *own* outfits.
    The /review page shows this when no character is active."""
    outfits = list((
        await session.execute(
            select(Outfit)
            .where(Outfit.tenant_id == scope.tenant_id, Outfit.member_id == scope.member_id)
            .order_by(Outfit.created_at.desc())
            .limit(limit)
        )
    ).scalars().all())
    if not outfits:
        return []

    request_ids = list({o.request_id for o in outfits if o.request_id})
    request_text_by_id: dict = {}
    used_hopit_by_id: dict = {}
    if request_ids:
        for rid, text, context in (
            await session.execute(
                select(StylingRequest.id, StylingRequest.raw_text, StylingRequest.context)
                .where(StylingRequest.id.in_(request_ids))
            )
        ).all():
            request_text_by_id[rid] = text
            used_hopit_by_id[rid] = bool((context or {}).get("used_hopit"))

    review_rows = (
        await session.execute(
            select(OwnOutfitReview, User.display_name, User.email)
            .join(User, User.id == OwnOutfitReview.reviewer_user_id)
            .where(OwnOutfitReview.outfit_id.in_([o.id for o in outfits]))
        )
    ).all()
    by_outfit: dict = {}
    for review, display_name, email in review_rows:
        by_outfit.setdefault(review.outfit_id, []).append(_own_review_to_result(review, display_name or email))

    items = []
    for outfit in outfits:
        reviews = by_outfit.get(outfit.id, [])
        items.append(
            OwnOutfitReviewItem(
                outfit=await build_outfit_result(session, outfit),
                generated_at=outfit.created_at.isoformat(),
                request_id=outfit.request_id,
                request_text=request_text_by_id.get(outfit.request_id),
                used_hopit=used_hopit_by_id.get(outfit.request_id, False),
                my_review=next((r for r in reviews if r.reviewer_user_id == scope.actor.id), None),
                reviews=reviews,
            )
        )
    return items


@router.put("/outfits/{outfit_id}/score", response_model=OwnOutfitReviewResult)
async def score_own_outfit(
    outfit_id: str,
    request: PersonaOutfitReviewRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """Upserts the caller's five-dimension score on an outfit generated in the acting account —
    one row per (outfit, reviewer), revised in place. Distinct from the older like/dislike
    /review endpoint above, which keeps one opinion per outfit and no reviewer."""
    outfit = await session.get(Outfit, outfit_id)
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Outfit '{outfit_id}' not found")
    if outfit.tenant_id != scope.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This outfit belongs to another account")

    review = (
        await session.execute(
            select(OwnOutfitReview).where(
                OwnOutfitReview.outfit_id == outfit_id,
                OwnOutfitReview.reviewer_user_id == scope.actor.id,
            )
        )
    ).scalars().first()
    if review is None:
        review = OwnOutfitReview(
            outfit_id=outfit_id,
            reviewer_user_id=scope.actor.id,
            tenant_id=outfit.tenant_id,
            member_id=outfit.member_id,
            rating=request.rating,
        )
        session.add(review)

    review.rating = request.rating
    review.vote = request.vote
    review.dimension_ratings = request.dimension_ratings
    review.comment = request.comment
    review.would_wear = request.would_wear
    review.tags = request.tags
    await session.commit()
    await session.refresh(review)
    return _own_review_to_result(review, scope.actor.display_name or scope.actor.email)


@router.get("/profile", response_model=StyleProfileResponse)
async def get_style_profile(
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """The authenticated user's learned styling preferences — boldness plus every tracked
    categorical attribute's per-value affinity (see app/rules/style_profile.py)."""
    profile_stmt = select(StyleProfile).where(
        StyleProfile.tenant_id == scope.tenant_id,
        StyleProfile.member_id == scope.member_id,
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
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Today's weather-aware outfit pick (3 ranked options) — runs the exact same
    /recommendations pipeline under the hood with a synthesized request combining real current
    weather (fetched live for `location`) and a context paragraph auto-derived from this
    member's real styling history and wardrobe (no manual persona/preset needed — see
    app/styling/member_context.py). Returns the already-generated pick for today if one
    exists, unless force_regenerate is set.
    """
    return await get_or_generate_ootd(
        session, storage, scope.tenant_id, scope.member_id,
        location=request.location, extra_hint=request.extra_hint, force=request.force_regenerate,
        generation_source="on_demand", use_hopit=request.use_hopit,
    )


@router.get("/outfit-of-the-day", response_model=OutfitOfTheDayResponse)
async def read_outfit_of_the_day(
    location: str,
    extra_hint: Optional[str] = None,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Convenience GET form of POST /outfit-of-the-day (e.g. for a browser/dashboard to just
    load a URL) — always returns today's cached pick if one exists; never force-regenerates."""
    return await get_or_generate_ootd(
        session, storage, scope.tenant_id, scope.member_id,
        location=location, extra_hint=extra_hint, force=False, generation_source="on_demand",
    )


@router.put("/outfit-of-the-day/subscription", response_model=OOTDSubscriptionResponse)
async def set_ootd_subscription(
    request: OOTDSubscriptionRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Opts this member into (or out of) fully automatic daily outfit-of-the-day generation — a
    background loop (app/worker/ootd_scheduler.py) runs once a day for every enabled
    subscription and pre-generates that day's pick, so it's already there before anyone checks.
    """
    stmt = select(OOTDSubscription).where(
        OOTDSubscription.tenant_id == scope.tenant_id,
        OOTDSubscription.member_id == scope.member_id,
    )
    sub = (await session.execute(stmt)).scalars().first()
    if not sub:
        sub = OOTDSubscription(tenant_id=scope.tenant_id, member_id=scope.member_id, location=request.location)
        session.add(sub)
    sub.location = request.location
    sub.extra_hint = request.extra_hint
    sub.enabled = request.enabled
    await session.commit()
    return OOTDSubscriptionResponse(location=sub.location, extra_hint=sub.extra_hint, enabled=sub.enabled)


@router.get("/outfit-of-the-day/subscription", response_model=OOTDSubscriptionResponse)
async def get_ootd_subscription(
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """This member's current outfit-of-the-day auto-generation settings, if any."""
    stmt = select(OOTDSubscription).where(
        OOTDSubscription.tenant_id == scope.tenant_id,
        OOTDSubscription.member_id == scope.member_id,
    )
    sub = (await session.execute(stmt)).scalars().first()
    if not sub:
        return OOTDSubscriptionResponse(location=None, extra_hint=None, enabled=False)
    return OOTDSubscriptionResponse(location=sub.location, extra_hint=sub.extra_hint, enabled=sub.enabled)


@router.get("/outfits/{outfit_id}/swap/candidates", response_model=List[SwapCandidateSummary])
async def get_swap_candidates(
    outfit_id: str,
    role: str,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Browse-and-pick option: this member's own real garments for `role` (e.g. 'FOOTWEAR') that
    aren't already in this outfit, so a caller can pick one and then call
    POST /outfits/{id}/swap with its garment_id — no compatibility check yet, that happens on
    the actual swap so a caller can see all real options before committing.
    """
    try:
        candidates = await list_swap_candidates(session, scope.tenant_id, scope.member_id, outfit_id, role)
    except SwapError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    return [
        SwapCandidateSummary(
            garment_id=g.id,
            subcategory=g.subcategory,
            canonical_image_url=f"/api/v1/wardrobe/images/{g.canonical_image_id}/bytes" if g.canonical_image_id else None,
        )
        for g in candidates
    ]


@router.post("/outfits/{outfit_id}/swap", response_model=OutfitResult)
async def swap_outfit_garment(
    outfit_id: str,
    request: SwapGarmentRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Replaces one role in an already-generated outfit with a specific real garment from this
    member's wardrobe (see GET .../swap/candidates to browse options first). Re-checks
    compatibility with the rest of the outfit — a genuinely incompatible replacement is
    rejected (409) rather than silently applied — and regenerates the outfit's composite image.
    """
    try:
        return await swap_garment_direct(
            session, storage, scope.tenant_id, scope.member_id,
            outfit_id, request.role, request.new_garment_id,
        )
    except SwapError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/outfits/{outfit_id}/swap/chat", response_model=OutfitResult)
async def chat_swap_outfit_garment(
    outfit_id: str,
    request: ChatSwapRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Same swap, described in free text instead of a garment_id, e.g. "the shoes feel too
    casual, give me something else" or "swap the top for something in a darker colour".
    Identifies which role is meant and picks the best-compatible real match from this
    member's own wardrobe — never invents a garment. Raises 422 if it can't confidently tell
    which part of the outfit is meant (asks to be more specific rather than guessing wrong).
    """
    try:
        return await swap_garment_by_chat(
            session, storage, scope.tenant_id, scope.member_id,
            outfit_id, request.instruction,
        )
    except SwapError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
