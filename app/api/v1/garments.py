"""Garment and Pipeline execution API endpoints."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user, get_db_session, get_storage
from app.config import settings
from app.models.base import utc_now
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.pipeline_stage import PipelineStageRun
from app.models.user import User
from app.observability import log_stage_event, metrics
from app.pipeline.lock import GarmentLockBusy, garment_execution_lock
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.stages.base import StageExecutionContext
from app.pipeline.state_machine import (
    PIPELINE_STAGES_ORDER,
    STAGE_TO_GARMENT_STATE,
    GarmentState,
    PipelineStage,
)
from app.api.v1.images import store_uploaded_image
from app.schemas.attributes import validate_extracted_attributes
from app.schemas.garment import (
    BulkGarmentUploadResponse,
    BulkGarmentUploadResult,
    CanonicalGarment,
    CoordinatedGarment,
    GarmentCreateRequest,
)
from app.schemas.styling import GarmentSummary
from app.schemas.pipeline import (
    PipelineStageRunRead,
    PipelineStatusResponse,
    RetryRequest,
    ReviewDecision,
    ReviewRequest,
)
from app.storage.base import StorageClient
from app.worker.queue import enqueue_garment_pipeline

router = APIRouter(prefix="/wardrobe/garments", tags=["Garments"])

# Guards the mutate/execute/restore window in execute_single_pipeline_step's per-request API
# key override — see the comment at its use site.
_step_api_key_lock = asyncio.Lock()

# Linear pipeline progress order (excludes FAILED/REVIEW_REQUIRED, which aren't part of forward
# progress) — used so a single-stage re-run can only ever advance garment.status, never regress
# it. Declaration order in GarmentState already matches pipeline progress.
_GARMENT_STATE_ORDER = {
    s.value: i for i, s in enumerate(GarmentState)
    if s not in (GarmentState.FAILED, GarmentState.REVIEW_REQUIRED)
}


def _garment_state_order(status_value: str) -> int:
    """-1 for REVIEW_REQUIRED/FAILED/unknown so a fresh success can always move a garment
    forward out of those states, same as landing there from any earlier successful stage."""
    return _GARMENT_STATE_ORDER.get(status_value, -1)


class StepRequest(BaseModel):
    stage: Optional[str] = Field(default=None, description="Specific stage to execute")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key (sk-or-v1-...)")
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA NIM API key")
    force: bool = Field(default=False, description="Force re-run even if already completed")


@router.post("", response_model=CanonicalGarment, status_code=status.HTTP_202_ACCEPTED)
async def create_garment(
    request: GarmentCreateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Initiates asynchronous ingestion pipeline for an uploaded image.
    Acknowledges quickly with initial Garment representation.
    """
    source_image = await session.get(ImageAsset, request.source_image_id)
    if not source_image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source image '{request.source_image_id}' not found",
        )
    if source_image.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Source image belongs to another account")

    garment = Garment(
        tenant_id=current_user.tenant_id,
        member_id=current_user.member_id,
        source_image_id=source_image.id,
        status=GarmentState.RECEIVED.value,
        quality_status="PENDING",
    )
    session.add(garment)
    await session.commit()
    await session.refresh(garment)

    # Enqueue pipeline execution asynchronously — unless the caller intends to drive every
    # stage itself (the interactive demo UI), in which case auto-enqueuing here would race the
    # background worker against the caller's own manual /step calls on the same garment.
    if request.auto_process:
        await enqueue_garment_pipeline(garment.id)

    return CanonicalGarment(
        garment_id=garment.id,
        source_image_refs=[source_image.object_uri],
        image_type=garment.image_type,
        garment_crop_refs=garment.garment_crop_refs,
        attributes=garment.attributes_json or None,
        canonical_image_ref=None,
        image_embedding=None,
        category=garment.category,
        compatibility_features=garment.compatibility_features,
        quality_status=garment.quality_status,
        provenance=garment.provenance,
        pipeline_version=garment.pipeline_version,
    )


@router.post("/bulk", response_model=BulkGarmentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def bulk_create_garments(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Bulk ingestion: uploads and queues many garment photos in one request. Each file is
    validated and enqueued independently via the SAME background worker queue single uploads
    use (app/worker/queue.py::enqueue_garment_pipeline) — the existing Redis-backed worker pool
    (settings.WORKER_CONCURRENCY concurrent loops, see app/worker/runner.py) then processes them
    in parallel. A bad file (wrong MIME type, corrupt, oversized) is recorded as a per-file error
    and does not abort the rest of the batch.
    """
    if len(files) > settings.MAX_BULK_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch of {len(files)} files exceeds the maximum of {settings.MAX_BULK_UPLOAD_FILES} per request.",
        )

    results: List[BulkGarmentUploadResult] = []
    for file in files:
        filename = file.filename or "unnamed"
        try:
            image_asset = await store_uploaded_image(file, current_user, session, storage)
        except HTTPException as e:
            results.append(BulkGarmentUploadResult(filename=filename, status="error", error=e.detail))
            continue

        garment = Garment(
            tenant_id=current_user.tenant_id,
            member_id=current_user.member_id,
            source_image_id=image_asset.id,
            status=GarmentState.RECEIVED.value,
            quality_status="PENDING",
        )
        session.add(garment)
        await session.commit()
        await session.refresh(garment)

        await enqueue_garment_pipeline(garment.id)
        results.append(BulkGarmentUploadResult(
            filename=filename, garment_id=garment.id, image_id=image_asset.id, status="queued",
        ))

    queued_count = sum(1 for r in results if r.status == "queued")
    return BulkGarmentUploadResponse(
        results=results,
        queued_count=queued_count,
        failed_count=len(results) - queued_count,
    )


@router.get("", response_model=List[GarmentSummary])
async def list_garments(
    category: Optional[str] = Query(default=None),
    status_filter: str = Query(default="COMPLETED", alias="status"),
    limit: int = Query(default=24, ge=1, le=3000),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Catalogue listing of ingested garments — the ingestion pipeline's real, persisted output.
    Always scoped to the authenticated user's own tenant — a wardrobe is private."""
    stmt = select(Garment).options(selectinload(Garment.canonical_image)).where(Garment.tenant_id == current_user.tenant_id)

    if category:
        stmt = stmt.where(Garment.category == category)
    if status_filter:
        stmt = stmt.where(Garment.status == status_filter)

    stmt = stmt.order_by(Garment.updated_at.desc()).offset(offset).limit(limit)
    res = await session.execute(stmt)
    garments = res.scalars().all()

    return [
        GarmentSummary(
            garment_id=g.id,
            category=g.category,
            subcategory=g.subcategory,
            garment_class=g.garment_class,
            attributes=g.attributes_json,
            canonical_image_url=f"/api/v1/wardrobe/images/{g.canonical_image_id}/bytes" if g.canonical_image_id else None,
            status=g.status,
            quality_status=g.quality_status,
            created_at=g.created_at.isoformat() if g.created_at else None,
        )
        for g in garments
    ]


_BULK_IMPORT_MODELS: Dict[str, Any] = {}


def _bulk_import_model_registry() -> Dict[str, Any]:
    """Lazily built so every model is imported (and thus registered on Base.metadata) before
    this is first used — avoids a hard import-order dependency at module load time."""
    if not _BULK_IMPORT_MODELS:
        from app.models.image_asset import ImageAsset as _ImageAsset
        from app.models.styling import StylingRequest, Outfit, OutfitGarment
        from app.models.style_profile import StyleProfile
        from app.models.ootd import OutfitOfTheDay

        _BULK_IMPORT_MODELS.update({
            "image_assets": _ImageAsset,
            "garments": Garment,
            "garment_embeddings": GarmentEmbedding,
            "styling_requests": StylingRequest,
            "outfits": Outfit,
            "outfit_garments": OutfitGarment,
            "style_profiles": StyleProfile,
            "outfits_of_the_day": OutfitOfTheDay,
        })
    return _BULK_IMPORT_MODELS


@router.post("/_bulk_import/image")
async def bulk_import_image(
    id: str = Form(...),
    object_uri: str = Form(...),
    tenant_id: str = Form(...),
    member_id: str = Form(...),
    mime_type: str = Form("image/jpeg"),
    width: int = Form(0),
    height: int = Form(0),
    sha256: str = Form(""),
    created_at: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """TEMPORARY one-time dev->prod data sync helper (see POST /_bulk_import/{table_name}
    below for the rest of the tables). Writes the uploaded bytes to this environment's own
    storage backend under the EXACT SAME object_uri the source environment used (S3StorageClient
    strips any "object://bucket/" prefix and rebuilds it from its own bucket name, so as long as
    both environments use the same bucket name — true here — the resulting object_uri matches
    exactly, meaning every Garment/Outfit row that references this id by string needs no
    remapping), then upserts the ImageAsset row with the SAME id as the source, so foreign keys
    from garments/outfits carry over unchanged. Safe to re-run (overwrites by same key/id).
    Remove once the one-time sync is complete."""
    from datetime import datetime as _dt

    content = await file.read()
    stored_uri = await storage.put_object(object_uri, content, content_type=mime_type)

    existing = await session.get(ImageAsset, id)
    parsed_created_at = _dt.fromisoformat(created_at) if created_at else utc_now()
    if existing:
        existing.object_uri = stored_uri
        existing.mime_type = mime_type
        existing.width = width
        existing.height = height
        existing.sha256 = sha256
    else:
        session.add(ImageAsset(
            id=id, tenant_id=tenant_id, member_id=member_id, object_uri=stored_uri,
            mime_type=mime_type, width=width, height=height, sha256=sha256,
            created_at=parsed_created_at,
        ))
    await session.commit()
    return {"id": id, "object_uri": stored_uri, "bytes": len(content)}


@router.post("/_bulk_import/{table_name}")
async def bulk_import_rows(
    table_name: str,
    rows: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """TEMPORARY one-time dev->prod data sync helper — see POST /_bulk_import/image above for
    the images/bytes half. Upserts (by primary key) into any of the non-image tables, using
    SQLAlchemy's merge() so each column's real Python type (including the pgvector embedding
    column and JSON columns) is handled correctly rather than hand-writing per-table SQL.
    Row dicts must be exactly what GET-ing the same model's columns would serialize to, with
    datetimes as ISO strings (auto-parsed below via each column's declared Python type). Never
    deletes anything; only ever inserts or overwrites by matching id. Remove once the one-time
    sync is complete."""
    from datetime import date as _date, datetime as _dt

    model_cls = _bulk_import_model_registry().get(table_name)
    if not model_cls:
        raise HTTPException(status_code=400, detail=f"Unknown bulk-import table '{table_name}'")

    imported = 0
    errors: List[str] = []
    for row in rows:
        try:
            parsed = dict(row)
            for col in model_cls.__table__.columns:
                if col.name in parsed and isinstance(parsed[col.name], str):
                    try:
                        py_type = col.type.python_type
                    except NotImplementedError:
                        continue
                    if py_type is _dt:
                        parsed[col.name] = _dt.fromisoformat(parsed[col.name])
                    elif py_type is _date:
                        parsed[col.name] = _date.fromisoformat(parsed[col.name])
            await session.merge(model_cls(**parsed))
            imported += 1
        except Exception as e:
            errors.append(f"{row.get('id', '?')}: {type(e).__name__}: {e}")
    await session.commit()
    return {"table": table_name, "imported": imported, "errors": errors}


@router.delete("/{garment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_garment(
    garment_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Permanently deletes a garment (and, via DB cascade, its pipeline stage runs, embedding,
    and any outfit references). Does not delete underlying storage objects (raw/crop/canonical
    images) — only the DB record."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Garment '{garment_id}' not found")
    if garment.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This garment belongs to another account")

    await session.delete(garment)
    await session.commit()


@router.get("/{garment_id}", response_model=CanonicalGarment)
async def get_garment(
    garment_id: str,
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Retrieves canonical garment representation matching PRD Section 2."""
    stmt = (
        select(Garment)
        .where(Garment.id == garment_id)
        .options(
            selectinload(Garment.source_image),
            selectinload(Garment.canonical_image),
        )
    )
    res = await session.execute(stmt)
    garment = res.scalars().first()
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )

    # Fetch embedding if available
    emb_stmt = (
        select(GarmentEmbedding)
        .where(GarmentEmbedding.garment_id == garment_id)
        .order_by(GarmentEmbedding.created_at.desc())
    )
    emb_res = await session.execute(emb_stmt)
    emb_record = emb_res.scalars().first()
    embedding_vec = emb_record.embedding if emb_record else None

    canonical_ref = garment.canonical_image.object_uri if garment.canonical_image else None

    # Co-ord set: other garments spawned from the SAME source photo (Stage 2 detects every
    # garment in a full-body shot and gives each its own Garment row sharing source_image_id —
    # see app/pipeline/stages/stage_02_crop.py). That shared id is exactly "these were worn
    # together" — no separate table needed, just query siblings by it.
    siblings_stmt = (
        select(Garment)
        .where(Garment.source_image_id == garment.source_image_id, Garment.id != garment.id)
        .options(selectinload(Garment.canonical_image))
    )
    siblings = (await session.execute(siblings_stmt)).scalars().all()
    coordinated = [
        CoordinatedGarment(
            garment_id=sib.id,
            detected_label=sib.detected_label,
            subcategory=sib.subcategory,
            category=sib.category,
            canonical_image_url=f"/api/v1/wardrobe/images/{sib.canonical_image_id}/bytes" if sib.canonical_image_id else None,
            status=sib.status,
        )
        for sib in siblings
    ]

    return CanonicalGarment(
        garment_id=garment.id,
        source_image_refs=[garment.source_image.object_uri] if garment.source_image else [],
        image_type=garment.image_type,
        garment_crop_refs=garment.garment_crop_refs,
        attributes=garment.attributes_json or None,
        canonical_image_ref=canonical_ref,
        image_embedding=embedding_vec,
        category=garment.category,
        compatibility_features=garment.compatibility_features,
        quality_status=garment.quality_status,
        provenance=garment.provenance,
        pipeline_version=garment.pipeline_version,
        coordinated_garments=coordinated,
    )


@router.get("/{garment_id}/pipeline", response_model=PipelineStatusResponse)
async def get_garment_pipeline_status(
    garment_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Inspectable audit log of all stage runs for a garment."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )

    stmt = (
        select(PipelineStageRun)
        .where(PipelineStageRun.garment_id == garment_id)
        .order_by(PipelineStageRun.started_at.asc(), PipelineStageRun.attempt.asc())
    )
    runs_res = await session.execute(stmt)
    stage_runs = runs_res.scalars().all()

    can_retry = garment.status in (
        GarmentState.FAILED.value,
        GarmentState.REVIEW_REQUIRED.value,
    )

    stage_reads = [
        PipelineStageRunRead(
            id=run.id,
            garment_id=run.garment_id,
            stage=run.stage,
            status=run.status,
            attempt=run.attempt,
            input_refs=run.input_refs,
            output_refs=run.output_refs,
            model=run.model,
            model_version=run.model_version,
            algorithm_version=run.algorithm_version,
            error=run.error,
            duration_ms=run.duration_ms,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )
        for run in stage_runs
    ]

    return PipelineStatusResponse(
        garment_id=garment.id,
        current_status=garment.status,
        quality_status=garment.quality_status,
        stages=stage_reads,
        can_retry=can_retry,
    )


@router.post("/{garment_id}/step")
async def execute_single_pipeline_step(
    garment_id: str,
    request: StepRequest = StepRequest(),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Executes a single pipeline stage step-by-step for live visual presentation.
    Returns intermediate visual artifacts (face bounding boxes, crops, canonical images, attributes).
    """
    stmt = (
        select(Garment)
        .where(Garment.id == garment_id)
        .options(
            selectinload(Garment.source_image),
            selectinload(Garment.canonical_image),
        )
    )
    res = await session.execute(stmt)
    garment = res.scalars().first()
    if not garment:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found")

    # If dynamic API keys were passed in the request, apply them for the duration of THIS
    # request only. settings is a process-wide singleton (api and worker are separate
    # processes/containers with independent settings instances — this can only ever leak
    # within one of them) shared by every concurrent request the api process handles, so a
    # save-mutate-restore alone isn't enough under concurrency: a second /step call (with or
    # without its own override) can start, read the FIRST call's temporarily-mutated key as if
    # it were real, and/or have its own "restore" overwrite the first call's restore out of
    # order. Confirmed live: this produced a run of real "401 Unauthorized" / "Missing
    # Authentication header" OpenRouter failures across an unrelated garment's Stage 1/2/4 —
    # not a rate limit — and, worse, Stage 3 silently reported SUCCEEDED with entirely
    # fabricated placeholder attributes instead of failing loud (see the matching fix in
    # app/providers/vlm/openrouter.py and app/providers/attributes/gemini.py). A per-process
    # lock around the whole mutate/execute/restore window closes the race — /step is an
    # interactive demo endpoint, not high-throughput traffic, so serializing it is a fine trade.
    async with _step_api_key_lock:
        original_openrouter_key = settings.OPENROUTER_API_KEY
        original_nvidia_key = settings.NVIDIA_API_KEY
        # A real OpenRouter key always starts with "sk-or-v1-" — reject anything else rather
        # than silently applying it as an override. Confirmed live: the demo UI's key field is
        # styled as a password input, which browser/password-manager autofill can silently
        # populate with an unrelated saved credential for this same domain (this app's own
        # /login page also has a password field) despite anti-autofill hints on the element;
        # every request from an affected browser then failed with a real 401 for the whole
        # duration. The frontend now filters this too — this is the server-side backstop for
        # any other caller of this endpoint.
        if request.openrouter_api_key and not request.openrouter_api_key.startswith("sk-or-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="openrouter_api_key doesn't look like a real OpenRouter key (expected to start with 'sk-or-') — refusing to apply it as an override.",
            )
        if request.openrouter_api_key:
            settings.OPENROUTER_API_KEY = request.openrouter_api_key
        if request.nvidia_api_key:
            settings.NVIDIA_API_KEY = request.nvidia_api_key

        try:
            return await _execute_step_with_keys_applied(garment_id, garment, request, session, storage)
        finally:
            settings.OPENROUTER_API_KEY = original_openrouter_key
            settings.NVIDIA_API_KEY = original_nvidia_key


async def _execute_step_with_keys_applied(
    garment_id: str,
    garment: Garment,
    request: StepRequest,
    session: AsyncSession,
    storage: StorageClient,
):
    # Determine stage to execute. Always route CLASSIFIED -> STAGE_02_CROP regardless of
    # image_type — Stage02Crop itself decides whether to actually crop (it only fast-skips
    # a CATALOG/CROP-classified image if no face is detected; a face still gets cropped).
    stage_mapping = {
        "RECEIVED": PipelineStage.STAGE_01_CLASSIFY,
        "CLASSIFIED": PipelineStage.STAGE_02_CROP,
        "CROPPED": PipelineStage.STAGE_03_ATTRIBUTES,
        "ATTRIBUTES_EXTRACTED": PipelineStage.STAGE_04_DIGITISE,
        "DIGITIZED": PipelineStage.STAGE_05_EMBED,
        "EMBEDDED": PipelineStage.STAGE_06_CATEGORY,
        "CATEGORY_BUNDLED": PipelineStage.STAGE_07_LAYERING,
        "LAYERING_ANALYZED": PipelineStage.STAGE_08_STRUCTURE,
        "STRUCTURE_ANALYZED": PipelineStage.STAGE_09_VISUAL,
    }

    if request.stage:
        stage_enum = PipelineStage(request.stage)
    else:
        stage_enum = stage_mapping.get(garment.status)
        if not stage_enum:
            if garment.status == "COMPLETED" and not request.force:
                return {
                    "stage": "COMPLETED",
                    "status": "COMPLETED",
                    "is_completed": True,
                    "garment_state": garment.status,
                    "quality_status": garment.quality_status,
                    "message": "Garment pipeline is already fully completed.",
                }
            stage_enum = PipelineStage.STAGE_01_CLASSIFY

    stage_class = PipelineOrchestrator.STAGE_MAP[stage_enum]
    stage_instance = stage_class()

    ctx = StageExecutionContext(
        session=session,
        garment=garment,
        storage=storage,
        pipeline_run_id=f"step_{garment_id[:8]}",
        attempt=1,
        force=request.force,
    )

    start_t = time.perf_counter()
    try:
        async with garment_execution_lock(garment_id, wait=False):
            result = await stage_instance.execute(ctx)
    except GarmentLockBusy:
        # Fail fast rather than hang — this is an interactive, user-watched call (the
        # step-by-step demo UI), unlike the background worker which can afford to wait.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Garment '{garment_id}' is already being processed by another pipeline run (e.g. the background worker) — try again shortly.",
        )
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    # Record stage run in DB
    stage_run = PipelineStageRun(
        garment_id=garment.id,
        stage=stage_enum.value,
        status=result.status,
        attempt=1,
        input_refs=result.input_refs,
        output_refs=result.output_refs,
        input_hash=result.input_hash,
        model=result.model,
        model_version=result.model_version,
        algorithm_version=result.algorithm_version,
        error=result.error,
        duration_ms=duration_ms,
    )
    session.add(stage_run)

    # State transition. Only ever ADVANCES garment.status, never regresses it — re-running an
    # earlier stage in isolation (e.g. force-refreshing Stage 3 attributes on a garment that
    # already completed the full pipeline) must not knock status backward from COMPLETED to
    # whatever that one stage's own next-state is, since stages 4-9's real output (canonical
    # image, embedding, compatibility scores) are all still intact and untouched. Confirmed live:
    # a bulk Stage-3-only re-run once demoted 1451 already-COMPLETED garments back to
    # ATTRIBUTES_EXTRACTED/REVIEW_REQUIRED this way, emptying the styling catalog.
    # A garment already sitting at REVIEW_REQUIRED specifically because Stage 3's cross-model
    # verifier disagreed with the extracted attributes must not get silently promoted back to
    # COMPLETED by force-running some OTHER, unrelated stage in isolation (e.g. Stage 6
    # category bundling, which just bundles whatever subcategory is already on the garment
    # without re-checking it) — confirmed live: this exact path resurrected already-rejected
    # attribute extractions (e.g. "cardigan" that the verifier said was clearly a poncho) back
    # to COMPLETED/APPROVED. Only a fresh, passing Stage 3 re-run may clear this flag.
    attribute_review_held = (
        garment.status == GarmentState.REVIEW_REQUIRED.value
        and garment.quality_status == "REVIEW_REQUIRED"
        and stage_enum != PipelineStage.STAGE_03_ATTRIBUTES
    )

    if result.status == "SUCCEEDED":
        if not attribute_review_held:
            next_state = STAGE_TO_GARMENT_STATE.get(stage_enum)
            if next_state and _garment_state_order(next_state.value) > _garment_state_order(garment.status):
                garment.status = next_state.value
            if result.quality_status:
                garment.quality_status = result.quality_status
    elif result.status == "REVIEW_REQUIRED":
        if stage_enum == PipelineStage.STAGE_03_ATTRIBUTES:
            # The only genuine halt condition — see app/pipeline/orchestrator.py's matching
            # comment. Every other stage's REVIEW_REQUIRED (e.g. a low-confidence heuristic
            # fallback) still advances below, same as SUCCEEDED.
            garment.status = GarmentState.REVIEW_REQUIRED.value
            garment.quality_status = "REVIEW_REQUIRED"
        elif not attribute_review_held:
            next_state = STAGE_TO_GARMENT_STATE.get(stage_enum)
            if next_state and _garment_state_order(next_state.value) > _garment_state_order(garment.status):
                garment.status = next_state.value
    else:
        garment.status = GarmentState.FAILED.value
        garment.quality_status = "REJECTED"

    await session.commit()
    await session.refresh(garment)

    spawned_garment_ids = result.output_refs.get("spawned_garment_ids", [])
    if spawned_garment_ids:
        for sibling_id in spawned_garment_ids:
            await enqueue_garment_pipeline(sibling_id, resume_stage=PipelineStage.STAGE_03_ATTRIBUTES.value)

    # Build direct media URLs for visual presentation
    raw_url = f"/api/v1/wardrobe/images/{garment.source_image.id}/bytes" if garment.source_image else None
    crop_url = None
    if garment.garment_crop_refs:
        clean_crop = garment.garment_crop_refs[0].replace(f"object://{settings.S3_BUCKET_NAME}/", "").lstrip("/")
        crop_url = f"/api/v1/wardrobe/images/media/{clean_crop}"

    overlay_url = None
    if "annotated_overlay_uri" in result.output_refs and result.output_refs["annotated_overlay_uri"]:
        clean_ann = result.output_refs["annotated_overlay_uri"].replace(f"object://{settings.S3_BUCKET_NAME}/", "").lstrip("/")
        overlay_url = f"/api/v1/wardrobe/images/media/{clean_ann}"

    # Every region Stage 2 detected (not just this garment's own kept crop) — lets the demo
    # UI show all garments found in the photo, not just the primary's single thumbnail.
    all_crop_urls = [
        f"/api/v1/wardrobe/images/media/{ref.replace(f'object://{settings.S3_BUCKET_NAME}/', '').lstrip('/')}"
        for ref in result.output_refs.get("garment_crop_refs", [])
    ]

    canonical_url = None
    if garment.canonical_image_id:
        canonical_url = f"/api/v1/wardrobe/images/{garment.canonical_image_id}/bytes"

    return {
        "stage": stage_enum.value,
        "status": result.status,
        "duration_ms": round(duration_ms, 2),
        "model": result.model,
        "model_version": result.model_version,
        "algorithm_version": result.algorithm_version,
        "garment_state": garment.status,
        "quality_status": garment.quality_status,
        "is_completed": garment.status == "COMPLETED",
        "output_data": result.output_refs,
        "error": result.error,
        "spawned_garment_ids": spawned_garment_ids,
        "visual_artifacts": {
            "raw_image_url": raw_url,
            "annotated_overlay_url": overlay_url or raw_url,
            "crop_image_url": crop_url or raw_url,
            "all_crop_urls": all_crop_urls,
            "canonical_image_url": canonical_url or crop_url or raw_url,
        },
    }


@router.post("/{garment_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_garment_pipeline(
    garment_id: str,
    request: RetryRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Retries a failed or reviewable stage in the garment pipeline."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )
    if garment.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This garment belongs to another account")

    # Enqueue pipeline run
    await enqueue_garment_pipeline(garment.id, force=request.force, resume_stage=request.stage)
    return {"status": "ENQUEUED", "garment_id": garment.id, "stage": request.stage}


@router.post("/{garment_id}/review", status_code=status.HTTP_200_OK)
async def review_garment_pipeline(
    garment_id: str,
    request: ReviewRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Submits operator review or overrides for a garment flagged for human review."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )
    if garment.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This garment belongs to another account")

    if request.decision == ReviewDecision.APPROVE:
        garment.quality_status = "APPROVED"
        if garment.status == GarmentState.REVIEW_REQUIRED.value:
            # Resume remaining stages
            await enqueue_garment_pipeline(garment.id, force=False)
        garment.provenance = {
            **garment.provenance,
            "manual_review": {"decision": "APPROVE", "notes": request.notes},
        }

    elif request.decision == ReviewDecision.REJECT:
        garment.quality_status = "REJECTED"
        garment.status = GarmentState.FAILED.value
        garment.provenance = {
            **garment.provenance,
            "manual_review": {"decision": "REJECT", "notes": request.notes},
        }

    elif request.decision == ReviewDecision.OVERRIDE:
        if request.attribute_overrides:
            # Strictly validate overrides
            validated = validate_extracted_attributes(request.attribute_overrides)
            garment.attributes_json = validated.model_dump(mode="json")
            garment.subcategory = validated.subcategory

        garment.quality_status = "APPROVED"
        garment.provenance = {
            **garment.provenance,
            "manual_review": {
                "decision": "OVERRIDE",
                "notes": request.notes,
                "overrides": request.attribute_overrides,
            },
        }
        # Resume pipeline from Category Bundling stage
        await enqueue_garment_pipeline(garment.id, force=True, resume_stage=PipelineStage.STAGE_06_CATEGORY.value)

    await session.commit()
    await session.refresh(garment)

    return {
        "garment_id": garment.id,
        "status": garment.status,
        "quality_status": garment.quality_status,
        "decision": request.decision.value,
    }
