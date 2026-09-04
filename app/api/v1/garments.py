"""Garment and Pipeline execution API endpoints."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user, get_db_session, get_storage
from app.config import settings
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.pipeline_stage import PipelineStageRun
from app.models.user import User
from app.observability import log_stage_event, metrics
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.stages.base import StageExecutionContext
from app.pipeline.state_machine import (
    PIPELINE_STAGES_ORDER,
    STAGE_TO_GARMENT_STATE,
    GarmentState,
    PipelineStage,
)
from app.schemas.attributes import validate_extracted_attributes
from app.schemas.garment import CanonicalGarment, GarmentCreateRequest
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

    # Enqueue pipeline execution asynchronously
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

    # If dynamic API key passed in request, set it
    if request.openrouter_api_key:
        settings.OPENROUTER_API_KEY = request.openrouter_api_key
    if request.nvidia_api_key:
        settings.NVIDIA_API_KEY = request.nvidia_api_key

    # Determine stage to execute
    # If image is CATALOG or CROP, skip Stage 2 (person/face detection and cropping)
    next_crop_or_attr = (
        PipelineStage.STAGE_03_ATTRIBUTES
        if garment.image_type in ("CATALOG", "CROP")
        else PipelineStage.STAGE_02_CROP
    )

    stage_mapping = {
        "RECEIVED": PipelineStage.STAGE_01_CLASSIFY,
        "CLASSIFIED": next_crop_or_attr,
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
    result = await stage_instance.execute(ctx)
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

    # State transition
    if result.status == "SUCCEEDED":
        next_state = STAGE_TO_GARMENT_STATE.get(stage_enum)
        if next_state:
            garment.status = next_state.value
        if result.quality_status:
            garment.quality_status = result.quality_status
    elif result.status == "REVIEW_REQUIRED":
        garment.status = GarmentState.REVIEW_REQUIRED.value
        garment.quality_status = "REVIEW_REQUIRED"
    else:
        garment.status = GarmentState.FAILED.value
        garment.quality_status = "REJECTED"

    await session.commit()
    await session.refresh(garment)

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
        "visual_artifacts": {
            "raw_image_url": raw_url,
            "annotated_overlay_url": overlay_url or raw_url,
            "crop_image_url": crop_url or raw_url,
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
