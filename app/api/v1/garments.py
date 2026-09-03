"""Garment and Pipeline execution API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_db_session, get_storage
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.pipeline_stage import PipelineStageRun
from app.pipeline.state_machine import GarmentState, PipelineStage
from app.schemas.attributes import validate_extracted_attributes
from app.schemas.garment import CanonicalGarment, GarmentCreateRequest
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


@router.post("", response_model=CanonicalGarment, status_code=status.HTTP_202_ACCEPTED)
async def create_garment(
    request: GarmentCreateRequest,
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

    garment = Garment(
        tenant_id=request.tenant_id,
        member_id=request.member_id,
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


@router.post("/{garment_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_garment_pipeline(
    garment_id: str,
    request: RetryRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Retries a failed or reviewable stage in the garment pipeline."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )

    # Enqueue pipeline run
    await enqueue_garment_pipeline(garment.id, force=request.force, resume_stage=request.stage)
    return {"status": "ENQUEUED", "garment_id": garment.id, "stage": request.stage}


@router.post("/{garment_id}/review", status_code=status.HTTP_200_OK)
async def review_garment_pipeline(
    garment_id: str,
    request: ReviewRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Submits operator review or overrides for a garment flagged for human review."""
    garment = await session.get(Garment, garment_id)
    if not garment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment '{garment_id}' not found",
        )

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
