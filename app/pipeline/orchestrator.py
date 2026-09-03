"""Pipeline Orchestrator coordinating all 9 stages with state machine & idempotency."""

from datetime import datetime, timezone
import time
import uuid
from typing import Dict, List, Optional, Type
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.garment import Garment
from app.models.pipeline_stage import PipelineStageRun
from app.observability import log_stage_event, metrics
from app.pipeline.idempotency import compute_idempotency_key
from app.pipeline.stages import (
    BaseStage,
    Stage01Classify,
    Stage02Crop,
    Stage03Attributes,
    Stage04Digitise,
    Stage05Embed,
    Stage06Category,
    Stage07Layering,
    Stage08Structure,
    Stage09Visual,
    StageExecutionContext,
)
from app.pipeline.state_machine import (
    PIPELINE_STAGES_ORDER,
    STAGE_TO_GARMENT_STATE,
    GarmentState,
    PipelineStage,
)
from app.storage.base import StorageClient


class PipelineOrchestrator:
    """Coordinates execution of sequential garment ingestion stages."""

    STAGE_MAP: Dict[PipelineStage, Type[BaseStage]] = {
        PipelineStage.STAGE_01_CLASSIFY: Stage01Classify,
        PipelineStage.STAGE_02_CROP: Stage02Crop,
        PipelineStage.STAGE_03_ATTRIBUTES: Stage03Attributes,
        PipelineStage.STAGE_04_DIGITISE: Stage04Digitise,
        PipelineStage.STAGE_05_EMBED: Stage05Embed,
        PipelineStage.STAGE_06_CATEGORY: Stage06Category,
        PipelineStage.STAGE_07_LAYERING: Stage07Layering,
        PipelineStage.STAGE_08_STRUCTURE: Stage08Structure,
        PipelineStage.STAGE_09_VISUAL: Stage09Visual,
    }

    def __init__(self, session: AsyncSession, storage: StorageClient):
        self.session = session
        self.storage = storage

    async def run(
        self,
        garment: Garment,
        force: bool = False,
        resume_stage: Optional[PipelineStage] = None,
    ) -> Garment:
        """Runs the sequential pipeline for a garment."""
        pipeline_run_id = f"run_{uuid.uuid4().hex[:12]}"
        stages_to_run = PIPELINE_STAGES_ORDER

        # If resuming from a specific stage, slice pipeline
        if resume_stage:
            try:
                idx = PIPELINE_STAGES_ORDER.index(resume_stage)
                stages_to_run = PIPELINE_STAGES_ORDER[idx:]
            except ValueError:
                pass

        for stage_enum in stages_to_run:
            stage_class = self.STAGE_MAP[stage_enum]
            stage_instance = stage_class()

            # Check for existing successful stage execution (Idempotency check)
            stmt = (
                select(PipelineStageRun)
                .where(
                    PipelineStageRun.garment_id == garment.id,
                    PipelineStageRun.stage == stage_enum.value,
                    PipelineStageRun.status == "SUCCEEDED",
                )
                .order_by(PipelineStageRun.attempt.desc())
            )
            res = await self.session.execute(stmt)
            existing_run = res.scalars().first()

            if existing_run and not force:
                # PRD Section 20: Computation exists and is valid -> reuse it
                log_stage_event(
                    pipeline_run_id=pipeline_run_id,
                    garment_id=garment.id,
                    stage=stage_enum.value,
                    status="REUSED",
                    model=existing_run.model,
                    model_version=existing_run.model_version,
                )
                # Advance garment state if needed
                next_state = STAGE_TO_GARMENT_STATE.get(stage_enum)
                if next_state and garment.status != GarmentState.COMPLETED.value:
                    garment.status = next_state.value
                continue

            # Compute attempt number
            attempt_stmt = (
                select(PipelineStageRun)
                .where(
                    PipelineStageRun.garment_id == garment.id,
                    PipelineStageRun.stage == stage_enum.value,
                )
                .order_by(PipelineStageRun.attempt.desc())
            )
            att_res = await self.session.execute(attempt_stmt)
            last_run = att_res.scalars().first()
            attempt = (last_run.attempt + 1) if last_run else 1

            # Initialize Stage Run record
            stage_run = PipelineStageRun(
                garment_id=garment.id,
                stage=stage_enum.value,
                status="RUNNING",
                attempt=attempt,
                input_refs={},
                output_refs={},
            )
            self.session.add(stage_run)
            await self.session.flush()

            ctx = StageExecutionContext(
                session=self.session,
                garment=garment,
                storage=self.storage,
                pipeline_run_id=pipeline_run_id,
                attempt=attempt,
                force=force,
            )

            start_t = time.perf_counter()
            try:
                result = await stage_instance.execute(ctx)
                duration_ms = (time.perf_counter() - start_t) * 1000.0

                # Update stage run record
                stage_run.status = result.status
                stage_run.input_refs = result.input_refs
                stage_run.output_refs = result.output_refs
                stage_run.input_hash = result.input_hash
                stage_run.model = result.model
                stage_run.model_version = result.model_version
                stage_run.algorithm_version = result.algorithm_version
                stage_run.error = result.error
                stage_run.duration_ms = duration_ms
                stage_run.completed_at = datetime.now(timezone.utc)

                # Observability
                log_stage_event(
                    pipeline_run_id=pipeline_run_id,
                    garment_id=garment.id,
                    stage=stage_enum.value,
                    status=result.status,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    model=result.model,
                    model_version=result.model_version,
                    input_hash=result.input_hash,
                    error=result.error,
                )
                metrics.record_stage_execution(stage_enum.value, result.status, duration_ms, attempt)

                # State machine transition
                if result.status == "SUCCEEDED":
                    next_state = STAGE_TO_GARMENT_STATE.get(stage_enum)
                    if next_state:
                        garment.status = next_state.value
                    if result.quality_status:
                        garment.quality_status = result.quality_status
                elif result.status == "REVIEW_REQUIRED":
                    garment.status = GarmentState.REVIEW_REQUIRED.value
                    garment.quality_status = "REVIEW_REQUIRED"
                    break
                else:  # FAILED or RETRYABLE
                    garment.status = GarmentState.FAILED.value
                    garment.quality_status = "REJECTED"
                    break

            except Exception as e:
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                stage_run.status = "FAILED"
                stage_run.error = str(e)
                stage_run.duration_ms = duration_ms

                garment.status = GarmentState.FAILED.value
                garment.quality_status = "REJECTED"

                log_stage_event(
                    pipeline_run_id=pipeline_run_id,
                    garment_id=garment.id,
                    stage=stage_enum.value,
                    status="FAILED",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                metrics.record_stage_execution(stage_enum.value, "FAILED", duration_ms, attempt)
                break

        await self.session.commit()
        return garment
