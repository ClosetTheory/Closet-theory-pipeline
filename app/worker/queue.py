"""Background job queue abstraction for async pipeline execution."""

import asyncio
from typing import Optional
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.garment import Garment
from app.observability import logger
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.state_machine import PipelineStage
from app.storage import get_storage_client

# In-memory async queue for lightweight standalone/local/testing execution
_in_memory_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None


def get_queue() -> asyncio.Queue:
    global _in_memory_queue
    if _in_memory_queue is None:
        _in_memory_queue = asyncio.Queue()
    return _in_memory_queue


async def process_job(garment_id: str, force: bool = False, resume_stage: Optional[str] = None):
    """Executes a single garment pipeline run asynchronously."""
    async with AsyncSessionLocal() as session:
        storage = get_storage_client()
        garment = await session.get(Garment, garment_id)
        if not garment:
            logger.error(f"Worker could not find garment {garment_id}")
            return

        stage_enum = PipelineStage(resume_stage) if resume_stage else None
        orchestrator = PipelineOrchestrator(session=session, storage=storage)
        await orchestrator.run(garment, force=force, resume_stage=stage_enum)


async def _worker_loop():
    queue = get_queue()
    while True:
        try:
            job = await queue.get()
            garment_id = job["garment_id"]
            force = job.get("force", False)
            resume_stage = job.get("resume_stage")
            await process_job(garment_id, force, resume_stage)
            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in background worker loop: {e}")


def start_in_memory_worker():
    """Starts background worker task if in-memory queue is enabled."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())


async def enqueue_garment_pipeline(
    garment_id: str,
    force: bool = False,
    resume_stage: Optional[str] = None,
):
    """Enqueues garment for asynchronous ingestion processing."""
    if settings.USE_IN_MEMORY_QUEUE:
        start_in_memory_worker()
        queue = get_queue()
        await queue.put({
            "garment_id": garment_id,
            "force": force,
            "resume_stage": resume_stage,
        })
    else:
        # If external Redis queue is configured:
        # In a full Redis/Celery deployment, this pushes a job payload to redis
        start_in_memory_worker()
        queue = get_queue()
        await queue.put({
            "garment_id": garment_id,
            "force": force,
            "resume_stage": resume_stage,
        })
