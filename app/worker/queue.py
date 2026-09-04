"""Background job queue abstraction for async pipeline execution."""

import asyncio
import json
from typing import Optional
import redis.asyncio as aioredis
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
REDIS_QUEUE_KEY = "wardrobe_pipeline_jobs"


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

        # Stage 2 may have detected multiple garments in one photo and spawned sibling
        # Garment rows (see Stage02Crop / PipelineOrchestrator). Enqueue them here — the one
        # real async-worker context — rather than from inside the orchestrator itself, which
        # stays a pure pipeline runner safe to call directly (e.g. in tests).
        for sibling_id in orchestrator.spawned_garment_ids:
            await enqueue_garment_pipeline(sibling_id, resume_stage=PipelineStage.STAGE_03_ATTRIBUTES.value)


async def _in_memory_worker_loop():
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
            logger.error(f"Error in in-memory worker loop: {e}")


async def _redis_worker_loop():
    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=15.0)
    logger.info(f"Connected to Redis at {settings.REDIS_URL}, listening on '{REDIS_QUEUE_KEY}'...")
    while True:
        try:
            item = await r.brpop(REDIS_QUEUE_KEY, timeout=5)
            if item is None:
                continue
            _, payload_str = item
            job = json.loads(payload_str)
            garment_id = job["garment_id"]
            force = job.get("force", False)
            resume_stage = job.get("resume_stage")
            await process_job(garment_id, force, resume_stage)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in Redis worker loop: {e}")
            await asyncio.sleep(2)


def start_in_memory_worker():
    """Starts background worker task if in-memory queue is enabled."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_in_memory_worker_loop())


async def enqueue_garment_pipeline(
    garment_id: str,
    force: bool = False,
    resume_stage: Optional[str] = None,
):
    """Enqueues garment for asynchronous ingestion processing."""
    payload = {
        "garment_id": garment_id,
        "force": force,
        "resume_stage": resume_stage,
    }

    if settings.USE_IN_MEMORY_QUEUE:
        start_in_memory_worker()
        queue = get_queue()
        await queue.put(payload)
    else:
        try:
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await r.lpush(REDIS_QUEUE_KEY, json.dumps(payload))
            await r.aclose()
        except Exception as e:
            logger.error(f"Failed to enqueue to Redis, falling back to in-memory: {e}")
            start_in_memory_worker()
            queue = get_queue()
            await queue.put(payload)
