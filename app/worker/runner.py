"""Standalone worker runner entrypoint."""

import asyncio
from app.config import settings
from app.database import init_db
from app.observability import logger
from app.worker.ootd_scheduler import ootd_scheduler_loop
from app.worker.queue import _in_memory_worker_loop, _redis_worker_loop


async def run_worker():
    logger.info("Initializing database for worker...")
    await init_db()
    if settings.USE_IN_MEMORY_QUEUE:
        logger.info("Starting in-memory worker loop...")
        await asyncio.gather(_in_memory_worker_loop(), ootd_scheduler_loop())
    else:
        concurrency = max(1, settings.WORKER_CONCURRENCY)
        logger.info(f"Starting {concurrency} concurrent Redis worker loops...")
        await asyncio.gather(*(_redis_worker_loop() for _ in range(concurrency)), ootd_scheduler_loop())


if __name__ == "__main__":
    asyncio.run(run_worker())
