"""Standalone worker runner entrypoint."""

import asyncio
from app.database import init_db
from app.observability import logger
from app.worker.queue import _worker_loop


async def run_worker():
    logger.info("Initializing database for worker...")
    await init_db()
    logger.info("Starting background worker loop...")
    await _worker_loop()


if __name__ == "__main__":
    asyncio.run(run_worker())
