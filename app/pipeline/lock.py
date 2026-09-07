"""Per-garment pipeline execution lock.

Two independent triggers can race on the same garment: the background worker picking up its
queued job (app/worker/queue.py), and the interactive step-by-step demo UI calling
POST /garments/{id}/step directly (app/api/v1/garments.py) — these run in separate containers
(api vs worker), so an in-process lock can't see across them; Redis (already the real
cross-container job transport here — see app/worker/queue.py) is the natural shared lock.

Confirmed live: exactly this race produced two STAGE_01_CLASSIFY runs for the same garment
half a second apart, one of which hit a transient "all vision models failed" result that then
overwrote the other's good one, leaving the garment stuck at REVIEW_REQUIRED even though the
classifier itself was working fine moments before and after.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis

from app.config import settings
from app.observability import logger

_LOCK_KEY_PREFIX = "wardrobe_pipeline_lock:"
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class GarmentLockBusy(Exception):
    """Raised when a lock is requested with wait=False and another execution already holds it."""

    def __init__(self, garment_id: str):
        super().__init__(f"Garment '{garment_id}' is already being processed by another pipeline run.")
        self.garment_id = garment_id


async def _try_acquire(client: aioredis.Redis, garment_id: str, token: str, ttl_seconds: int) -> bool:
    return bool(await client.set(f"{_LOCK_KEY_PREFIX}{garment_id}", token, nx=True, ex=ttl_seconds))


async def _release(client: aioredis.Redis, garment_id: str, token: str) -> None:
    try:
        await client.eval(_RELEASE_SCRIPT, 1, f"{_LOCK_KEY_PREFIX}{garment_id}", token)
    except Exception as e:
        # Losing the lock is not fatal — the TTL guarantees it clears itself eventually; this
        # is just a best-effort early release so a healthy next run doesn't wait out the TTL.
        logger.warning(f"Failed to release pipeline lock for garment {garment_id}: {e}")


@asynccontextmanager
async def garment_execution_lock(
    garment_id: str,
    wait: bool = True,
    ttl_seconds: int = 180,
    max_wait_seconds: int = 60,
    poll_interval_seconds: float = 0.5,
):
    """
    Ensures only one pipeline stage execution runs for a given garment at a time.

    wait=True (background worker): retries until the lock frees up or max_wait_seconds elapses,
    then proceeds anyway rather than dropping a legitimate queued job — logs a warning so a
    stuck lock (e.g. a crashed holder, though the TTL already bounds that) is visible.

    wait=False (interactive API call): raises GarmentLockBusy immediately if another execution
    already holds the lock — an interactive request should fail fast with a clear "try again
    shortly" rather than hang, since the user is waiting on it live.
    """
    import asyncio

    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=10.0)
    token = uuid.uuid4().hex
    acquired = False
    try:
        acquired = await _try_acquire(client, garment_id, token, ttl_seconds)
        if not acquired and wait:
            waited = 0.0
            while not acquired and waited < max_wait_seconds:
                await asyncio.sleep(poll_interval_seconds)
                waited += poll_interval_seconds
                acquired = await _try_acquire(client, garment_id, token, ttl_seconds)
            if not acquired:
                logger.warning(
                    f"Pipeline lock for garment {garment_id} still held after {max_wait_seconds}s "
                    f"wait — proceeding anyway rather than dropping the queued job."
                )
        elif not acquired:
            raise GarmentLockBusy(garment_id)

        yield
    finally:
        if acquired:
            await _release(client, garment_id, token)
        await client.aclose()
