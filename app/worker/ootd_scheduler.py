"""Daily Outfit-of-the-Day generation loop.

Hand-rolled rather than a scheduling library (APScheduler etc.) — this worker process already
runs its own indefinite asyncio loops (see app/worker/queue.py), and a single "sleep until the
next run time, then run" loop is simpler than adding a new dependency for one daily job.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ootd import OOTDSubscription
from app.observability import logger
from app.storage import get_storage_client
from app.styling.ootd import get_or_generate_ootd


def _seconds_until_next_run() -> float:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=settings.OOTD_DAILY_GENERATION_HOUR_UTC, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def run_daily_ootd_generation() -> None:
    """Pre-generates today's pick for every enabled OOTDSubscription — each member gets their
    own session/transaction so one failure (e.g. a bad location, a transient API error) can't
    abort the batch; that member's pick just stays ungenerated until the next daily run or an
    on-demand request."""
    storage = get_storage_client()
    async with AsyncSessionLocal() as session:
        subs = (await session.execute(select(OOTDSubscription).where(OOTDSubscription.enabled.is_(True)))).scalars().all()
    logger.info(f"Daily OOTD run starting for {len(subs)} enabled subscription(s).")

    succeeded = 0
    for sub in subs:
        try:
            async with AsyncSessionLocal() as session:
                await get_or_generate_ootd(
                    session, storage, sub.tenant_id, sub.member_id,
                    location=sub.location, persona=sub.persona, force=False,
                    generation_source="scheduled",
                )
            succeeded += 1
        except Exception as e:
            logger.error(f"Daily OOTD generation failed for tenant={sub.tenant_id} member={sub.member_id}: {e}")

    logger.info(f"Daily OOTD run finished: {succeeded}/{len(subs)} succeeded.")


async def ootd_scheduler_loop() -> None:
    while True:
        sleep_seconds = _seconds_until_next_run()
        logger.info(f"OOTD scheduler sleeping {sleep_seconds:.0f}s until next daily run.")
        await asyncio.sleep(sleep_seconds)
        try:
            await run_daily_ootd_generation()
        except Exception as e:
            logger.error(f"Daily OOTD generation run failed: {e}")
