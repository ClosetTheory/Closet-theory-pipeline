"""One-off backfill: populate `gender` on every already-ingested garment that doesn't have it.

Stage 3's attribute schema gained a `gender` field (women/men/unisex) after the styling
pipeline was found recommending gender-mismatched outfits — garments ingested before that
change have no gender data. Re-running the FULL pipeline on every garment would needlessly
regenerate canonical images (Stage 4) etc.; this only re-runs Stage 3 in isolation via the
existing single-stage `/step` endpoint (unauthenticated, demo-purpose endpoint — see
app/api/v1/garments.py::execute_single_pipeline_step), exactly the mechanism already used
manually throughout this project's development.

Safe to re-run: only processes garments where `gender IS NULL`, so an interrupted run picks
up where it left off. Usage:

    python -m scripts.backfill_gender [--base-url http://localhost:8000] [--concurrency 5]
"""

import argparse
import asyncio
import sys

import httpx
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.garment import Garment


async def _fetch_target_ids() -> list[str]:
    async with AsyncSessionLocal() as session:
        stmt = select(Garment.id).where(
            Garment.status == "COMPLETED",
            Garment.gender.is_(None),
        )
        result = await session.execute(stmt)
        return [row[0] for row in result.all()]


async def _backfill_one(client: httpx.AsyncClient, base_url: str, garment_id: str, sem: asyncio.Semaphore) -> tuple[str, bool, str]:
    async with sem:
        try:
            resp = await client.post(
                f"{base_url}/api/v1/wardrobe/garments/{garment_id}/step",
                json={"stage": "STAGE_03_ATTRIBUTES", "force": True},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            return garment_id, status == "SUCCEEDED", status
        except Exception as e:
            return garment_id, False, str(e)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    ids = await _fetch_target_ids()
    total = len(ids)
    if total == 0:
        print("Nothing to backfill — every COMPLETED garment already has gender data.")
        return

    print(f"Backfilling gender on {total} garment(s), concurrency={args.concurrency}...")
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    failed: list[tuple[str, str]] = []

    async with httpx.AsyncClient() as client:
        tasks = [_backfill_one(client, args.base_url, gid, sem) for gid in ids]
        for coro in asyncio.as_completed(tasks):
            garment_id, ok, status = await coro
            done += 1
            if not ok:
                failed.append((garment_id, status))
            if done % 20 == 0 or done == total:
                print(f"  {done}/{total} processed, {len(failed)} failed so far")

    print(f"\nDone. {total - len(failed)}/{total} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed garment ids (re-run this script to retry — it only touches gender IS NULL):")
        for gid, status in failed:
            print(f"  {gid}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
