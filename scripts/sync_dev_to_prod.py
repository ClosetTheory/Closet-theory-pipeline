"""One-off: push the repaired dev data to production.

The canonical-image repair (605 regenerated images), the embedding backfill (1176 vectors that
were hash-seeded noise) and the attribute cleanups all ran against dev. Production runs the same
code over the same garment ids but still holds the old data, so its duplicate detection and
similarity retrieval are still comparing random numbers.

Regenerating on production would mean paying for ~605 image generations a second time, so this
copies the finished result instead, via the temporary /garments/_sync_from_dev endpoint.

Sends garments in small batches because each canonical image is ~1.5MB and base64 adds a third
again. Safe to re-run: every write is an overwrite of the same garment id, never an insert of a
new garment.

    python -m scripts.sync_dev_to_prod --dry-run
    python -m scripts.sync_dev_to_prod --what embeddings      # cheapest, fixes prod dedup first
    python -m scripts.sync_dev_to_prod --what all
"""

import argparse
import asyncio
import base64
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.storage import get_storage_client

PROD_URL = "https://visualiser-dsfja238un9.closettheory.co"


async def _load(what: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    Garment.id,
                    Garment.attributes_json,
                    ImageAsset.object_uri,
                    GarmentEmbedding.embedding,
                )
                .outerjoin(ImageAsset, ImageAsset.id == Garment.canonical_image_id)
                .outerjoin(GarmentEmbedding, GarmentEmbedding.garment_id == Garment.id)
                .where(Garment.status == "COMPLETED")
                .order_by(Garment.id)
            )
        ).all()

    items = [
        {
            "garment_id": r[0],
            "attributes": r[1],
            "canonical_uri": r[2],
            "embedding": list(r[3]) if r[3] is not None else None,
        }
        for r in rows
    ]
    if limit:
        items = items[:limit]
    return items


async def _authenticate(client: httpx.AsyncClient, email: str, password: str) -> str:
    resp = await client.post(
        f"{PROD_URL}/api/v1/auth/login", json={"email": email, "password": password}, timeout=30.0
    )
    resp.raise_for_status()
    return resp.json()["token"]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["all", "embeddings", "images", "attributes"], default="all")
    ap.add_argument("--batch", type=int, default=4, help="garments per request (images are large)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--email", default="demo@closettheory.local")
    ap.add_argument("--password", default="demo1234")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = await _load(args.what, args.limit)
    with_image = sum(1 for i in items if i["canonical_uri"])
    with_emb = sum(1 for i in items if i["embedding"])
    print(f"dev garments: {len(items)}  (with canonical image: {with_image}, with embedding: {with_emb})")
    print(f"syncing: {args.what}")

    if args.dry_run:
        print("[dry run] nothing sent.")
        return

    storage = get_storage_client()
    send_images = args.what in ("all", "images")
    send_emb = args.what in ("all", "embeddings")
    send_attrs = args.what in ("all", "attributes")
    # Embeddings/attributes are small, so a batch of 4 (sized for ~1.5MB images) would mean
    # hundreds of pointless round-trips.
    batch_size = args.batch if send_images else 60

    async with httpx.AsyncClient(timeout=300.0) as client:
        token = await _authenticate(client, args.email, args.password)
        headers = {"Authorization": f"Bearer {token}"}

        applied = failed = 0
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            payload = []
            for item in chunk:
                entry: Dict[str, Any] = {"garment_id": item["garment_id"]}
                if send_emb and item["embedding"]:
                    entry["embedding"] = item["embedding"]
                if send_attrs and item["attributes"]:
                    entry["attributes"] = item["attributes"]
                if send_images and item["canonical_uri"]:
                    try:
                        raw = await storage.get_object(item["canonical_uri"])
                        entry["canonical_image_b64"] = base64.b64encode(raw).decode("utf-8")
                    except Exception as e:
                        print(f"  ! {item['garment_id']}: could not read image ({type(e).__name__})")
                if len(entry) > 1:
                    payload.append(entry)

            if not payload:
                continue
            try:
                resp = await client.post(
                    f"{PROD_URL}/api/v1/wardrobe/garments/_sync_from_dev", json=payload, headers=headers
                )
                resp.raise_for_status()
                applied += resp.json().get("applied", 0)
            except Exception as e:
                failed += len(payload)
                print(f"  ! batch at {start} failed: {type(e).__name__}: {str(e)[:120]}")

            done = min(start + batch_size, len(items))
            if done % (batch_size * 10) == 0 or done >= len(items):
                print(f"  {done}/{len(items)} sent, {applied} applied, {failed} failed")

    print(f"\nDone. {applied} garment(s) updated on production, {failed} failed.")


if __name__ == "__main__":
    asyncio.run(main())
