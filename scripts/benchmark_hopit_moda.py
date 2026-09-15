"""Exercise and benchmark the Hopit MODA API against this wardrobe's real garments.

Written from the integration scope doc (11 Sep 2026). That document specifies request and
response shapes but no base URL and no auth scheme, so both are configuration here:

    export HOPIT_BASE_URL=https://...        # required
    export HOPIT_API_KEY=...                 # required
    export HOPIT_AUTH_HEADER=Authorization   # optional, default Authorization: Bearer <key>

Their ingestion fetches images by URL, so garments are sent with their production media URL
(/api/v1/wardrobe/images/media/... is served unauthenticated). Nothing here writes to our
database — it only reads garments and calls their API.

Covers the three capabilities marked Available in the scope doc. Styling (/v1/outfits:rank) is
marked "in development" there, so it is attempted only with --include-styling and a failure is
reported rather than treated as a defect.

    python -m scripts.benchmark_hopit_moda --limit 50 --dry-run   # show what would be sent
    python -m scripts.benchmark_hopit_moda --limit 50
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset

PROD_MEDIA_BASE = os.getenv(
    "PROD_MEDIA_BASE", "https://visualiser-dsfja238un9.closettheory.co/api/v1/wardrobe/images/media"
)

# Real wardrobe-shaped queries, not synthetic ones — the scope doc explicitly asks for a sample
# of real queries to measure intent-style edits on wardrobe-sized pools.
TEXT_QUERIES = [
    "black oversized shirt for a casual dinner",
    "something warm for a cold rainy morning",
    "light floral dress for a summer brunch",
    "smart navy blazer for the office",
    "comfortable black leggings",
    "traditional outfit for a wedding",
]
COMPOSED_INSTRUCTIONS = [
    "but sleeveless, in navy",
    "make this more suitable for a date night",
    "a warmer version of this",
]


def _media_url(object_uri: str) -> str:
    return f"{PROD_MEDIA_BASE}/{object_uri.replace('object://wardrobe-assets/', '')}"


async def _load(limit: int) -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    Garment.id,
                    Garment.attributes_json,
                    Garment.member_id,
                    ImageAsset.object_uri,
                    GarmentEmbedding.embedding,
                )
                .join(ImageAsset, ImageAsset.id == Garment.canonical_image_id)
                .outerjoin(GarmentEmbedding, GarmentEmbedding.garment_id == Garment.id)
                .where(Garment.status == "COMPLETED")
                .order_by(Garment.id)
                .limit(limit)
            )
        ).all()
    return [
        {
            "garment_id": r[0],
            "attributes": r[1] or {},
            "owner_ref": r[2],
            "image_url": _media_url(r[3]),
            "our_embedding": list(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]


class Hopit:
    def __init__(self, base_url: str, api_key: str, header: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            header: f"Bearer {api_key}" if header.lower() == "authorization" else api_key,
            "Content-Type": "application/json",
        }
        self.timings: Dict[str, List[float]] = {}

    async def call(self, client: httpx.AsyncClient, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        resp = await client.post(f"{self.base_url}{path}", headers=self.headers, json=payload, timeout=120.0)
        elapsed = (time.perf_counter() - started) * 1000
        self.timings.setdefault(path, []).append(elapsed)
        resp.raise_for_status()
        return resp.json()


def _report_timings(api: "Hopit") -> None:
    print("\n=== latency (ms) ===")
    for path, values in api.timings.items():
        print(f"  {path:<26} n={len(values):<3} median={statistics.median(values):.0f}  max={max(values):.0f}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="garments to ingest for the benchmark")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-styling", action="store_true", help="also try /v1/outfits:rank")
    args = ap.parse_args()

    garments = await _load(args.limit)
    print(f"loaded {len(garments)} garments with canonical images")
    print(f"  with our own embedding: {sum(1 for g in garments if g['our_embedding'])}")
    print(f"  sample image_url: {garments[0]['image_url'] if garments else '-'}")

    base_url, api_key = os.getenv("HOPIT_BASE_URL"), os.getenv("HOPIT_API_KEY")
    header = os.getenv("HOPIT_AUTH_HEADER", "Authorization")

    if args.dry_run or not (base_url and api_key):
        if not (base_url and api_key):
            print("\nHOPIT_BASE_URL / HOPIT_API_KEY are not set — showing the payloads only.")
        print("\n=== POST /v1/garments:batch (first garment) ===")
        g = garments[0]
        print(json.dumps({"garments": [{k: g[k] for k in ("garment_id", "image_url", "owner_ref")} |
                                       {"attributes": {k: v for k, v in list(g["attributes"].items())[:8]}}]},
                         indent=2)[:900])
        print("\n=== POST /v1/search/text ===")
        print(json.dumps({"candidates": [x["garment_id"] for x in garments[:3]] + ["..."],
                          "query": TEXT_QUERIES[0], "limit": 20}, indent=2))
        print("\n=== POST /v1/embeddings ===")
        print(json.dumps({"garment_ids": [garments[0]["garment_id"]], "space": "moda-768"}, indent=2))
        return

    api = Hopit(base_url, api_key, header)
    ids = [g["garment_id"] for g in garments]

    async with httpx.AsyncClient() as client:
        print("\n=== 1. ingestion ===")
        for start in range(0, len(garments), 200):  # scope doc: batches of up to 200
            chunk = garments[start : start + 200]
            body = {"garments": [
                {"garment_id": g["garment_id"], "image_url": g["image_url"],
                 "owner_ref": g["owner_ref"], "attributes": g["attributes"]}
                for g in chunk
            ]}
            result = await api.call(client, "/v1/garments:batch", body)
            print(f"  sent {len(chunk)}: {json.dumps(result)[:220]}")

        print("\n=== 2. text -> image retrieval ===")
        for query in TEXT_QUERIES:
            try:
                res = await api.call(client, "/v1/search/text", {"candidates": ids, "query": query, "limit": 5})
                hits = res.get("results", [])[:3]
                print(f"  {query!r}")
                for h in hits:
                    match = next((g for g in garments if g["garment_id"] == h.get("garment_id")), None)
                    name = (match or {}).get("attributes", {}).get("casual_name") or "?"
                    print(f"      {h.get('score'):.3f}  {name}")
            except Exception as e:
                print(f"  ! {query!r}: {type(e).__name__}: {str(e)[:120]}")

        print("\n=== 3. text + image -> image retrieval ===")
        for instruction in COMPOSED_INSTRUCTIONS:
            try:
                res = await api.call(client, "/v1/search/composed", {
                    "candidates": ids, "reference": {"garment_id": ids[0]},
                    "instruction": instruction, "limit": 5,
                })
                print(f"  {instruction!r} -> {json.dumps(res.get('results', [])[:3])[:200]}")
            except Exception as e:
                print(f"  ! {instruction!r}: {type(e).__name__}: {str(e)[:120]}")

        print("\n=== 4. embeddings (vs our stored SigLIP vectors) ===")
        try:
            res = await api.call(client, "/v1/embeddings", {"garment_ids": ids[:20], "space": "moda-768"})
            print(f"  model_version: {res.get('model_version')}")
            agreements = []
            by_id = {e["garment_id"]: e for e in res.get("embeddings", [])}
            for g in garments[:20]:
                theirs, ours = by_id.get(g["garment_id"]), g["our_embedding"]
                if not theirs or not ours:
                    continue
                a = np.array(theirs["vector"], dtype=float)
                b = np.array(ours, dtype=float)
                if a.shape != b.shape:
                    print(f"  dim differs: theirs={a.shape[0]} ours={b.shape[0]}")
                    break
                agreements.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            if agreements:
                print(f"  cosine(theirs, ours) over {len(agreements)}: "
                      f"mean={statistics.mean(agreements):.4f} min={min(agreements):.4f}")
                print("  (near 1.0 would mean the same space; a low value just means a different space)")
        except Exception as e:
            print(f"  ! embeddings failed: {type(e).__name__}: {str(e)[:160]}")

        if args.include_styling:
            print("\n=== 5. styling (marked 'in development' in the scope doc) ===")
            try:
                res = await api.call(client, "/v1/outfits:rank", {
                    "candidates": ids,
                    "context": {"occasion": "dinner", "formality": 0.4, "season": "summer",
                                "temperature_c": 28, "precipitation_probability": 0.2},
                    "preferences": {"boldness": 0.72, "warm_tones": 0.65},
                    "usage": {"highly_worn": [], "highly_neglected": []},
                    "outfit_count": 3,
                })
                print(json.dumps(res, indent=2)[:900])
            except Exception as e:
                print(f"  not available yet (expected): {type(e).__name__}: {str(e)[:140]}")

    _report_timings(api)


if __name__ == "__main__":
    asyncio.run(main())
