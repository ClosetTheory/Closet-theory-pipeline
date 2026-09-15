"""Fetch garment images for a set of object keys exported from production.

`garments.original_image_refs` and `garments.enhanced_image_ref` hold bare object keys
(`wardrobe/<user>/raw/<file>.jpg`, `processed/<id>.canonical.png`), not URLs, so a base has to
be supplied. Two shapes work:

    # 1. Object storage directly (public bucket, or a bucket you hold keys for)
    python -m scripts.fetch_garment_images --keys unknown_material_samples/object_keys.json \
        --base https://<bucket>.<region>.digitaloceanspaces.com

    # 2. Through the app's own media route, which is what the product itself uses
    python -m scripts.fetch_garment_images --keys unknown_material_samples/object_keys.json \
        --base https://<app-domain>/api/media --header "Authorization: Bearer <token>"

Files land as `<out>/<garment_id>__<kind>.<ext>` so each image stays tied to the garment record
it came from. Already-downloaded files are skipped, so the script is safe to re-run.
"""

import argparse
import asyncio
import json
import os
from typing import Dict, List, Tuple

import httpx


def load_keys(path: str) -> List[Tuple[str, str, str]]:
    """Accepts the JSON triples written by the export, or a plain newline-separated key list."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if text.startswith("["):
        return [tuple(row) for row in json.loads(text)]
    return [("key%04d" % i, "raw", line.strip()) for i, line in enumerate(text.splitlines()) if line.strip()]


async def fetch_one(
    client: httpx.AsyncClient, base: str, garment_id: str, kind: str, key: str, out: str,
    sem: asyncio.Semaphore,
) -> Tuple[str, str]:
    ext = os.path.splitext(key)[1] or ".jpg"
    dest = os.path.join(out, f"{garment_id}__{kind}{ext}")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped", dest
    url = f"{base.rstrip('/')}/{key.lstrip('/')}"
    async with sem:
        try:
            resp = await client.get(url, timeout=60.0, follow_redirects=True)
        except Exception as e:
            return f"error {type(e).__name__}", url
    if resp.status_code != 200:
        return f"HTTP {resp.status_code}", url
    if not resp.content:
        return "empty", url
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return "ok", dest


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", required=True, help="object_keys.json (triples) or a .txt key list")
    ap.add_argument("--base", required=True, help="bucket URL or <app>/api/media")
    ap.add_argument("--out", default="unknown_material_samples/images")
    ap.add_argument("--header", action="append", default=[], help='e.g. "Authorization: Bearer ..."')
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_keys(args.keys)
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    headers: Dict[str, str] = {}
    for h in args.header:
        name, _, value = h.partition(":")
        headers[name.strip()] = value.strip()

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(
            *(fetch_one(client, args.base, gid, kind, key, args.out, sem) for gid, kind, key in rows)
        )

    counts: Dict[str, int] = {}
    for status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    print(f"{len(rows)} keys -> " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for status, where in results:
        if status not in ("ok", "skipped"):
            print(f"  {status:<18} {where}")
            break   # one example is enough to diagnose a wrong base or missing auth
    print(f"images in {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
