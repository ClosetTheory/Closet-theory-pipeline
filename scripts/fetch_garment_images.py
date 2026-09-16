"""Fetch garment images for a set of object keys exported from production.

`garments.original_image_refs` and `garments.enhanced_image_ref` hold bare object keys
(`wardrobe/<user>/raw/<file>.jpg`, `processed/<id>.canonical.png`), not URLs, so a source has to
be supplied. Two modes:

    # 1. Object storage, using the DOS_* credentials in .env (the usual case)
    python -m scripts.fetch_garment_images --keys unknown_material_samples/object_keys.json --s3

    # 2. Over HTTP, through the app's own media route
    python -m scripts.fetch_garment_images --keys unknown_material_samples/object_keys.json \
        --base https://<app-domain>/api/media --header "Authorization: Bearer <token>"

Note the `--prefix`, which defaults to `production/`. The bucket holds more than one environment
at its root — a bare `wardrobe/...` key resolves to a different environment's copy, or to
nothing — so the stored key is only half the address. Getting this wrong returns 404 rather than
the wrong image, which is the good kind of wrong.

Files land as `<out>/<garment_id>__<kind>.<ext>` so every image stays tied to the record it came
from. Already-downloaded files are skipped, so re-runs are cheap.
"""

import argparse
import asyncio
import io
import json
import os
from typing import Dict, List, Optional, Tuple

import httpx

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_env() -> Dict[str, str]:
    """Reads .env directly rather than through app.config, which would require the whole
    settings model to validate for what is a standalone utility."""
    out: Dict[str, str] = {}
    if not os.path.exists(ENV_PATH):
        return out
    with io.open(ENV_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip()
    return out


def load_keys(path: str) -> List[Tuple[str, str, str]]:
    """Accepts the JSON triples written by the export, or a plain newline-separated key list."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if text.startswith("["):
        return [tuple(row) for row in json.loads(text)]
    return [("key%04d" % i, "raw", line.strip()) for i, line in enumerate(text.splitlines()) if line.strip()]


def _dest(out: str, garment_id: str, kind: str, key: str) -> str:
    ext = os.path.splitext(key)[1] or ".jpg"
    return os.path.join(out, f"{garment_id}__{kind}{ext}")


def _already(dest: str) -> bool:
    return os.path.exists(dest) and os.path.getsize(dest) > 0


# --- object storage -----------------------------------------------------------------------


def fetch_via_s3(rows: List[Tuple[str, str, str]], out: str, prefix: str) -> Dict[str, int]:
    import boto3
    from botocore.config import Config

    env = load_env()
    missing = [k for k in ("DOS_ACCESS_KEY_ID", "DOS_SECRET_ACCESS_KEY", "DOS_BUCKET", "DOS_REGION")
               if not env.get(k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")

    bucket, region = env["DOS_BUCKET"], env["DOS_REGION"]
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        region_name=region,
        aws_access_key_id=env["DOS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["DOS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )
    print(f"bucket {bucket} ({region}), prefix {prefix!r}")

    counts: Dict[str, int] = {}
    for garment_id, kind, key in rows:
        dest = _dest(out, garment_id, kind, key)
        if _already(dest):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        full = f"{prefix}{key.lstrip('/')}"
        try:
            client.download_file(bucket, full, dest)
            counts["ok"] = counts.get("ok", 0) + 1
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
            counts[f"error {code}"] = counts.get(f"error {code}", 0) + 1
            if counts[f"error {code}"] == 1:
                print(f"  first {code}: {full}")
    return counts


# --- over HTTP ----------------------------------------------------------------------------


async def fetch_via_http(
    rows: List[Tuple[str, str, str]], out: str, base: str, headers: Dict[str, str], concurrency: int
) -> Dict[str, int]:
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, garment_id: str, kind: str, key: str) -> str:
        dest = _dest(out, garment_id, kind, key)
        if _already(dest):
            return "skipped"
        async with sem:
            try:
                resp = await client.get(
                    f"{base.rstrip('/')}/{key.lstrip('/')}", timeout=60.0, follow_redirects=True
                )
            except Exception as exc:
                return f"error {type(exc).__name__}"
        if resp.status_code != 200 or not resp.content:
            return f"HTTP {resp.status_code}"
        with open(dest, "wb") as fh:
            fh.write(resp.content)
        return "ok"

    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(*(one(client, g, k, key) for g, k, key in rows))

    counts: Dict[str, int] = {}
    for status in results:
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", required=True, help="object_keys.json (triples) or a .txt key list")
    ap.add_argument("--s3", action="store_true", help="fetch from Spaces using the DOS_* credentials")
    ap.add_argument("--prefix", default="production/", help="key prefix inside the bucket")
    ap.add_argument("--base", help="HTTP base, e.g. https://<app>/api/media (instead of --s3)")
    ap.add_argument("--out", default="unknown_material_samples/images")
    ap.add_argument("--header", action="append", default=[], help='e.g. "Authorization: Bearer ..."')
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kind", choices=["raw", "canonical"], help="fetch only one kind")
    args = ap.parse_args()

    rows = load_keys(args.keys)
    if args.kind:
        rows = [r for r in rows if r[1] == args.kind]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(rows)} object key(s) -> {args.out}")

    if args.s3:
        counts = fetch_via_s3(rows, args.out, args.prefix)
    elif args.base:
        headers = {}
        for h in args.header:
            name, _, value = h.partition(":")
            headers[name.strip()] = value.strip()
        counts = asyncio.run(fetch_via_http(rows, args.out, args.base, headers, args.concurrency))
    else:
        raise SystemExit("Pass --s3, or --base for the HTTP route.")

    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    on_disk = [f for f in os.listdir(args.out) if not f.startswith(".")]
    print(f"{len(on_disk)} file(s) in {args.out}")


if __name__ == "__main__":
    main()
