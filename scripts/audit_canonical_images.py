"""Audit: retroactively verify that each garment's canonical image depicts its real garment.

Stage 4 generates the canonical studio image and Stage 5 embeds it, so a hallucinated canonical
silently becomes the garment's identity everywhere downstream — dedup, similarity retrieval and
the styling shortlist all describe a garment the member does not own. Confirmed by hand: a
striped belt and a wide-brim hat were both rendered as the same white dress shirt, then embedded
as near-identical "duplicates" of each other.

They were approved because the verifier failed open. `validate_digitisation` used to return
(is_valid=True, score=0.9) whenever it had no API key or the call raised — above the 0.75 accept
threshold — so any hiccup rubber-stamped whatever the generator produced. Measured on this
wardrobe, 1118 of 1177 digitisation runs recorded no verification history at all.

This re-runs that skipped gate over stored data, using the same provider method the pipeline
uses (a different model family from the generator, by design). Attributes are extracted from the
SOURCE photo and spot-checked as correct, so source + attributes are the reference and the
canonical image is what is on trial.

Read-only by default; --mark flags failures as REVIEW_REQUIRED for a follow-up regeneration.

    python -m scripts.audit_canonical_images --limit 60        # sample first
    python -m scripts.audit_canonical_images                   # full wardrobe
    python -m scripts.audit_canonical_images --mark            # also flag failures
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import aliased

from app.database import AsyncSessionLocal
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.providers.base import VerifierUnavailableError
from app.providers.digitisation import get_digitisation_provider
from app.schemas.attributes import GarmentAttributes
from app.storage import get_storage_client


async def _load_targets(limit: Optional[int], only_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    source_img = aliased(ImageAsset)
    canon_img = aliased(ImageAsset)
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    Garment.id,
                    Garment.subcategory,
                    Garment.detected_label,
                    Garment.attributes_json,
                    source_img.object_uri,
                    canon_img.object_uri,
                )
                .join(source_img, source_img.id == Garment.source_image_id)
                .join(canon_img, canon_img.id == Garment.canonical_image_id)
                .where(Garment.status == "COMPLETED")
                .order_by(Garment.created_at)
            )
        ).all()

    targets = [
        {
            "garment_id": r[0],
            "subcategory": r[1],
            "detected_label": r[2],
            "attributes": r[3] or {},
            "source_uri": r[4],
            "canonical_uri": r[5],
        }
        for r in rows
    ]
    if only_ids:
        targets = [t for t in targets if any(t["garment_id"].startswith(p) for p in only_ids)]
    elif limit and limit < len(targets):
        step = max(1, len(targets) // limit)
        targets = targets[::step][:limit]
    return targets


async def _verify_one(target: Dict[str, Any], provider, storage, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        out = dict(target)
        out.pop("attributes", None)
        try:
            source_bytes = await storage.get_object(target["source_uri"])
            canonical_bytes = await storage.get_object(target["canonical_uri"])
            attributes = GarmentAttributes.model_validate(target["attributes"])
            is_valid, score, reason = await provider.validate_digitisation(
                source_bytes, canonical_bytes, attributes, garment_label=target["detected_label"]
            )
            verification = getattr(provider, "_last_verification", None) or {}
            out.update(
                {
                    "is_valid": is_valid,
                    "score": score,
                    "reason": reason,
                    "mismatches": verification.get("mismatches", []),
                    "error": None,
                }
            )
        except VerifierUnavailableError as e:
            out.update({"is_valid": None, "score": None, "reason": None, "error": f"verifier unavailable: {e}"})
        except Exception as e:
            out.update({"is_valid": None, "score": None, "reason": None, "error": f"{type(e).__name__}: {e}"})
        return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="audit an evenly-spread sample of N")
    ap.add_argument("--ids", nargs="*", default=None, help="audit specific garment id prefixes")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None, help="score below which to count as a failure (default: the pipeline's own DIGITISATION_QUALITY_THRESHOLD)")
    ap.add_argument("--mark", action="store_true", help="flag failures as REVIEW_REQUIRED")
    ap.add_argument("--out", default="/tmp/canonical_audit.json")
    args = ap.parse_args()

    from app.config import settings

    threshold = args.threshold if args.threshold is not None else settings.DIGITISATION_QUALITY_THRESHOLD

    targets = await _load_targets(args.limit, args.ids)
    print(f"auditing {len(targets)} garment(s) at concurrency {args.concurrency}, threshold {threshold}")

    provider = get_digitisation_provider()
    storage = get_storage_client()
    sem = asyncio.Semaphore(args.concurrency)

    results: List[Dict[str, Any]] = []
    done = 0
    for coro in asyncio.as_completed([_verify_one(t, provider, storage, sem) for t in targets]):
        results.append(await coro)
        done += 1
        if done % 25 == 0 or done == len(targets):
            print(f"  {done}/{len(targets)}")

    errored = [r for r in results if r["error"]]
    scored = [r for r in results if r["error"] is None]
    failed = [r for r in scored if not r["is_valid"] or (r["score"] or 0) < threshold]
    passed = [r for r in scored if r not in failed]

    print(f"\n=== results ===")
    print(f"  verified ok:  {len(passed)}")
    print(f"  FAILED:       {len(failed)}" + (f"  ({100*len(failed)/len(scored):.1f}% of scored)" if scored else ""))
    print(f"  errored:      {len(errored)}")

    if failed:
        print(f"\n=== worst 25 failures ===")
        for r in sorted(failed, key=lambda x: x["score"] or 0)[:25]:
            mm = "; ".join(r["mismatches"][:2]) if r["mismatches"] else (r["reason"] or "")[:90]
            print(f"  {r['score'] if r['score'] is not None else '?':<6} {(r['subcategory'] or '?'):<18} {r['garment_id'][:24]}  {mm[:88]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nfull results -> {args.out}")

    if args.mark and failed:
        async with AsyncSessionLocal() as session:
            for r in failed:
                g = await session.get(Garment, r["garment_id"])
                if g:
                    g.quality_status = "REVIEW_REQUIRED"
            await session.commit()
        print(f"marked {len(failed)} garment(s) as REVIEW_REQUIRED")


if __name__ == "__main__":
    asyncio.run(main())
