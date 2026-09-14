"""Regenerate canonical images that the audit found don't depict their real garment.

Reads the JSON written by scripts/audit_canonical_images.py, picks the failures at or below a
severity threshold, and re-runs Stage 4 for each via the existing single-stage /step endpoint.

This repairs legacy damage rather than working around a broken generator: the canonical images
were produced by older Stage 4 code and approved by a verifier that failed open (returned
is_valid=True / score=0.9 on any error, above the 0.75 accept threshold). Today's Stage 4 was
re-run by hand on a garment whose belt had been rendered as a white dress shirt and scored 0.95,
producing an actual belt. The verifier now fails closed, so a regeneration that goes wrong is
reported here instead of being silently accepted again.

Image generation costs real money per garment, so this defaults to a small diverse trial and
prints a per-category breakdown of what it would touch before doing anything.

Re-embedding is deliberately NOT done here — run it afterwards, since the stored vector is a
real embedding of the *previous* render and the noise detector will not flag it:

    python -m scripts.backfill_embeddings --ids <ids...>

    python -m scripts.repair_canonical_images --max-score 0.2 --limit 10 --dry-run
    python -m scripts.repair_canonical_images --max-score 0.2 --limit 10
    python -m scripts.repair_canonical_images --max-score 0.2
"""

import argparse
import asyncio
import collections
import json
from typing import Any, Dict, List

import httpx


def _select(results: List[Dict[str, Any]], max_score: float, limit: int | None) -> List[Dict[str, Any]]:
    failures = [
        r
        for r in results
        if r.get("error") is None
        and (not r.get("is_valid") or (r.get("score") or 0) <= max_score)
    ]
    failures.sort(key=lambda r: r.get("score") or 0)
    if not limit or limit >= len(failures):
        return failures

    # Spread a small trial across subcategories rather than taking 10 of the same item, so the
    # success rate it reports actually generalises to the full batch.
    by_sub: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in failures:
        by_sub[r.get("subcategory") or "?"].append(r)
    picked: List[Dict[str, Any]] = []
    while len(picked) < limit:
        progressed = False
        for sub in sorted(by_sub):
            if by_sub[sub]:
                picked.append(by_sub[sub].pop(0))
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    return picked


async def _repair_one(client: httpx.AsyncClient, base_url: str, target: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    garment_id = target["garment_id"]
    async with sem:
        out = {
            "garment_id": garment_id,
            "subcategory": target.get("subcategory"),
            "old_score": target.get("score"),
        }
        try:
            resp = await client.post(
                f"{base_url}/api/v1/wardrobe/garments/{garment_id}/step",
                json={"stage": "STAGE_04_DIGITISE", "force": True},
                timeout=420.0,
            )
            data = resp.json() if resp.content else {}
            refs = data.get("output_data") or data.get("output_refs") or {}
            history = refs.get("verification_history") or []
            last = history[-1] if history else {}
            out.update(
                {
                    "status": data.get("status"),
                    "new_score": refs.get("quality_score", last.get("score")),
                    "reason": last.get("reason"),
                    "error": data.get("error") if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:160]}",
                }
            )
        except Exception as e:
            out.update({"status": "ERROR", "new_score": None, "error": f"{type(e).__name__}: {e}"})
        return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", default="/tmp/canonical_audit_full.json")
    ap.add_argument("--max-score", type=float, default=0.2, help="repair failures scoring at or below this (default 0.2 = wrong garment entirely)")
    ap.add_argument("--limit", type=int, default=None, help="trial size, spread across subcategories")
    ap.add_argument("--concurrency", type=int, default=3, help="image generation is slow and paid; keep this modest")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="/tmp/canonical_repair.json")
    args = ap.parse_args()

    with open(args.audit, encoding="utf-8") as f:
        results = json.load(f)

    targets = _select(results, args.max_score, args.limit)
    by_sub = collections.Counter(t.get("subcategory") or "?" for t in targets)
    print(f"selected {len(targets)} garment(s) at score <= {args.max_score}")
    print("  " + ", ".join(f"{s}:{c}" for s, c in by_sub.most_common(18)))

    if args.dry_run:
        print("\n[dry run] no image generated, nothing written.")
        return
    if not targets:
        return

    sem = asyncio.Semaphore(args.concurrency)
    repaired: List[Dict[str, Any]] = []
    done = 0
    async with httpx.AsyncClient() as client:
        for coro in asyncio.as_completed([_repair_one(client, args.base_url, t, sem) for t in targets]):
            r = await coro
            repaired.append(r)
            done += 1
            print(f"  [{done}/{len(targets)}] {(r['subcategory'] or '?'):<18} {r['old_score']} -> {r['new_score']}  {r['status']}")

    ok = [r for r in repaired if r["status"] == "SUCCEEDED" and (r["new_score"] or 0) >= 0.75]
    bad = [r for r in repaired if r not in ok]
    print(f"\n=== repaired {len(ok)}/{len(repaired)} ===")
    if bad:
        print("still failing / errored:")
        for r in bad[:20]:
            print(f"  {(r['subcategory'] or '?'):<18} {r['garment_id'][:24]} {r['status']} {(r.get('error') or r.get('reason') or '')[:80]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(repaired, f, indent=2, default=str)
    print(f"\nresults -> {args.out}")
    if ok:
        print("\nNow re-embed the repaired garments (their stored vector is still the old render):")
        print("  python -m scripts.backfill_embeddings --ids " + " ".join(r["garment_id"] for r in ok[:6]) + (" ..." if len(ok) > 6 else ""))


if __name__ == "__main__":
    asyncio.run(main())
