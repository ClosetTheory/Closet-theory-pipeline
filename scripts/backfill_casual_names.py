"""Backfill `casual_name` — the everyday name a person would use for a garment.

Newly ingested garments get this from Stage 3 for free, since it is one extra field in an
extraction call that already happens. Garments ingested before that field existed need filling
in, and re-running Stage 3 would mean paying for a fresh vision call per garment to recover
information already sitting in the database.

So this composes the name from the stored attributes instead, with a text-only model: colour,
pattern, pattern_detail, visual_description and brand all already describe the print, graphic or
standout feature the name should lead with. Garments extracted before the enriched schema have
only the basics, and correctly end up with a plainer name like "BLACK LEATHER JACKET".

Batches many garments per request — one call per garment would be both slow and needlessly
expensive for a short label. Safe to re-run: only garments still missing a name are touched.

    python -m scripts.backfill_casual_names --dry-run
    python -m scripts.backfill_casual_names [--batch 20] [--limit N]
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.garment import Garment

MAX_DESC = 320


def _summarise(garment_id: str, attrs: Dict[str, Any], subcategory: Optional[str]) -> Dict[str, Any]:
    """Only the fields that help name the garment — colour, what is printed on it, what it is."""
    colour = attrs.get("colour")
    if isinstance(colour, list):
        colour = ", ".join(colour)
    fields = {
        "id": garment_id,
        "type": attrs.get("subcategory") or subcategory or attrs.get("category"),
        "colour": colour,
        "pattern": attrs.get("pattern"),
        "pattern_detail": attrs.get("pattern_detail"),
        "motif": attrs.get("pattern_motif"),
        "material": attrs.get("material"),
        "brand": attrs.get("brand_label"),
        "sleeve_length": attrs.get("sleeve_length"),
        "collar": attrs.get("collar_detail"),
        "description": (attrs.get("visual_description") or "")[:MAX_DESC] or None,
    }
    return {k: v for k, v in fields.items() if v}


async def _name_batch(client: httpx.AsyncClient, items: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt = (
        "For each garment below, give the short everyday name its owner would use when picking "
        "it out of their own wardrobe, in UPPERCASE.\n\n"
        "Lead with what makes the piece recognisable at a glance — its colour, and any "
        "print/graphic/character/logo or standout feature — then the garment type. 2-5 words.\n"
        'Examples: "RED SPIDERMAN TSHIRT", "RED SPIDERMAN POLO TSHIRT", "BLACK LEATHER BIKER '
        'JACKET", "BEIGE RED STRIPED BELT", "WHITE RUFFLED BLOUSE".\n\n'
        "Use ONLY what the given fields state. Never invent a character, brand or motif that is "
        "not mentioned. A plain garment should simply be colour plus type, e.g. "
        '"NAVY CREWNECK TSHIRT". No punctuation or quotes inside a name.\n\n'
        f"Garments:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        'Return ONLY JSON: {"names": {"<id>": "<NAME>", ...}} covering every id given.'
    )
    resp = await client.post(
        f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    names = data.get("names") or {}
    return {str(k): str(v).strip().upper() for k, v in names.items() if str(v).strip()}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Garment.id, Garment.attributes_json, Garment.subcategory)
                .where(Garment.status == "COMPLETED")
                .order_by(Garment.id)
            )
        ).all()

    todo = [r for r in rows if not (r[1] or {}).get("casual_name")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"completed garments: {len(rows)}   missing a casual name: {len(todo)}")

    if args.dry_run:
        sample = [_summarise(r[0], r[1] or {}, r[2]) for r in todo[:3]]
        print("\nwould send, e.g.:")
        for s in sample:
            print(" ", json.dumps(s, ensure_ascii=False)[:200])
        return
    if not todo:
        return

    named = failed = 0
    async with httpx.AsyncClient() as client:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start : start + args.batch]
            payload = [_summarise(r[0], r[1] or {}, r[2]) for r in chunk]
            try:
                names = await _name_batch(client, payload)
            except Exception as e:
                failed += len(chunk)
                print(f"  ! batch at {start} failed: {type(e).__name__}: {str(e)[:110]}")
                continue

            async with AsyncSessionLocal() as session:
                for garment_id, attrs, _sub in chunk:
                    name = names.get(garment_id)
                    if not name:
                        failed += 1
                        continue
                    garment = await session.get(Garment, garment_id)
                    if not garment:
                        failed += 1
                        continue
                    # Replace the dict wholesale and flag it: SQLAlchemy does not track
                    # mutations made inside a JSON column in place.
                    garment.attributes_json = {**(garment.attributes_json or {}), "casual_name": name}
                    flag_modified(garment, "attributes_json")
                    named += 1
                await session.commit()

            done = min(start + args.batch, len(todo))
            print(f"  {done}/{len(todo)} processed, {named} named, {failed} failed")

    print(f"\nDone. {named} named, {failed} failed.")


if __name__ == "__main__":
    asyncio.run(main())
