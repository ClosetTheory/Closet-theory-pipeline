"""Delete every generated outfit (and cached OOTD pick) for the evaluation characters, so
scripts.daily_character_styling can regenerate them correctly-likeness after the mannequin fix.

Every outfit generated before that fix rendered characters on the anonymous mannequin instead of
their own portrait. Those rows are actively wrong to leave in place -- a stylist reviewing them is
judging a figure that doesn't belong to the character at all -- so this removes them rather than
leaving them to be quietly outnumbered by new, correct ones.

**Why OutfitOfTheDay has to go too, not just Outfit.** OutfitOfTheDay caches by
`styling_request_id`, not `outfit_id` (see app/models/ootd.py) -- it wraps a StylingRequest, and
that request's real Outfit rows are found by a separate query at read time. Deleting only Outfit
rows and leaving OutfitOfTheDay in place would mean `get_or_generate_ootd(force=False)` keeps
returning today's cached pick, whose StylingRequest now points at nothing: the next OOTD read
would either come back empty or error, and worse, daily_character_styling's OOTD call is
explicitly force=False, so it would never regenerate on its own. OutfitOfTheDay rows are deleted
first, before Outfit, for exactly this reason.

**Why StylingRequest is left alone.** Outfit.request_id -> StylingRequest is CASCADE in the other
direction only (deleting a StylingRequest cascades to its Outfits; deleting an Outfit does not
touch its StylingRequest), so nothing here requires touching it. StylingRequest.normalized_intent
also feeds derive_member_context's "what this member tends to ask for" signal -- keeping it means
the next generation still benefits from today's real intent history instead of starting cold.

**Generated images are deleted too, not left orphaned.** Outfit.generated_image_id -> ImageAsset
is SET NULL on delete, so the asset row (and the underlying stored object) would otherwise survive
the outfit with nothing pointing at it -- a wrong-likeness image with no owner, indistinguishable
from a real one to anything that stumbled onto its id. IDs are collected before the Outfit rows
are deleted, then the assets and their storage objects are removed explicitly.

    python -m scripts.purge_character_outfits --dry-run
    python -m scripts.purge_character_outfits --only shalini_rao
    python -m scripts.purge_character_outfits                    # all 22

Does not regenerate anything -- that's scripts.daily_character_styling, deliberately kept as a
separate script since purging and generating are different failure domains (a purge should never
be blocked on a provider being down, and a generation run should never accidentally re-purge).
"""

import argparse
import asyncio
from typing import List, Optional

from sqlalchemy import delete, select

from app.database import AsyncSessionLocal
from app.models.image_asset import ImageAsset
from app.models.ootd import OutfitOfTheDay
from app.models.persona import Persona
from app.models.styling import Outfit
from app.storage import get_storage_client


async def purge_one(session, storage, persona: Persona, dry_run: bool) -> dict:
    tenant_id = persona.user_id

    ootd_ids = (
        await session.execute(
            select(OutfitOfTheDay.id).where(OutfitOfTheDay.tenant_id == tenant_id)
        )
    ).scalars().all()

    outfit_rows = (
        await session.execute(
            select(Outfit.id, Outfit.generated_image_id).where(Outfit.tenant_id == tenant_id)
        )
    ).all()
    outfit_ids = [r[0] for r in outfit_rows]
    image_ids = [r[1] for r in outfit_rows if r[1]]

    result = {"slug": persona.slug, "ootd": len(ootd_ids), "outfits": len(outfit_ids),
              "images": len(image_ids), "image_delete_errors": 0}
    if dry_run or not (ootd_ids or outfit_ids):
        return result

    if ootd_ids:
        await session.execute(delete(OutfitOfTheDay).where(OutfitOfTheDay.id.in_(ootd_ids)))
    if outfit_ids:
        await session.execute(delete(Outfit).where(Outfit.id.in_(outfit_ids)))
    await session.commit()

    # Best-effort: a storage failure here must not resurrect the DB rows we already committed
    # removed, and must not stop the rest of this character's (or the next character's) cleanup.
    for image_id in image_ids:
        try:
            asset = await session.get(ImageAsset, image_id)
            if asset:
                try:
                    await storage.delete_object(asset.object_uri)
                except Exception:
                    pass
                await session.delete(asset)
        except Exception:
            result["image_delete_errors"] += 1
    await session.commit()

    return result


async def run(only: Optional[List[str]], dry_run: bool) -> None:
    storage = get_storage_client()
    async with AsyncSessionLocal() as session:
        personas = (await session.execute(select(Persona).order_by(Persona.slug))).scalars().all()
    if only:
        wanted = set(only)
        personas = [p for p in personas if p.slug in wanted]
        missing = wanted - {p.slug for p in personas}
        if missing:
            print(f"! unknown slug(s), skipped: {', '.join(sorted(missing))}")
    if not personas:
        raise SystemExit("no characters found")

    print(f"{'DRY RUN — ' if dry_run else ''}purging {len(personas)} character(s)\n")
    totals = {"ootd": 0, "outfits": 0, "images": 0, "image_delete_errors": 0}
    for p in personas:
        async with AsyncSessionLocal() as session:
            r = await purge_one(session, storage, p, dry_run)
        print(f"  {r['slug']:26s} ootd={r['ootd']:3d}  outfits={r['outfits']:3d}  "
              f"images={r['images']:3d}"
              + (f"  ({r['image_delete_errors']} image delete error(s))"
                 if r["image_delete_errors"] else ""))
        for k in totals:
            totals[k] += r[k]

    print(f"\ntotal: {totals['outfits']} outfit(s), {totals['ootd']} OOTD pick(s), "
          f"{totals['images']} image(s) removed"
          + (f", {totals['image_delete_errors']} image delete error(s)"
             if totals["image_delete_errors"] else ""))
    if dry_run:
        print("dry run — nothing was actually deleted")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="a persona slug, or a comma-separated list of slugs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    asyncio.run(run(only, args.dry_run))


if __name__ == "__main__":
    main()
