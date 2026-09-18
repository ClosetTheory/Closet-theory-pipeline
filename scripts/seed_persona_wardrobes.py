"""Fill the 22 evaluation characters' wardrobes by cloning garments that are already ingested.

The panel needs wardrobes at realistic and unrealistic sizes before stylists can judge anything.
Running ~3,200 photographs through the nine-stage pipeline would cost real money twice over --
once for classification and attribute extraction, again for Stage 4 image generation -- to produce
canonical images we already have. So this clones instead: every garment that already reached
COMPLETED is copied into a character's wardrobe as new rows pointing at the *same* stored objects.

What that means concretely:

- **No model calls and no storage writes.** A clone is four inserts (two ImageAsset rows, a
  Garment, a GarmentEmbedding) whose `object_uri` values are the originals'. Stage 4's canonical
  image and Stage 5's vector come across intact, so the wardrobes are immediately searchable and
  styleable.
- **ImageAsset rows are cloned, not shared.** They carry `tenant_id`/`member_id`, and pointing two
  tenants at one row would put a hole in the isolation every endpoint relies on. Duplicating the
  row is free; duplicating the object would not be.
- **Source garments are reused across characters.** The pool is smaller than the target, so the
  same garment appears in several wardrobes. That is fine -- they are different tenants -- and it
  gives per-character duplicate detection something real to find.

Deliberate gender crossing: some characters are seeded partly from the opposite gender's garments
(`CROSS_GENDER` below). `app/styling/orchestrator.py` defaults styling gender from `User.gender`,
and `Garment.gender` is a filter, so a men's wardrobe holding 15% women's garments is the case
that shows whether the engine filters, ignores, or quietly recommends them.

    python -m scripts.seed_persona_wardrobes --dry-run     # pool, plan, no writes
    python -m scripts.seed_persona_wardrobes
    python -m scripts.seed_persona_wardrobes --only shalini_rao
    python -m scripts.seed_persona_wardrobes --purge       # remove cloned garments only

Re-running tops each wardrobe up to its target rather than duplicating: clones are marked in
`provenance` and counted before anything is inserted.
"""

import argparse
import asyncio
import random
from typing import Dict, List, Optional, Sequence

from sqlalchemy import delete, func, select

from app.database import AsyncSessionLocal
from app.models.base import generate_uuid
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.persona import Persona
from app.pipeline.state_machine import GarmentState

# Marks a row as produced by this script, so `--purge` and the top-up count can find them without
# touching anything a stylist or the pipeline created.
CLONE_MARKER = "seeded_wardrobe_clone"

# Sizes span 150 to 2,500, averaging ~560. Each is chosen to fit the character rather than being
# an arbitrary stress number: Rohan's roster note is a deliberately minimal wardrobe, so he sits
# at the floor, while Shalini wears ethnic as workwear and so plausibly owns the most.
#
# The 2,500 goes to a woman on purpose. The clone pool has 1,055 distinct garments available to a
# women's wardrobe against 413 for a men's, so the same target on a man would repeat every garment
# six times over instead of roughly twice.
TARGETS: Dict[str, int] = {
    "shalini_rao": 2500,

    "kabir_malhotra": 900,
    "meenakshi_subramaniam": 900,
    "priyanka_jadhav": 900,

    "aarushi_deshpande": 600,
    "fatima_qureshi": 600,
    "gurpreet_singh_bhullar": 600,
    "neha_gupta": 600,
    "nikhil_chatterjee": 600,
    "vikram_rathore": 600,

    "arjun_nair": 400,
    "devansh_patel": 400,
    "harsh_vardhan_sinha": 400,
    "ira_sengupta": 400,
    "joseph_fernandes": 400,
    "riya_menon": 400,
    "tanvi_sethi": 400,

    "ananya_baruah": 150,
    "bhavna_chauhan": 150,
    "lhamu_bhutia": 150,
    "rohan_iyer": 150,
    "samar_ali_khan": 150,
}
DEFAULT_TARGET = 400

# Fraction of each wardrobe drawn from the opposite gender. Chosen to cover both directions and
# both wardrobe sizes, so the failure (if there is one) cannot be blamed on scale alone.
CROSS_GENDER: Dict[str, float] = {
    "rohan_iyer": 0.30,          # small men's wardrobe, heavily crossed
    "vikram_rathore": 0.15,      # large men's wardrobe, lightly crossed
    "tanvi_sethi": 0.25,         # women's wardrobe taking men's garments
    "kabir_malhotra": 0.10,      # the 600 -- crossing at volume
}

OPPOSITE = {"women": "men", "men": "women"}


async def load_pool(session) -> Dict[str, List[Garment]]:
    """Completed garments that were not themselves produced by this script, grouped by gender."""
    rows = (await session.execute(
        select(Garment).where(Garment.status == GarmentState.COMPLETED.value)
    )).scalars().all()
    pool: Dict[str, List[Garment]] = {"women": [], "men": [], "unisex": []}
    for g in rows:
        if (g.provenance or {}).get("source") == CLONE_MARKER:
            continue
        if not g.canonical_image_id:
            # Without a canonical image the clone would be a catalogue entry with nothing to
            # show and nothing Stage 5 ever embedded.
            continue
        pool.setdefault(g.gender or "unisex", []).append(g)
    return pool


def pick(pool: Dict[str, List[Garment]], gender: str, n: int, rng: random.Random) -> List[Garment]:
    """Draw n garments of a gender, cycling the pool when it is smaller than the request."""
    src = list(pool.get(gender) or [])
    src += list(pool.get("unisex") or [])
    if not src:
        src = [g for gs in pool.values() for g in gs]
    if not src:
        return []
    out: List[Garment] = []
    while len(out) < n:
        batch = src[:]
        rng.shuffle(batch)
        out.extend(batch[: n - len(out)])
    return out[:n]


async def clone_image(session, asset_id: Optional[str], tenant: str, member: str,
                      cache: Dict[str, str]) -> Optional[str]:
    """Copy an ImageAsset row for the new owner, pointing at the same stored object."""
    if not asset_id:
        return None
    key = f"{asset_id}:{member}"
    if key in cache:
        return cache[key]
    src = await session.get(ImageAsset, asset_id)
    if not src:
        return None
    new = ImageAsset(
        id=generate_uuid("img"), tenant_id=tenant, member_id=member,
        object_uri=src.object_uri, mime_type=src.mime_type,
        width=src.width, height=src.height, sha256=src.sha256,
    )
    session.add(new)
    await session.flush()
    cache[key] = new.id
    return new.id


async def clone_garment(session, src: Garment, persona: Persona, cache: Dict[str, str]) -> str:
    # Both ids are the backing account's own id -- scripts.seed_personas sets
    # tenant_id = member_id = user_id, and get_acting_scope reads them straight off that row. A
    # cloned garment carrying anything else is invisible to the character it was cloned for.
    tenant = member = persona.user_id

    source_id = await clone_image(session, src.source_image_id, tenant, member, cache)
    canon_id = await clone_image(session, src.canonical_image_id, tenant, member, cache)

    gid = generate_uuid("garm")
    session.add(Garment(
        id=gid, tenant_id=tenant, member_id=member,
        source_image_id=source_id, canonical_image_id=canon_id,
        image_type=src.image_type, garment_crop_refs=list(src.garment_crop_refs or []),
        detected_label=src.detected_label, gender=src.gender,
        category=src.category, subcategory=src.subcategory, garment_class=src.garment_class,
        attributes_json=dict(src.attributes_json or {}),
        compatibility_features=dict(src.compatibility_features or {}),
        status=src.status, quality_status=src.quality_status,
        pipeline_version=src.pipeline_version,
        provenance={**(src.provenance or {}), "source": CLONE_MARKER,
                    "cloned_from": src.id, "persona_slug": persona.slug},
    ))
    await session.flush()

    emb = (await session.execute(
        select(GarmentEmbedding).where(GarmentEmbedding.garment_id == src.id)
    )).scalars().first()
    if emb:
        # No tenant_id/member_id here: the embedding is scoped through its garment, which is
        # already the clone's.
        session.add(GarmentEmbedding(
            id=generate_uuid("emb"), garment_id=gid,
            embedding=list(emb.embedding), model=emb.model, model_version=emb.model_version,
            dimension=emb.dimension, source_image_version=emb.source_image_version,
        ))
    return gid


async def existing_count(session, persona: Persona) -> int:
    return (await session.execute(
        select(func.count()).select_from(Garment)
        .where(Garment.member_id == persona.user_id)
    )).scalar_one()


async def run(only: Optional[str], dry_run: bool, purge: bool, seed: int) -> None:
    rng = random.Random(seed)
    async with AsyncSessionLocal() as session:
        personas = (await session.execute(
            select(Persona).order_by(Persona.slug)
        )).scalars().all()
        if only:
            personas = [p for p in personas if p.slug == only]
        if not personas:
            raise SystemExit("no personas found — run scripts.seed_personas first")

        if purge:
            ids = (await session.execute(
                select(Garment.id).where(Garment.provenance["source"].astext == CLONE_MARKER)
            )).scalars().all()
            print(f"purging {len(ids)} cloned garments")
            if not dry_run and ids:
                await session.execute(delete(GarmentEmbedding)
                                      .where(GarmentEmbedding.garment_id.in_(ids)))
                await session.execute(delete(Garment).where(Garment.id.in_(ids)))
                await session.commit()
            return

        pool = await load_pool(session)
        total_pool = sum(len(v) for v in pool.values())
        print(f"pool: {total_pool} completed garments  "
              + "  ".join(f"{k}={len(v)}" for k, v in pool.items()))
        if not total_pool:
            raise SystemExit("no COMPLETED garments with a canonical image to clone")

        planned = 0
        for p in personas:
            target = TARGETS.get(p.slug, DEFAULT_TARGET)
            have = await existing_count(session, p)
            need = max(0, target - have)
            cross = CROSS_GENDER.get(p.slug, 0.0)
            n_cross = int(round(need * cross))
            planned += need
            print(f"  {p.slug:26s} target {target:4d}  have {have:4d}  add {need:4d}"
                  + (f"  ({n_cross} cross-gender from {OPPOSITE.get(p.gender, 'other')})"
                     if n_cross else ""))
        print(f"\n{planned} garments to clone across {len(personas)} characters")
        if dry_run:
            print("dry run — nothing written")
            return

        cache: Dict[str, str] = {}
        for p in personas:
            target = TARGETS.get(p.slug, DEFAULT_TARGET)
            have = await existing_count(session, p)
            need = max(0, target - have)
            if not need:
                print(f"  {p.slug:26s} already at {have}")
                continue
            cross = CROSS_GENDER.get(p.slug, 0.0)
            n_cross = int(round(need * cross))
            own = pick(pool, p.gender, need - n_cross, rng)
            other = pick(pool, OPPOSITE.get(p.gender, p.gender), n_cross, rng) if n_cross else []
            for i, src in enumerate(own + other, 1):
                await clone_garment(session, src, p, cache)
                if i % 100 == 0:
                    await session.commit()
                    print(f"  {p.slug:26s} {i}/{need}")
            await session.commit()
            print(f"  {p.slug:26s} +{need} (now {await existing_count(session, p)})"
                  + (f", {n_cross} cross-gender" if n_cross else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="a single persona slug")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", action="store_true", help="delete cloned garments and stop")
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()
    asyncio.run(run(args.only, args.dry_run, args.purge, args.seed))


if __name__ == "__main__":
    main()
