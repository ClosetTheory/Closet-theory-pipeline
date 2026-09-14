"""One-off backfill: replace hash-seeded mock embeddings with real SigLIP vectors.

`SigLIPEmbeddingProvider` used to fall back to `MockEmbeddingProvider` on any error, writing a
hash-seeded Gaussian into `garment_embeddings` labelled as the real model. That fallback is gone
(see app/providers/base.py::EmbeddingUnavailableError), but the vectors it already wrote are
still there: measured on this wardrobe, 970 of 1176 (82.5%) were noise.

Those rows are not merely low quality, they are meaningless: two photos of the same garment get
orthogonal vectors, so near-duplicate detection (app/pipeline/stages/stage_05_embed.py) and
similarity retrieval (app/styling/retrieval.py) have both been running against random numbers.

Detection is statistical rather than by label, because the fallback recorded the real model's
name. A 768-dim random unit vector has pairwise cosine ~ N(0, 1/sqrt(768) = 0.036) against
everything, so its best match across a whole wardrobe stays near zero; a real fashion embedding
always has *something* it resembles. Measured here the two populations do not overlap at all:
mock best-match max was 0.189, real embeddings averaged 0.53 against each other.

Re-embeds from the same image Stage 5 would use (canonical -> crop -> source) and updates the
existing row in place rather than inserting another. Safe to re-run: an already-real row is
skipped, so an interrupted or partially-failed run picks up where it left off.

    python -m scripts.backfill_embeddings --dry-run
    python -m scripts.backfill_embeddings [--concurrency 4] [--threshold 0.25] [--limit N]
"""

import argparse
import asyncio
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.providers.base import EmbeddingUnavailableError
from app.providers.embedding.siglip import SigLIPEmbeddingProvider
from app.storage import get_storage_client


async def _classify(threshold: float) -> Tuple[List[str], int, int]:
    """Returns (garment_ids_with_mock_vectors, real_count, garments_missing_an_embedding)."""
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(GarmentEmbedding.garment_id, GarmentEmbedding.embedding)
            )
        ).all()
        completed_ids = {
            r[0]
            for r in (
                await session.execute(select(Garment.id).where(Garment.status == "COMPLETED"))
            ).all()
        }

    embedded_ids = {r[0] for r in rows}
    missing = len(completed_ids - embedded_ids)
    if not rows:
        return [], 0, missing

    ids = [r[0] for r in rows]
    vecs = np.array([r[1] for r in rows], dtype=np.float64)
    vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]

    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    # Ignore exact twins when asking "does this resemble anything?": several legacy rows share a
    # byte-identical source image, so they match each other at 1.0 while still being noise.
    sim = np.where(sim > 0.9999, -1.0, sim)
    best_match = sim.max(axis=1)

    mock_ids = [gid for gid, best in zip(ids, best_match) if best < threshold]
    return mock_ids, len(ids) - len(mock_ids), missing


def _pick_image_uri(garment: Garment, canonical: Optional[ImageAsset], source: Optional[ImageAsset]) -> Optional[str]:
    """Same preference order Stage 5 uses: canonical render, then crop, then raw source."""
    if canonical:
        return canonical.object_uri
    if garment.garment_crop_refs:
        return garment.garment_crop_refs[0]
    return source.object_uri if source else None


async def _reembed_one(
    garment_id: str, provider: SigLIPEmbeddingProvider, storage, sem: asyncio.Semaphore
) -> Tuple[str, bool, str]:
    async with sem:
        try:
            async with AsyncSessionLocal() as session:
                garment = await session.get(Garment, garment_id)
                if not garment:
                    return garment_id, False, "garment not found"

                canonical = (
                    await session.get(ImageAsset, garment.canonical_image_id)
                    if garment.canonical_image_id
                    else None
                )
                source = (
                    await session.get(ImageAsset, garment.source_image_id)
                    if garment.source_image_id
                    else None
                )
                image_uri = _pick_image_uri(garment, canonical, source)
                if not image_uri:
                    return garment_id, False, "no image to embed"

                image_bytes = await storage.get_object(image_uri)
                vector = await provider.embed(image_bytes)

                arr = np.array(vector, dtype=np.float64)
                norm = float(np.linalg.norm(arr))
                if len(arr) != provider.dimension or norm == 0.0:
                    return garment_id, False, f"bad vector: dim={len(arr)} norm={norm:.4f}"
                vector = (arr / norm).tolist()

                record = (
                    await session.execute(
                        select(GarmentEmbedding).where(GarmentEmbedding.garment_id == garment_id)
                    )
                ).scalars().first()
                if record is None:
                    record = GarmentEmbedding(garment_id=garment_id)
                    session.add(record)
                record.embedding = vector
                record.model = provider.model_name
                record.model_version = provider.model_version
                record.dimension = len(vector)
                await session.commit()

            return garment_id, True, "ok"
        except EmbeddingUnavailableError as e:
            return garment_id, False, f"embedding unavailable: {e}"
        except Exception as e:
            return garment_id, False, f"{type(e).__name__}: {e}"


async def _run(targets: List[str], concurrency: int) -> None:
    provider = SigLIPEmbeddingProvider()
    storage = get_storage_client()
    sem = asyncio.Semaphore(concurrency)

    print(f"\nRe-embedding {len(targets)} garment(s) at concurrency {concurrency}...")
    done = 0
    failed: List[Tuple[str, str]] = []
    for coro in asyncio.as_completed([_reembed_one(gid, provider, storage, sem) for gid in targets]):
        garment_id, ok, detail = await coro
        done += 1
        if not ok:
            failed.append((garment_id, detail))
        if done % 25 == 0 or done == len(targets):
            print(f"  {done}/{len(targets)} processed, {len(failed)} failed")

    print(f"\nDone. {len(targets) - len(failed)}/{len(targets)} re-embedded, {len(failed)} failed.")
    if failed:
        print("Failures (re-run to retry — already-real rows are skipped):")
        for garment_id, detail in failed[:20]:
            print(f"  {garment_id}: {detail}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="classify and report, write nothing")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="best-match cosine below which a vector is treated as noise (default 0.25)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="re-embed these garment ids (or id prefixes) regardless of classification. Needed "
        "after a canonical image is regenerated: the stored vector is a real embedding, just of "
        "the previous (wrong) render, so the mock detector will not flag it.",
    )
    args = parser.parse_args()

    if args.ids:
        async with AsyncSessionLocal() as session:
            all_ids = [
                r[0] for r in (await session.execute(select(Garment.id))).all()
            ]
        targets = [gid for gid in all_ids if any(gid.startswith(p) for p in args.ids)]
        print(f"Re-embedding {len(targets)} garment(s) matched by --ids")
        if args.dry_run:
            print("[dry run] nothing written.")
            return
        if not targets:
            return
        await _run(targets, args.concurrency)
        return

    print("Classifying stored embeddings...")
    mock_ids, real_count, missing = await _classify(args.threshold)
    total = len(mock_ids) + real_count
    print(f"  real embeddings:      {real_count}")
    print(f"  mock/noise vectors:   {len(mock_ids)}" + (f"  ({100*len(mock_ids)/total:.1f}%)" if total else ""))
    print(f"  COMPLETED garments with no embedding row at all: {missing}")

    if not mock_ids:
        print("\nNothing to backfill.")
        return

    targets = mock_ids[: args.limit] if args.limit else mock_ids
    if args.dry_run:
        print(f"\n[dry run] would re-embed {len(targets)} garment(s). Nothing written.")
        return

    await _run(targets, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
