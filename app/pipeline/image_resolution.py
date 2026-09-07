"""Shared "which image best represents this garment" resolution.

Priority: canonical (Stage 4's clean digitised studio shot) > first crop ref > raw source
photo. Used wherever a real image of a garment is needed outside Stage 5 itself (e.g. a
vision-model visual-compatibility check) — factored out here so this priority order is defined
in exactly one place rather than re-derived at each call site.
"""

from typing import Optional
from app.models.garment import Garment
from app.storage.base import StorageClient


def resolve_garment_image_uri(garment: Garment) -> Optional[str]:
    """Returns the storage URI for the best available image of this garment, or None if
    nothing is available yet (e.g. Stage 2 hasn't run). Relies on `canonical_image` and
    `source_image` being loaded (both are `lazy="joined"` on the Garment model, so this is
    safe to call without an active session on an already-fetched Garment)."""
    if garment.canonical_image_id and garment.canonical_image:
        return garment.canonical_image.object_uri
    if garment.garment_crop_refs:
        return garment.garment_crop_refs[0]
    if garment.source_image:
        return garment.source_image.object_uri
    return None


async def resolve_garment_image_bytes(storage: StorageClient, garment: Garment) -> Optional[bytes]:
    """Fetches the actual bytes for resolve_garment_image_uri()'s result, or None if there's
    nothing to fetch yet or the fetch itself fails (missing/evicted object) — callers must
    treat None as "no image available," not raise on it, since a garment mid-pipeline
    (no canonical image yet) or a storage hiccup are both real, expected states."""
    image_uri = resolve_garment_image_uri(garment)
    if not image_uri:
        return None
    try:
        return await storage.get_object(image_uri)
    except Exception:
        return None
