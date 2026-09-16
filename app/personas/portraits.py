"""Generating and storing a character's portrait.

Mirrors `persist_generated_image` in app/styling/orchestrator.py: bytes in, an ImageAsset row and
a storage object out. The portrait is stored under the character's own tenant, so it lives
alongside that character's wardrobe rather than in a shared bucket prefix.

Generated once and then left alone. A panel where the same character looks different week to week
would make cross-session comparisons meaningless, so regeneration is always explicit.
"""

import hashlib
import io
import uuid
from typing import Optional, Tuple
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.image_asset import ImageAsset
from app.models.persona import Persona
from app.observability import logger
from app.providers.portrait import build_portrait_prompt, get_portrait_provider
from app.storage.base import StorageClient


async def generate_and_persist_portrait(
    session: AsyncSession,
    storage: StorageClient,
    persona: Persona,
    force: bool = False,
) -> Tuple[Optional[str], str]:
    """Returns (image_asset_id, status). Status is one of: created, skipped, failed.

    `skipped` when the character already has a portrait and `force` is not set — the common case
    on a re-run, and the reason re-seeding is cheap.
    """
    if persona.portrait_image_id and not force:
        return persona.portrait_image_id, "skipped"

    prompt = build_portrait_prompt(persona)
    provider = get_portrait_provider()
    image_bytes = await provider.generate(prompt)
    if not image_bytes:
        logger.warning(f"Portrait generation returned nothing for {persona.slug}")
        return None, "failed"

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    sha256_hash = hashlib.sha256(image_bytes).hexdigest()
    key = f"personas/{persona.user_id}/{sha256_hash[:16]}_{uuid.uuid4().hex[:8]}.jpg"
    object_uri = await storage.put_object(key, image_bytes, content_type="image/jpeg")

    asset = ImageAsset(
        tenant_id=persona.user_id,
        member_id=persona.user_id,
        object_uri=object_uri,
        mime_type="image/jpeg",
        width=width,
        height=height,
        sha256=sha256_hash,
    )
    session.add(asset)
    await session.flush()

    persona.portrait_image_id = asset.id
    # Kept so a portrait can be traced back to the exact wording that produced it — the same
    # reason stage runs record their prompts.
    persona.portrait_prompt = prompt
    return asset.id, "created"
