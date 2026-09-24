"""Image Asset API endpoints."""

import asyncio
import hashlib
import io
import uuid
from typing import Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import ActingScope, get_acting_scope, get_current_user, get_db_session, get_storage
from app.config import settings
from app.models.image_asset import ImageAsset
from app.models.user import User
from app.observability import logger
from app.schemas.image import ImageUploadResponse
from app.storage.base import StorageClient

router = APIRouter(prefix="/wardrobe/images", tags=["Images"])

# Image assets are content-addressed (the object key and the ImageAsset row both carry the
# SHA-256 of the bytes), so a given asset id / object key never changes what it returns. That
# makes long browser caching safe and is what stops the catalogue re-fetching hundreds of
# thumbnails on every visit.
IMMUTABLE_CACHE_HEADERS: Dict[str, str] = {"Cache-Control": "public, max-age=604800, immutable"}

# Bounds for the ?thumb= edge length. Anything above the upper bound is not a thumbnail and
# would just be a slower copy of the original.
THUMB_MIN_EDGE = 16
THUMB_MAX_EDGE = 1024


def thumbnail_object_key(sha256: str, edge: int) -> str:
    """Storage key for the cached thumbnail of a content-addressed asset."""
    return f"thumbs/{sha256}_{edge}.jpg"


def build_thumbnail(data: bytes, edge: int) -> bytes:
    """Decode, downscale to fit `edge` x `edge`, re-encode as JPEG q80. Pure CPU; call via
    asyncio.to_thread from request handlers so it never blocks the event loop."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

# Enforce decompression bomb defense
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


async def store_uploaded_image(
    file: UploadFile,
    scope: ActingScope,
    session: AsyncSession,
    storage: StorageClient,
) -> ImageAsset:
    """
    Securely validates and stores a raw wardrobe/catalog image, shared by both the single-image
    (`upload_image`) and bulk (`POST /garments/bulk`) upload paths so there's exactly one place
    that enforces MIME type, byte size, and decompression-bomb limits.
    """
    # 1. MIME type validation
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. Allowed: {settings.ALLOWED_MIME_TYPES}",
        )

    # 2. File size limit
    content = await file.read()
    if len(content) > settings.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_BYTES / (1024*1024)}MB",
        )

    # 3. Pillow decompression and integrity validation
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
            width, height = img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or malformed image file: {str(e)}",
        )

    # 4. Compute SHA-256
    sha256_hash = hashlib.sha256(content).hexdigest()

    # Note: no exact-duplicate (SHA256 collision) check here — deliberately removed. A garment
    # that failed partway through the pipeline needs to be re-ingestible from the exact same
    # photo, and blocking on a byte-identical hash match made that impossible (the ImageAsset
    # row for a failed garment's source photo persists even after the garment itself is
    # deleted/retried). Perceptual near-duplicate detection (a different photo of the same
    # physical garment) still runs in Stage 5 via embedding similarity and only flags
    # REVIEW_REQUIRED rather than blocking — see app/pipeline/stages/stage_05_embed.py.

    # 5. Generate secure, non-client key
    ext = "jpg" if "jpeg" in file.content_type else "png"
    storage_key = f"raw/{scope.tenant_id}/{sha256_hash[:16]}_{uuid.uuid4().hex[:8]}.{ext}"

    # 6. Save bytes to storage
    object_uri = await storage.put_object(storage_key, content, content_type=file.content_type)

    # 7. Persist ImageAsset record
    image_asset = ImageAsset(
        tenant_id=scope.tenant_id,
        member_id=scope.member_id,
        object_uri=object_uri,
        mime_type=file.content_type,
        width=width,
        height=height,
        sha256=sha256_hash,
    )
    session.add(image_asset)
    await session.commit()
    await session.refresh(image_asset)
    return image_asset


@router.post("", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Securely uploads a raw wardrobe/catalog image.
    Validates MIME type, byte size, decompression limits, generates immutable SHA256 key.
    """
    image_asset = await store_uploaded_image(file, scope, session, storage)

    return ImageUploadResponse(
        image_id=image_asset.id,
        object_uri=image_asset.object_uri,
        width=image_asset.width,
        height=image_asset.height,
        mime_type=image_asset.mime_type,
        sha256=image_asset.sha256,
        created_at=image_asset.created_at,
    )


@router.get("/media/{object_key:path}")
async def get_media_bytes(
    object_key: str,
    storage: StorageClient = Depends(get_storage),
):
    """Streams stored image asset bytes directly for browser visual presentation."""
    try:
        data = await storage.get_object(object_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Media asset not found")
    return Response(content=data, media_type="image/jpeg", headers=IMMUTABLE_CACHE_HEADERS)


@router.get("/{image_id}/bytes")
async def get_image_asset_bytes(
    image_id: str,
    thumb: Optional[int] = None,
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Streams an ImageAsset by ID directly to the browser.

    `thumb=<max_dimension>` returns a resized, more heavily compressed JPEG instead of the
    original — canonical studio images are stored at their full generated resolution (typically
    1000-1450px, up to ~1.6MB each), which is unnecessarily heavy for a ~300px-wide catalogue
    grid cell.

    The catalogue asks for every garment in one go (`?limit=3000`, each cell `?thumb=320`), so
    this endpoint is the hottest path on the box by a wide margin. Two things keep it from
    taking the whole API down with it, which is exactly what happened in production (the
    trivial `/health` route hanging 20s+ while a catalogue page loaded):

    1. The thumbnail is built once and cached back into object storage under a
       content-addressed key, so the 1.6MB decode + LANCZOS resize happens one time per
       asset, not once per page view per visitor.
    2. The build itself runs in a worker thread (`asyncio.to_thread`), so even a cache miss
       never blocks the event loop the other requests are waiting on.
    """
    asset = await session.get(ImageAsset, image_id)
    if not asset:
        raise HTTPException(status_code=404, detail="ImageAsset not found")

    if thumb and asset.mime_type.startswith("image/"):
        edge = max(THUMB_MIN_EDGE, min(int(thumb), THUMB_MAX_EDGE))
        thumb_key = thumbnail_object_key(asset.sha256, edge)

        try:
            if await storage.exists(thumb_key):
                cached = await storage.get_object(thumb_key)
                return Response(content=cached, media_type="image/jpeg", headers=IMMUTABLE_CACHE_HEADERS)
        except Exception as e:  # a cache-read failure must never break image serving
            logger.warning(f"Thumbnail cache read failed for {thumb_key}: {e}")

        try:
            data = await storage.get_object(asset.object_uri)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Image bytes could not be retrieved: {e}")

        try:
            thumb_bytes = await asyncio.to_thread(build_thumbnail, data, edge)
        except Exception as e:
            # Serve the original rather than a hard failure, same as before.
            logger.warning(f"Thumbnail build failed for {image_id}: {e}")
            return Response(content=data, media_type=asset.mime_type, headers=IMMUTABLE_CACHE_HEADERS)

        try:
            await storage.put_object(thumb_key, thumb_bytes, content_type="image/jpeg")
        except Exception as e:
            logger.warning(f"Thumbnail cache write failed for {thumb_key}: {e}")
        return Response(content=thumb_bytes, media_type="image/jpeg", headers=IMMUTABLE_CACHE_HEADERS)

    try:
        data = await storage.get_object(asset.object_uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image bytes could not be retrieved: {e}")
    return Response(content=data, media_type=asset.mime_type, headers=IMMUTABLE_CACHE_HEADERS)

