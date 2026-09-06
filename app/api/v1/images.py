"""Image Asset API endpoints."""

import hashlib
import io
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user, get_db_session, get_storage
from app.config import settings
from app.models.image_asset import ImageAsset
from app.models.user import User
from app.schemas.image import ImageUploadResponse
from app.storage.base import StorageClient

router = APIRouter(prefix="/wardrobe/images", tags=["Images"])

# Enforce decompression bomb defense
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


async def store_uploaded_image(
    file: UploadFile,
    current_user: User,
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

    # 5. Generate secure, non-client key
    ext = "jpg" if "jpeg" in file.content_type else "png"
    storage_key = f"raw/{current_user.tenant_id}/{sha256_hash[:16]}_{uuid.uuid4().hex[:8]}.{ext}"

    # 6. Save bytes to storage
    object_uri = await storage.put_object(storage_key, content, content_type=file.content_type)

    # 7. Persist ImageAsset record
    image_asset = ImageAsset(
        tenant_id=current_user.tenant_id,
        member_id=current_user.member_id,
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
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Securely uploads a raw wardrobe/catalog image.
    Validates MIME type, byte size, decompression limits, generates immutable SHA256 key.
    """
    image_asset = await store_uploaded_image(file, current_user, session, storage)

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
    from fastapi import Response
    try:
        data = await storage.get_object(object_key)
        return Response(content=data, media_type="image/jpeg")
    except Exception:
        raise HTTPException(status_code=404, detail="Media asset not found")


@router.get("/{image_id}/bytes")
async def get_image_asset_bytes(
    image_id: str,
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Streams an ImageAsset by ID directly to the browser."""
    from fastapi import Response
    asset = await session.get(ImageAsset, image_id)
    if not asset:
        raise HTTPException(status_code=404, detail="ImageAsset not found")
    try:
        data = await storage.get_object(asset.object_uri)
        return Response(content=data, media_type=asset.mime_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image bytes could not be retrieved: {e}")

