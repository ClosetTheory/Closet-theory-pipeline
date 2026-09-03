"""Pydantic schemas for Image assets."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class ImageAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    member_id: str
    object_uri: str
    mime_type: str
    width: int
    height: int
    sha256: str
    created_at: datetime


class ImageUploadResponse(BaseModel):
    image_id: str
    object_uri: str
    width: int
    height: int
    mime_type: str
    sha256: str
    created_at: datetime
