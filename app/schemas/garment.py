"""Schemas for CanonicalGarment and Garment creation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.attributes import GarmentAttributes


class GarmentCreateRequest(BaseModel):
    source_image_id: str


class BulkGarmentUploadResult(BaseModel):
    """Per-file outcome of a bulk ingestion request — either queued for background processing
    or a validation failure, so one bad file doesn't abort the rest of the batch."""

    filename: str
    garment_id: Optional[str] = None
    image_id: Optional[str] = None
    status: str  # "queued" | "error"
    error: Optional[str] = None


class BulkGarmentUploadResponse(BaseModel):
    results: List[BulkGarmentUploadResult]
    queued_count: int
    failed_count: int


class CanonicalGarment(BaseModel):
    """The canonical product representation matching PRD Section 2."""

    model_config = ConfigDict(from_attributes=True)

    garment_id: str
    source_image_refs: List[str]
    image_type: Optional[str] = None
    garment_crop_refs: List[str] = Field(default_factory=list)
    attributes: Optional[Dict[str, Any]] = None
    canonical_image_ref: Optional[str] = None
    image_embedding: Optional[List[float]] = None
    category: Optional[str] = None
    compatibility_features: Dict[str, Any] = Field(default_factory=dict)
    quality_status: str
    provenance: Dict[str, Any] = Field(default_factory=dict)
    pipeline_version: str
