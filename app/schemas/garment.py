"""Schemas for CanonicalGarment and Garment creation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.attributes import GarmentAttributes


class GarmentCreateRequest(BaseModel):
    source_image_id: str
    # False for the interactive step-by-step demo UI (app/static/index.html), which drives
    # every stage itself via POST /garments/{id}/step — auto-enqueuing a background run too
    # raced the two against each other (confirmed live: a background-worker Stage 1 run and
    # the demo's own manual Stage 1 call both fired within milliseconds of each other on the
    # same garment). True (default) preserves the documented "create and let it process
    # automatically" behavior for any other real caller.
    auto_process: bool = True


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


class CoordinatedGarment(BaseModel):
    """A sibling garment spawned from the SAME source photo — i.e. it was physically worn
    together with this one when the photo was ingested (e.g. a kurta + palazzo pants + dupatta
    all detected in one full-body shot). Not a styling suggestion — a fact about how the item
    was ingested, which the styling pipeline can also use as a strong "these were worn together"
    signal."""

    garment_id: str
    detected_label: Optional[str] = None
    subcategory: Optional[str] = None
    category: Optional[str] = None
    canonical_image_url: Optional[str] = None
    status: str


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
    coordinated_garments: List[CoordinatedGarment] = Field(
        default_factory=list,
        description="Other garments detected in the same source photo (a 'co-ord' set — items ingested as one outfit).",
    )
