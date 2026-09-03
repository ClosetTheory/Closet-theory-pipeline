"""Schemas for pipeline execution, stages, and requests."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ImageType(str, Enum):
    CATALOG = "CATALOG"
    CROP = "CROP"
    FULL_BODY = "FULL_BODY"


class ClassificationResult(BaseModel):
    image_type: ImageType
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str
    model_version: str


class GarmentRegion(BaseModel):
    label: str  # e.g. "upper_body", "lower_body", "footwear", "outerwear"
    box: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    mask_ref: Optional[str] = Field(default=None, description="Object storage URI to segmentation mask")


class DetectionResult(BaseModel):
    person_detected: bool
    face_box: Optional[List[int]] = Field(default=None, description="[x1, y1, x2, y2]")
    garment_regions: List[GarmentRegion] = Field(default_factory=list)
    model: str
    model_version: str


class DigitisationResult(BaseModel):
    canonical_image_uri: str
    canonical_image_id: Optional[str] = None
    quality_score: float = Field(..., ge=0.0, le=1.0)
    model: str
    model_version: str
    prompt_version: str
    attempts: int = 1


class PipelineStageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PipelineStageRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    garment_id: str
    stage: str
    status: str
    attempt: int
    input_refs: Dict[str, Any]
    output_refs: Dict[str, Any]
    model: Optional[str] = None
    model_version: Optional[str] = None
    algorithm_version: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class PipelineStatusResponse(BaseModel):
    garment_id: str
    current_status: str
    quality_status: str
    stages: List[PipelineStageRunRead]
    can_retry: bool


class RetryRequest(BaseModel):
    stage: Optional[str] = Field(default=None, description="Specific stage to retry, or None to resume from failed stage")
    force: bool = Field(default=False, description="Force re-run even if stage succeeded")


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    OVERRIDE = "OVERRIDE"


class ReviewRequest(BaseModel):
    decision: ReviewDecision
    notes: Optional[str] = None
    attribute_overrides: Optional[Dict[str, Any]] = None
