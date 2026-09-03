"""Schemas for Compatibility evaluation."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class CompatibilityType(str, Enum):
    LAYERING = "LAYERING"
    STRUCTURAL = "STRUCTURAL"
    VISUAL = "VISUAL"


class CompatibilityDecision(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class CompatibilityItemResult(BaseModel):
    compatibility_type: CompatibilityType
    decision: CompatibilityDecision
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason: str
    algorithm_version: str
    model_version: Optional[str] = None


class CompatibilityEvaluateRequest(BaseModel):
    garment_a_id: str
    garment_b_id: str
    types: Optional[List[CompatibilityType]] = None


class CompatibilityEvaluateResponse(BaseModel):
    garment_a_id: str
    garment_b_id: str
    overall_decision: CompatibilityDecision
    results: List[CompatibilityItemResult]
