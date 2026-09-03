"""BaseStage abstraction for Image Ingestion Pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.garment import Garment
from app.storage.base import StorageClient


@dataclass
class StageExecutionContext:
    session: AsyncSession
    garment: Garment
    storage: StorageClient
    pipeline_run_id: str
    attempt: int = 1
    force: bool = False
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageExecutionResult:
    status: str  # "SUCCEEDED", "FAILED", "RETRYABLE", "REVIEW_REQUIRED"
    input_refs: Dict[str, Any]
    output_refs: Dict[str, Any]
    input_hash: str
    model: Optional[str] = None
    model_version: Optional[str] = None
    algorithm_version: Optional[str] = None
    error: Optional[str] = None
    quality_status: Optional[str] = None


class BaseStage(ABC):
    """Abstract base class for each single-responsibility pipeline stage."""

    stage_name: str

    @abstractmethod
    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        """Executes stage logic. Must be idempotent and side-effect safe."""
        pass
