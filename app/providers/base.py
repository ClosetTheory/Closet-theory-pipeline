"""Abstract interfaces for AI/ML model providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import ClassificationResult, DetectionResult, DigitisationResult


class BaseClassifierProvider(ABC):
    """Stage 1: Image Classifier provider interface."""

    @abstractmethod
    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        """Classify input image into CATALOG, CROP, or FULL_BODY."""
        pass


class BaseDetectionProvider(ABC):
    """Stage 2: Facial/Person localization and garment region crop provider interface."""

    @abstractmethod
    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        """Localize faces and extract garment region bounding boxes and masks."""
        pass


class BaseAttributeExtractorProvider(ABC):
    """Stage 3: Visual evidence to canonical structured garment attributes provider interface."""

    @abstractmethod
    async def extract_attributes(self, image_bytes: bytes) -> GarmentAttributes:
        """Extract structured attributes conforming to the 7-step validation pipeline."""
        pass


class BaseDigitisationProvider(ABC):
    """Stage 4: Clean canonical image generation provider interface (FLUX.2)."""

    @abstractmethod
    async def digitise(
        self,
        crop_bytes: bytes,
        attributes: GarmentAttributes,
        attempt: int = 1,
    ) -> DigitisationResult:
        """Generate standardized clean canonical garment representation."""
        pass

    @abstractmethod
    async def validate_digitisation(
        self,
        original_crop_bytes: bytes,
        generated_bytes: bytes,
        attributes: GarmentAttributes,
    ) -> Tuple[bool, float, str]:
        """Validates that generated image preserves garment color, silhouette, and details."""
        pass


class BaseEmbeddingProvider(ABC):
    """Stage 5: Vector representation provider interface (SigLIP Distilled)."""

    @abstractmethod
    async def embed(self, image_bytes: bytes) -> List[float]:
        """Compute normalized unit vector representation for similarity search."""
        pass


class BaseVLMProvider(ABC):
    """Stage 9 Fallback: Vision-Language Model provider for nuanced visual compatibility."""

    @abstractmethod
    async def evaluate_visual_compatibility(
        self,
        image_a_bytes: Optional[bytes],
        image_b_bytes: Optional[bytes],
        attrs_a: Dict[str, Any],
        attrs_b: Dict[str, Any],
    ) -> Tuple[str, float, str]:
        """
        Evaluates visual harmony between two garments when deterministic rules are inconclusive.
        Returns: (decision: 'COMPATIBLE' | 'INCOMPATIBLE' | 'REVIEW_REQUIRED', score: float, reason: str)
        """
        pass
