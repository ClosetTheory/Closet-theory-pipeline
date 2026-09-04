"""Abstract interfaces for AI/ML model providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import ClassificationResult, DetectionResult, DigitisationResult
from app.schemas.styling import (
    GarmentSummary,
    OutfitCandidate,
    SemanticGateResult,
    StylingContext,
    StylingIntent,
    ValidationResult,
    VisualGateResult,
)


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
    async def extract_attributes(
        self, image_bytes: bytes, image_type: Optional[str] = None, garment_label: Optional[str] = None
    ) -> GarmentAttributes:
        """Extract structured attributes conforming to the 7-step validation pipeline.

        image_type: one of "CATALOG" | "CROP" | "FULL_BODY" (from Stage 1 classification),
        when available. VLM-based providers infer framing from the image itself and may
        ignore it; MODA_NER uses it to select the matching track.

        garment_label: which garment (of possibly several visible in image_bytes) to focus
        on, e.g. "outerwear" or "the white tank top worn underneath". None when image_bytes
        already shows a single isolated garment (catalog/crop images). Providers that can't
        be steered per-garment may ignore it.
        """
        pass


class BaseDigitisationProvider(ABC):
    """Stage 4: Clean canonical image generation provider interface (FLUX.2)."""

    @abstractmethod
    async def digitise(
        self,
        crop_bytes: bytes,
        attributes: GarmentAttributes,
        attempt: int = 1,
        garment_label: Optional[str] = None,
    ) -> DigitisationResult:
        """Generate standardized clean canonical garment representation.

        garment_label: which garment (of possibly several visible in crop_bytes) to isolate
        and reproduce, e.g. "outerwear". None when crop_bytes already shows a single isolated
        garment. Providers that can't be steered per-garment may ignore it.
        """
        pass

    @abstractmethod
    async def validate_digitisation(
        self,
        original_crop_bytes: bytes,
        generated_bytes: bytes,
        attributes: GarmentAttributes,
        garment_label: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """Validates that generated image preserves garment color, silhouette, and details.

        garment_label: which garment (of possibly several visible in original_crop_bytes) is
        being validated. None when original_crop_bytes already shows a single isolated garment.
        """
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


class BaseRequestNormalizerProvider(ABC):
    """Styling Stage 1: Natural-language styling request -> structured StylingIntent."""

    @abstractmethod
    async def normalize(self, request_text: str, anchor_categories: List[str]) -> StylingIntent:
        """Translates free-text into a validated StylingIntent. Must never invent garments/IDs."""
        pass


class BaseSemanticValidatorProvider(ABC):
    """Styling Stage 8: Checks whether a composed outfit makes semantic sense for the request."""

    @abstractmethod
    async def validate(
        self,
        context: StylingContext,
        outfit: OutfitCandidate,
        garments: List[GarmentSummary],
    ) -> ValidationResult:
        """Returns PASS | FAIL | NEEDS_REVIEW. Must not invent/replace garments or mutate inventory."""
        pass

    @abstractmethod
    async def validate_generated(
        self,
        context: StylingContext,
        outfit: OutfitCandidate,
        garments: List[GarmentSummary],
        generated_image: bytes,
    ) -> SemanticGateResult:
        """
        SPEC.md Section 35 Semantic Gate: validates the GENERATED image against the
        original request/context/selected garments (binary PASS/FAIL + violations).
        Flags generation drift (e.g. a different garment rendered than was selected).
        """
        pass


class BaseOutfitImageProvider(ABC):
    """Styling Stage 9: Composite outfit image generation from selected canonical garments."""

    @abstractmethod
    async def generate(
        self,
        garments: List[GarmentSummary],
        canonical_images: List[bytes],
    ) -> Optional[bytes]:
        """Generates a single presentation image for the outfit, or None on failure."""
        pass


class BaseVisualValidatorProvider(ABC):
    """Styling Stage 10: SPEC.md Section 34 Visual Gate — evaluates the generated outfit image."""

    @abstractmethod
    async def validate_image(
        self,
        generated_image: bytes,
        garments: List[GarmentSummary],
    ) -> VisualGateResult:
        """Returns a 0-10 quality score + structured feedback. A quality score, not the styling decision."""
        pass
