"""Mock / Test Provider for FLUX.2 Digitisation with Validation Loop."""

import io
from typing import Optional, Tuple
from PIL import Image, ImageDraw
from app.config import settings
from app.providers.base import BaseDigitisationProvider
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import DigitisationResult


class MockDigitisationProvider(BaseDigitisationProvider):
    """Generates standardized canonical image representations and validates fidelity."""

    def __init__(
        self,
        model_name: str = settings.DIGITISATION_MODEL_NAME,
        model_version: str = settings.DIGITISATION_MODEL_VERSION,
        prompt_version: str = settings.DIGITISATION_PROMPT_VERSION,
        fail_attempts: int = 0,  # simulate validation failures for retry testing
        quality_score: float = 0.92,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.fail_attempts = fail_attempts
        self.quality_score = quality_score
        self.current_attempt_count = 0

    async def digitise(
        self,
        crop_bytes: bytes,
        attributes: GarmentAttributes,
        attempt: int = 1,
        garment_label: Optional[str] = None,
    ) -> DigitisationResult:
        self.current_attempt_count = attempt

        # In testing/local mode: render a clean studio canonical garment representation
        # with solid background and garment attributes labeled
        img = Image.new("RGB", (768, 1024), color=(248, 248, 248))
        draw = ImageDraw.Draw(img)

        # Draw garment silhouette container
        draw.rounded_rectangle(
            [(120, 150), (648, 880)],
            radius=20,
            fill=(230, 235, 240),
            outline=(200, 210, 220),
            width=3,
        )
        draw.text(
            (150, 200),
            f"CANONICAL: {attributes.subcategory.upper()}\n"
            f"Color: {', '.join(attributes.colour)}\n"
            f"Pattern: {attributes.pattern.value}\n"
            f"Silhouette: {attributes.silhouette.value}\n"
            f"Attempt: {attempt}",
            fill=(40, 50, 60),
        )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        generated_bytes = buf.getvalue()

        # In mock, URI will be handled by the stage storing bytes into storage client
        return DigitisationResult(
            canonical_image_uri="",  # Will be populated by storage in Stage 4
            quality_score=self.quality_score if attempt > self.fail_attempts else 0.40,
            model=self.model_name,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            attempts=attempt,
        )

    async def validate_digitisation(
        self,
        original_crop_bytes: bytes,
        generated_bytes: bytes,
        attributes: GarmentAttributes,
        garment_label: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        if self.current_attempt_count <= self.fail_attempts:
            return (
                False,
                0.40,
                f"Attempt {self.current_attempt_count} failed quality check: color drift detected.",
            )
        return (
            True,
            self.quality_score,
            "Validation successful: high silhouette alignment and color fidelity.",
        )
