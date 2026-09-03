"""FLUX.2 Image Digitisation Provider."""

from typing import Tuple
from app.config import settings
from app.providers.base import BaseDigitisationProvider
from app.providers.digitisation.mock import MockDigitisationProvider
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import DigitisationResult


class FluxDigitisationProvider(BaseDigitisationProvider):
    """FLUX.2 provider for synthesizing clean canonical garments from crop & attributes."""

    def __init__(
        self,
        model_name: str = settings.DIGITISATION_MODEL_NAME,
        model_version: str = settings.DIGITISATION_MODEL_VERSION,
        prompt_version: str = settings.DIGITISATION_PROMPT_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.prompt_version = prompt_version
        self._fallback = MockDigitisationProvider(
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
        )

    async def digitise(
        self,
        crop_bytes: bytes,
        attributes: GarmentAttributes,
        attempt: int = 1,
    ) -> DigitisationResult:
        # In production, invokes FLUX.2 text-to-image/inpainting API or local inference
        return await self._fallback.digitise(crop_bytes, attributes, attempt=attempt)

    async def validate_digitisation(
        self,
        original_crop_bytes: bytes,
        generated_bytes: bytes,
        attributes: GarmentAttributes,
    ) -> Tuple[bool, float, str]:
        return await self._fallback.validate_digitisation(
            original_crop_bytes,
            generated_bytes,
            attributes,
        )
