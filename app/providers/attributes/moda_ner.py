"""MODA_NER Attribute Extractor Provider."""

from app.config import settings
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.schemas.attributes import GarmentAttributes


class ModaNerAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """MODA fashion named entity and attribute recognition provider."""

    def __init__(
        self,
        model_name: str = settings.ATTRIBUTE_MODEL_NAME,
        model_version: str = settings.ATTRIBUTE_MODEL_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = MockAttributeExtractorProvider(model_name=model_name, model_version=model_version)

    async def extract_attributes(self, image_bytes: bytes) -> GarmentAttributes:
        return await self._fallback.extract_attributes(image_bytes)
