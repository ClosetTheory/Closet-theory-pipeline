"""Mock Attribute Extractor Provider."""

from typing import Any, Dict, Optional
from app.config import settings
from app.providers.base import BaseAttributeExtractorProvider
from app.schemas.attributes import (
    GarmentAttributes,
    PatternEnum,
    FitEnum,
    SilhouetteEnum,
    SleeveLengthEnum,
    OccasionEnum,
    SeasonEnum,
    LayeringRoleEnum,
    validate_extracted_attributes,
)


class MockAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """Provides validated GarmentAttributes for testing and local runs."""

    def __init__(
        self,
        model_name: str = settings.ATTRIBUTE_MODEL_NAME,
        model_version: str = settings.ATTRIBUTE_MODEL_VERSION,
        preset_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.preset_attributes = preset_attributes

    async def extract_attributes(self, image_bytes: bytes) -> GarmentAttributes:
        if self.preset_attributes is not None:
            return validate_extracted_attributes(self.preset_attributes)

        # Canonical default sample: Oxford Shirt
        sample_dict = {
            "category": "shirt",
            "subcategory": "oxford_shirt",
            "colour": ["white"],
            "pattern": "solid",
            "material": "cotton",
            "fit": "regular",
            "silhouette": "straight",
            "sleeve_length": "long",
            "occasion": ["smart_casual", "work"],
            "season": ["summer", "spring"],
            "layering_role": "base",
            "warmth": 0.25,
            "versatility": 0.85,
            "confidence": 0.96,
        }
        return validate_extracted_attributes(sample_dict)
