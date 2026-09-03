"""Mock Visual Validator: always PASS (no real vision check)."""

from typing import List
from app.providers.base import BaseVisualValidatorProvider
from app.schemas.styling import GarmentSummary, ValidationResult, ValidationStatus


class MockVisualValidatorProvider(BaseVisualValidatorProvider):
    def __init__(self, model_name: str = "mock-visual-validator", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def validate_image(self, generated_image: bytes, garments: List[GarmentSummary]) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.PASS,
            confidence=0.6,
            reason="Mock visual validator: no vision model configured, assuming generated image matches.",
            model=self.model_name,
            model_version=self.model_version,
        )
