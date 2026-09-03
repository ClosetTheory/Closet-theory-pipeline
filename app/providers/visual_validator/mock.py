"""Mock Visual Validator: neutral 0-10 score (no real vision check)."""

from typing import List
from app.providers.base import BaseVisualValidatorProvider
from app.schemas.styling import GarmentSummary, VisualGateResult


class MockVisualValidatorProvider(BaseVisualValidatorProvider):
    def __init__(self, model_name: str = "mock-visual-validator", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def validate_image(self, generated_image: bytes, garments: List[GarmentSummary]) -> VisualGateResult:
        return VisualGateResult(
            score=7.0,
            feedback={"overall_aesthetic": "Mock visual gate: no vision model configured, assuming a reasonable result."},
            model=self.model_name,
            model_version=self.model_version,
        )
