"""Mock Semantic Validator: heuristic PASS/NEEDS_REVIEW based on compatibility score."""

from typing import List
from app.providers.base import BaseSemanticValidatorProvider
from app.schemas.styling import GarmentSummary, OutfitCandidate, StylingContext, ValidationResult, ValidationStatus


class MockSemanticValidatorProvider(BaseSemanticValidatorProvider):
    def __init__(self, model_name: str = "mock-semantic-validator", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def validate(
        self,
        context: StylingContext,
        outfit: OutfitCandidate,
        garments: List[GarmentSummary],
    ) -> ValidationResult:
        compatibility = outfit.scores.compatibility
        if compatibility >= 0.6:
            status = ValidationStatus.PASS
            reason = "Outfit garments are structurally and stylistically compatible with the request."
        elif compatibility >= 0.4:
            status = ValidationStatus.NEEDS_REVIEW
            reason = "Compatibility score is borderline; recommend manual review."
        else:
            status = ValidationStatus.FAIL
            reason = "Compatibility score too low for this combination to make semantic sense."

        return ValidationResult(
            status=status,
            confidence=max(0.5, compatibility),
            reason=reason,
            model=self.model_name,
            model_version=self.model_version,
        )
