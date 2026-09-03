"""Semantic validator provider factory."""

from app.config import settings
from app.providers.base import BaseSemanticValidatorProvider
from app.providers.semantic_validator.mock import MockSemanticValidatorProvider


def get_semantic_validator_provider() -> BaseSemanticValidatorProvider:
    provider_name = settings.STYLING_VALIDATOR_PROVIDER.lower()
    if provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        from app.providers.vlm.openrouter import OpenRouterGPTProvider

        return OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockSemanticValidatorProvider()
