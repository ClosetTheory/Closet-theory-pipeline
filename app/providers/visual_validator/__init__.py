"""Visual validator provider factory."""

from app.config import settings
from app.providers.base import BaseVisualValidatorProvider
from app.providers.visual_validator.mock import MockVisualValidatorProvider


def get_visual_validator_provider() -> BaseVisualValidatorProvider:
    provider_name = settings.STYLING_VISUAL_VALIDATOR_PROVIDER.lower()
    if provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        from app.providers.vlm.openrouter import OpenRouterGPTProvider

        return OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockVisualValidatorProvider()
