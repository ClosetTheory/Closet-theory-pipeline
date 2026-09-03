"""Request normalizer provider factory."""

from app.config import settings
from app.providers.base import BaseRequestNormalizerProvider
from app.providers.normalizer.mock import MockRequestNormalizerProvider


def get_request_normalizer_provider() -> BaseRequestNormalizerProvider:
    provider_name = settings.STYLING_NORMALIZER_PROVIDER.lower()
    if provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        from app.providers.vlm.openrouter import OpenRouterGPTProvider

        return OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockRequestNormalizerProvider()
