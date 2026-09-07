"""VLM provider factory."""

from app.config import settings
from app.providers.base import BaseVLMProvider
from app.providers.vlm.mock import MockVLMProvider


def get_vlm_provider() -> BaseVLMProvider:
    if settings.OPENROUTER_API_KEY:
        from app.providers.vlm.openrouter import OpenRouterGPTProvider

        return OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockVLMProvider(
        model_name=settings.VLM_MODEL_NAME,
        model_version=settings.VLM_MODEL_VERSION,
    )
