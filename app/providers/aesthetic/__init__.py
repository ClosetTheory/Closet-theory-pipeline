"""Aesthetic provider factory."""

from app.config import settings
from app.providers.base import BaseAestheticProvider
from app.providers.aesthetic.mock import MockAestheticProvider


def get_aesthetic_provider() -> BaseAestheticProvider:
    provider_name = settings.STYLING_AESTHETIC_PROVIDER.lower()
    if provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        from app.providers.aesthetic.openrouter import OpenRouterAestheticProvider

        return OpenRouterAestheticProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockAestheticProvider()
