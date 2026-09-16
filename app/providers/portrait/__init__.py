"""Portrait provider factory — same shape as the outfit-imaging factory next door."""

from app.config import settings
from app.providers.base import BasePortraitProvider
from app.providers.portrait.mock import MockPortraitProvider
from app.providers.portrait.prompt import MONK_HEX, build_portrait_prompt

__all__ = ["get_portrait_provider", "build_portrait_prompt", "MONK_HEX"]


def get_portrait_provider() -> BasePortraitProvider:
    """Falls back to the mock without an API key, so seeding and CI work offline."""
    if settings.PORTRAIT_PROVIDER.lower() == "gpt" and settings.OPENROUTER_API_KEY:
        from app.providers.portrait.gpt_portrait_provider import GPTPortraitProvider

        return GPTPortraitProvider(api_key=settings.OPENROUTER_API_KEY)
    return MockPortraitProvider()
