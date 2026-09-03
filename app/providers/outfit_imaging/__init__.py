"""Outfit image provider factory."""

from app.config import settings
from app.providers.base import BaseOutfitImageProvider
from app.providers.outfit_imaging.mock import MockOutfitImageProvider


def get_outfit_image_provider() -> BaseOutfitImageProvider:
    provider_name = settings.STYLING_OUTFIT_IMAGE_PROVIDER.lower()
    if provider_name == "gpt" and settings.OPENROUTER_API_KEY:
        from app.providers.outfit_imaging.gpt_outfit_provider import GPTOutfitImageProvider

        return GPTOutfitImageProvider(api_key=settings.OPENROUTER_API_KEY)
    return MockOutfitImageProvider()
