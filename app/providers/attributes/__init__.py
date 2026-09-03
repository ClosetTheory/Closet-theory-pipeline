"""Attribute extractor provider factory."""

from app.config import settings
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.claude import ClaudeAttributeExtractorProvider
from app.providers.attributes.gemini import GeminiAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.providers.attributes.moda_ner import ModaNerAttributeExtractorProvider


def get_attribute_provider() -> BaseAttributeExtractorProvider:
    provider_name = settings.ATTRIBUTE_PROVIDER.lower()
    if provider_name == "gemini":
        return GeminiAttributeExtractorProvider(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.ATTRIBUTE_MODEL_NAME,
            model_version=settings.ATTRIBUTE_MODEL_VERSION,
        )
    elif provider_name == "claude":
        return ClaudeAttributeExtractorProvider(
            api_key=settings.ANTHROPIC_API_KEY,
            model_name=settings.ATTRIBUTE_MODEL_NAME,
            model_version=settings.ATTRIBUTE_MODEL_VERSION,
        )
    elif provider_name == "moda_ner":
        return ModaNerAttributeExtractorProvider(
            model_name=settings.ATTRIBUTE_MODEL_NAME,
            model_version=settings.ATTRIBUTE_MODEL_VERSION,
        )
    return MockAttributeExtractorProvider(
        model_name=settings.ATTRIBUTE_MODEL_NAME,
        model_version=settings.ATTRIBUTE_MODEL_VERSION,
    )
