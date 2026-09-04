"""Classifier provider factory."""

from app.config import settings
from app.providers.base import BaseClassifierProvider
from app.providers.classifier.mobilenet import MobileNetV3ClassifierProvider
from app.providers.classifier.mock import MockClassifierProvider


def get_classifier_provider() -> BaseClassifierProvider:
    provider_name = settings.CLASSIFIER_PROVIDER.lower()
    if provider_name == "openrouter":
        from app.providers.vlm.openrouter import OpenRouterGPTProvider

        return OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    if provider_name == "mobilenet":
        return MobileNetV3ClassifierProvider(
            model_name=settings.CLASSIFIER_MODEL_NAME,
            model_version=settings.CLASSIFIER_MODEL_VERSION,
        )
    return MockClassifierProvider(
        model_name=settings.CLASSIFIER_MODEL_NAME,
        model_version=settings.CLASSIFIER_MODEL_VERSION,
    )
