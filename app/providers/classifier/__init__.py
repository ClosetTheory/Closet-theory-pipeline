"""Classifier provider factory."""

from app.config import settings
from app.providers.base import BaseClassifierProvider
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
    # "mock" (or any other value): the local face/aspect-ratio heuristic — there is no real
    # trained classifier model in this codebase; a previous "mobilenet" option here claimed to
    # run PyTorch MobileNetV3 inference but silently delegated to this exact same heuristic
    # while reporting a fake model name. Removed rather than fixed, since OpenRouter (above) is
    # the only classifier that has ever been real.
    return MockClassifierProvider(
        model_name=settings.CLASSIFIER_MODEL_NAME,
        model_version=settings.CLASSIFIER_MODEL_VERSION,
    )
