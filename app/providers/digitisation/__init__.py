"""Digitisation provider factory."""

from app.config import settings
from app.providers.base import BaseDigitisationProvider
from app.providers.digitisation.flux import FluxDigitisationProvider
from app.providers.digitisation.gpt_digitiser import GPTStudioDigitisationProvider
from app.providers.digitisation.mock import MockDigitisationProvider


def get_digitisation_provider() -> BaseDigitisationProvider:
    provider_name = settings.DIGITISATION_PROVIDER.lower()
    if provider_name == "gpt":
        return GPTStudioDigitisationProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.DIGITISATION_MODEL_NAME,
            model_version=settings.DIGITISATION_MODEL_VERSION,
            prompt_version=settings.DIGITISATION_PROMPT_VERSION,
        )
    elif provider_name == "flux":
        return FluxDigitisationProvider(
            model_name=settings.DIGITISATION_MODEL_NAME,
            model_version=settings.DIGITISATION_MODEL_VERSION,
            prompt_version=settings.DIGITISATION_PROMPT_VERSION,
        )
    return MockDigitisationProvider(
        model_name=settings.DIGITISATION_MODEL_NAME,
        model_version=settings.DIGITISATION_MODEL_VERSION,
        prompt_version=settings.DIGITISATION_PROMPT_VERSION,
    )
