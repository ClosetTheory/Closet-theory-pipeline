"""VLM provider factory."""

from app.config import settings
from app.providers.base import BaseVLMProvider
from app.providers.vlm.mock import MockVLMProvider


def get_vlm_provider() -> BaseVLMProvider:
    return MockVLMProvider(
        model_name=settings.VLM_MODEL_NAME,
        model_version=settings.VLM_MODEL_VERSION,
    )
