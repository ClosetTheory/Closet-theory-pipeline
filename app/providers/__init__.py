"""Providers package exports."""

from app.providers.base import (
    BaseClassifierProvider,
    BaseDetectionProvider,
    BaseAttributeExtractorProvider,
    BaseDigitisationProvider,
    BaseEmbeddingProvider,
    BaseVLMProvider,
)
from app.providers.classifier import get_classifier_provider
from app.providers.detection import get_detection_provider
from app.providers.attributes import get_attribute_provider
from app.providers.digitisation import get_digitisation_provider
from app.providers.embedding import get_embedding_provider
from app.providers.vlm import get_vlm_provider

__all__ = [
    "BaseClassifierProvider",
    "BaseDetectionProvider",
    "BaseAttributeExtractorProvider",
    "BaseDigitisationProvider",
    "BaseEmbeddingProvider",
    "BaseVLMProvider",
    "get_classifier_provider",
    "get_detection_provider",
    "get_attribute_provider",
    "get_digitisation_provider",
    "get_embedding_provider",
    "get_vlm_provider",
]
