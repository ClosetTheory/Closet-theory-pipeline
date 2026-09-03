"""Embedding provider factory."""

from app.config import settings
from app.providers.base import BaseEmbeddingProvider
from app.providers.embedding.mock import MockEmbeddingProvider
from app.providers.embedding.siglip import SigLIPEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    provider_name = settings.EMBEDDING_PROVIDER.lower()
    if provider_name in ("siglip", "huggingface"):
        return SigLIPEmbeddingProvider(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_version=settings.EMBEDDING_MODEL_VERSION,
            dimension=settings.EMBEDDING_DIMENSION,
        )
    return MockEmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_version=settings.EMBEDDING_MODEL_VERSION,
        dimension=settings.EMBEDDING_DIMENSION,
    )
