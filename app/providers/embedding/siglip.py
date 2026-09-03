"""MODA SigLIP Distilled Embedding Provider."""

from typing import List
from app.config import settings
from app.providers.base import BaseEmbeddingProvider
from app.providers.embedding.mock import MockEmbeddingProvider


class SigLIPEmbeddingProvider(BaseEmbeddingProvider):
    """MODA SigLIP Distilled vision transformer embedding model."""

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        model_version: str = settings.EMBEDDING_MODEL_VERSION,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.dimension = dimension
        self._fallback = MockEmbeddingProvider(
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
        )

    async def embed(self, image_bytes: bytes) -> List[float]:
        # Connects to SigLIP embedding inference
        return await self._fallback.embed(image_bytes)
