"""MODA SigLIP Distilled Embedding Provider."""

from typing import List
import httpx
import numpy as np
from app.config import settings
from app.observability import logger
from app.providers.base import BaseEmbeddingProvider
from app.providers.embedding.mock import MockEmbeddingProvider


class SigLIPEmbeddingProvider(BaseEmbeddingProvider):
    """MODA SigLIP Distilled vision transformer embedding model, backed by the
    Hugging Face Inference API (keeps the Docker image light by avoiding a
    local model download)."""

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
        if not settings.HF_API_KEY:
            return await self._fallback.embed(image_bytes)

        try:
            headers = {
                "Authorization": f"Bearer {settings.HF_API_KEY}",
                "Content-Type": "image/jpeg",
            }
            url = f"{settings.HF_INFERENCE_BASE_URL}/{settings.HF_EMBEDDING_MODEL}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, content=image_bytes)
                resp.raise_for_status()
                result = resp.json()

            arr = np.array(result, dtype=np.float64)

            # HF vision feature-extraction models may return a single pooled
            # vector, or a 2D array of per-token embeddings that needs pooling.
            if arr.ndim > 1:
                arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
            arr = arr.flatten()

            norm = np.linalg.norm(arr)
            if norm == 0:
                raise ValueError("HF inference API returned a zero vector")
            arr = arr / norm

            if arr.shape[0] != self.dimension:
                logger.warning(
                    f"SigLIP embedding dimension mismatch: got {arr.shape[0]}, "
                    f"expected {self.dimension}. Truncating/padding to fit."
                )
                if arr.shape[0] > self.dimension:
                    arr = arr[: self.dimension]
                else:
                    arr = np.pad(arr, (0, self.dimension - arr.shape[0]))

            return arr.tolist()
        except Exception as e:
            logger.warning(
                f"Hugging Face SigLIP embedding call failed: {e}. Falling back to mock embedding."
            )
            return await self._fallback.embed(image_bytes)
