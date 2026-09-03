"""MODA SigLIP Distilled Embedding Provider."""

import base64
from typing import List
import httpx
import numpy as np
from app.config import settings
from app.observability import logger
from app.providers.base import BaseEmbeddingProvider
from app.providers.embedding.mock import MockEmbeddingProvider


class SigLIPEmbeddingProvider(BaseEmbeddingProvider):
    """MODA SigLIP Distilled (HopitAI/moda-fashion-distilled) vision transformer
    embedding model, served from a Runpod Serverless endpoint (see runpod/moda_embed.py)."""

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
        if not settings.RUNPOD_API_KEY or not settings.RUNPOD_EMBEDDING_ENDPOINT_ID:
            return await self._fallback.embed(image_bytes)

        try:
            headers = {
                "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            }
            url = f"{settings.RUNPOD_BASE_URL}/{settings.RUNPOD_EMBEDDING_ENDPOINT_ID}/runsync"
            payload = {"input": {"image_b64": base64.b64encode(image_bytes).decode("utf-8")}}
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()

            if result.get("status") != "COMPLETED":
                raise ValueError(f"Runpod job did not complete: {result.get('status')}")

            arr = np.array(result["output"]["embedding"], dtype=np.float64)

            norm = np.linalg.norm(arr)
            if norm == 0:
                raise ValueError("Runpod embedding endpoint returned a zero vector")
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
                f"Runpod SigLIP embedding call failed: {e}. Falling back to mock embedding."
            )
            return await self._fallback.embed(image_bytes)
