"""Mock / Deterministic Provider for MODA SigLIP Distilled Embeddings."""

import hashlib
from typing import List
import numpy as np
from app.config import settings
from app.providers.base import BaseEmbeddingProvider


MOCK_EMBEDDING_MODEL_NAME = "mock-hash-embedding"


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Generates deterministic, normalized unit embeddings (768-dim) for garments.
    Guarantees ||v|| = 1.0 so that cosine_similarity(a, b) == dot(a, b).

    These vectors are hash-seeded Gaussians: identical bytes give an identical vector, but any
    two different images are orthogonal (cosine ~0) no matter how visually alike they are. They
    carry no perceptual meaning, so nothing downstream (dedup, similarity retrieval) can work on
    them. Reports its own model name rather than the configured real-model label precisely so a
    stored vector always says which provider actually produced it.
    """

    def __init__(
        self,
        model_name: str = MOCK_EMBEDDING_MODEL_NAME,
        model_version: str = settings.EMBEDDING_MODEL_VERSION,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.dimension = dimension

    async def embed(self, image_bytes: bytes) -> List[float]:
        # Generate pseudo-random deterministic vector from image SHA256 hash
        digest = hashlib.sha256(image_bytes).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)

        # Sample Gaussian vector
        raw_vec = rng.standard_normal(self.dimension)

        # L2-normalize to unit length: ||v|| = 1.0
        norm = np.linalg.norm(raw_vec)
        if norm == 0:
            raw_vec[0] = 1.0
            norm = 1.0
        normalized_vec = raw_vec / norm

        return normalized_vec.tolist()
