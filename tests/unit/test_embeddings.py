"""Unit tests for Stage 5: MODA SigLIP Distilled Embeddings."""

import numpy as np
import pytest
from app.providers.embedding.mock import MockEmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_dimension_and_normalization(sample_catalog_image_bytes):
    provider = MockEmbeddingProvider(dimension=768)
    vec = await provider.embed(sample_catalog_image_bytes)

    assert len(vec) == 768
    # Test L2 norm is exactly 1.0
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 0.0001) == 1.0


@pytest.mark.asyncio
async def test_cosine_similarity_via_dot_product(sample_catalog_image_bytes, sample_full_body_image_bytes):
    provider = MockEmbeddingProvider()
    vec_a = np.array(await provider.embed(sample_catalog_image_bytes))
    vec_b = np.array(await provider.embed(sample_full_body_image_bytes))

    # Dot product of normalized vectors equals cosine similarity
    dot_product = float(np.dot(vec_a, vec_b))
    assert -1.0 <= dot_product <= 1.0

    # Self-similarity must be 1.0
    self_similarity = float(np.dot(vec_a, vec_a))
    assert pytest.approx(self_similarity, 0.0001) == 1.0
