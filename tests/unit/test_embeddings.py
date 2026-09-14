"""Unit tests for Stage 5: MODA SigLIP Distilled Embeddings."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from app.providers.base import EmbeddingUnavailableError
from app.providers.embedding.mock import MockEmbeddingProvider
from app.providers.embedding.siglip import SigLIPEmbeddingProvider


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


@pytest.mark.asyncio
async def test_siglip_provider_calls_runpod_and_normalizes(sample_catalog_image_bytes, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RUNPOD_EMBEDDING_ENDPOINT_ID", "test-endpoint")

    raw_vector = [1.0] * 768  # unnormalized on purpose, to exercise normalization

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json = lambda: {"status": "COMPLETED", "output": {"embedding": raw_vector}}

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        provider = SigLIPEmbeddingProvider(dimension=768)
        vec = await provider.embed(sample_catalog_image_bytes)

    assert len(vec) == 768
    assert pytest.approx(np.linalg.norm(vec), 0.0001) == 1.0


@pytest.mark.asyncio
async def test_siglip_provider_raises_rather_than_substituting_a_fake_vector(
    sample_catalog_image_bytes, monkeypatch
):
    """A failed embedding call must never return a stand-in vector.

    Falling back to the hash-seeded mock used to look like success, so unusable vectors were
    written to the DB labelled as the real model — dedup and similarity retrieval then ran
    against random numbers with no visible error anywhere.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RUNPOD_EMBEDDING_ENDPOINT_ID", "test-endpoint")

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=RuntimeError("network down"))):
        provider = SigLIPEmbeddingProvider(dimension=768)
        with pytest.raises(EmbeddingUnavailableError):
            await provider.embed(sample_catalog_image_bytes)


@pytest.mark.asyncio
async def test_siglip_provider_raises_when_runpod_is_not_configured(
    sample_catalog_image_bytes, monkeypatch
):
    from app.config import settings

    monkeypatch.setattr(settings, "RUNPOD_API_KEY", None)
    monkeypatch.setattr(settings, "RUNPOD_EMBEDDING_ENDPOINT_ID", None)

    provider = SigLIPEmbeddingProvider(dimension=768)
    with pytest.raises(EmbeddingUnavailableError):
        await provider.embed(sample_catalog_image_bytes)


@pytest.mark.asyncio
async def test_mock_provider_reports_its_own_identity(sample_catalog_image_bytes):
    """A stored embedding must always say which provider really produced it."""
    from app.providers.embedding import get_embedding_provider
    from app.providers.embedding.mock import MOCK_EMBEDDING_MODEL_NAME
    from app.config import settings

    assert MockEmbeddingProvider().model_name == MOCK_EMBEDDING_MODEL_NAME

    original = settings.EMBEDDING_PROVIDER
    try:
        settings.EMBEDDING_PROVIDER = "mock"
        assert get_embedding_provider().model_name == MOCK_EMBEDDING_MODEL_NAME
        assert get_embedding_provider().model_name != settings.EMBEDDING_MODEL_NAME
    finally:
        settings.EMBEDDING_PROVIDER = original
