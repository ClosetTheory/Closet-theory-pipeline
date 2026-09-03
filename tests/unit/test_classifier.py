"""Unit tests for Stage 1 Classifier (MobileNetV3)."""

import pytest
from app.providers.classifier.mock import MockClassifierProvider
from app.schemas.pipeline import ImageType


@pytest.mark.asyncio
async def test_classifier_catalog(sample_catalog_image_bytes):
    provider = MockClassifierProvider()
    result = await provider.classify(sample_catalog_image_bytes)

    assert result.image_type == ImageType.CATALOG
    assert result.confidence >= 0.70
    assert result.model == "MobileNetV3"
    assert result.model_version == "v1"


@pytest.mark.asyncio
async def test_classifier_full_body(sample_full_body_image_bytes):
    provider = MockClassifierProvider()
    result = await provider.classify(sample_full_body_image_bytes)

    assert result.image_type == ImageType.FULL_BODY
    assert result.confidence >= 0.70


@pytest.mark.asyncio
async def test_classifier_low_confidence_forced():
    provider = MockClassifierProvider(confidence=0.45)
    result = await provider.classify(b"fake-bytes")

    assert result.confidence == 0.45
    assert result.confidence < 0.70
