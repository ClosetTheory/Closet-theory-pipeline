"""Unit tests for Stage 4: FLUX.2 Digitisation with Validation & Retry Loop."""

import pytest
from app.config import settings
from app.providers.digitisation.mock import MockDigitisationProvider
from app.schemas.attributes import validate_extracted_attributes


@pytest.mark.asyncio
async def test_digitisation_success_first_attempt(sample_catalog_image_bytes, valid_attributes_dict):
    provider = MockDigitisationProvider(fail_attempts=0, quality_score=0.90)
    attrs = validate_extracted_attributes(valid_attributes_dict)

    res = await provider.digitise(sample_catalog_image_bytes, attrs, attempt=1)
    is_valid, score, _ = await provider.validate_digitisation(sample_catalog_image_bytes, b"fake", attrs)

    assert is_valid is True
    assert res.model in ("FLUX.2", "GPT-4o-Studio", "black-forest-labs/flux.2-pro", settings.DIGITISATION_MODEL_NAME)


@pytest.mark.asyncio
async def test_digitisation_retry_succeeds_on_second_attempt(sample_catalog_image_bytes, valid_attributes_dict):
    provider = MockDigitisationProvider(fail_attempts=1, quality_score=0.88)
    attrs = validate_extracted_attributes(valid_attributes_dict)

    # Attempt 1: should fail
    await provider.digitise(sample_catalog_image_bytes, attrs, attempt=1)
    is_valid_1, score_1, reason_1 = await provider.validate_digitisation(sample_catalog_image_bytes, b"fake", attrs)
    assert is_valid_1 is False
    assert score_1 < 0.75

    # Attempt 2: should succeed
    await provider.digitise(sample_catalog_image_bytes, attrs, attempt=2)
    is_valid_2, score_2, _ = await provider.validate_digitisation(sample_catalog_image_bytes, b"fake", attrs)
    assert is_valid_2 is True
    assert score_2 >= 0.75
