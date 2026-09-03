"""Unit tests for Stage 2 Detection & Garment Crop."""

import pytest
from app.providers.detection.mock import MockDetectionProvider


@pytest.mark.asyncio
async def test_detection_person_present(sample_full_body_image_bytes):
    provider = MockDetectionProvider(person_detected=True)
    result = await provider.detect_and_crop(sample_full_body_image_bytes)

    assert result.person_detected is True
    assert result.face_box is not None
    assert len(result.face_box) == 4
    assert len(result.garment_regions) >= 1

    for reg in result.garment_regions:
        assert reg.label in ("upper_body", "lower_body")
        assert len(reg.box) == 4
        assert reg.box[2] > reg.box[0]
        assert reg.box[3] > reg.box[1]


@pytest.mark.asyncio
async def test_detection_no_person(sample_catalog_image_bytes):
    provider = MockDetectionProvider(person_detected=False)
    result = await provider.detect_and_crop(sample_catalog_image_bytes)

    assert result.person_detected is False
    assert result.face_box is None
    assert len(result.garment_regions) == 1
    assert result.garment_regions[0].label == "upper_body"
