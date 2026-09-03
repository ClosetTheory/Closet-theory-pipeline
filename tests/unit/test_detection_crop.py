"""Unit tests for Stage 2 Detection & Garment Crop."""

import pytest
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.pipeline.stages.base import StageExecutionContext
from app.pipeline.stages.stage_02_crop import Stage02Crop
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


@pytest.mark.asyncio
async def test_stage_02_crop_skipped_for_catalog(db_session, test_storage, sample_catalog_image_bytes):
    uri = await test_storage.put_object("raw/tenant_1/catalog.jpg", sample_catalog_image_bytes)
    asset = ImageAsset(
        tenant_id="tenant_1",
        member_id="member_1",
        object_uri=uri,
        mime_type="image/jpeg",
        width=600,
        height=600,
        sha256="test_catalog_sha",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)

    garment = Garment(
        tenant_id="tenant_1",
        member_id="member_1",
        source_image_id=asset.id,
        image_type="CATALOG",
        status="CLASSIFIED",
    )
    garment.source_image = asset
    db_session.add(garment)
    await db_session.commit()
    await db_session.refresh(garment)

    ctx = StageExecutionContext(
        session=db_session,
        garment=garment,
        storage=test_storage,
        pipeline_run_id="test_skip_run",
        attempt=1,
    )

    stage = Stage02Crop()
    res = await stage.execute(ctx)

    assert res.status == "SUCCEEDED"
    assert res.output_refs["skipped"] is True
    assert garment.garment_crop_refs == [asset.object_uri]
