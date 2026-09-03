"""Integration tests for PipelineOrchestrator."""

import pytest
from sqlalchemy import select
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.pipeline_stage import PipelineStageRun
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.state_machine import GarmentState, PipelineStage


@pytest.mark.asyncio
async def test_full_pipeline_run_to_completion(
    db_session,
    test_storage,
    sample_catalog_image_bytes,
):
    # 1. Store raw image asset
    uri = await test_storage.put_object("raw/tenant_1/test_item.jpg", sample_catalog_image_bytes)
    raw_asset = ImageAsset(
        tenant_id="tenant_1",
        member_id="member_1",
        object_uri=uri,
        mime_type="image/jpeg",
        width=800,
        height=800,
        sha256="fake_sha",
    )
    db_session.add(raw_asset)
    await db_session.commit()
    await db_session.refresh(raw_asset)

    # 2. Create Garment entity
    garment = Garment(
        tenant_id="tenant_1",
        member_id="member_1",
        source_image_id=raw_asset.id,
        status=GarmentState.RECEIVED.value,
    )
    db_session.add(garment)
    await db_session.commit()
    await db_session.refresh(garment)

    # 3. Execute Orchestrator
    orchestrator = PipelineOrchestrator(session=db_session, storage=test_storage)
    completed_garment = await orchestrator.run(garment)

    # 4. Assertions on Garment
    assert completed_garment.status == GarmentState.COMPLETED.value
    assert completed_garment.quality_status == "APPROVED"
    assert completed_garment.image_type is not None
    assert len(completed_garment.garment_crop_refs) > 0
    assert completed_garment.subcategory in ("oxford_shirt", "blazer", "shirt")
    assert completed_garment.category in ("TOP", "OUTERWEAR")
    assert completed_garment.canonical_image_id is not None
    assert "layering" in completed_garment.compatibility_features
    assert "structure" in completed_garment.compatibility_features
    assert "visual" in completed_garment.compatibility_features

    # 5. Assertions on PipelineStageRun audit records
    stage_runs_res = await db_session.execute(
        select(PipelineStageRun).where(PipelineStageRun.garment_id == garment.id)
    )
    stage_runs = stage_runs_res.scalars().all()
    assert len(stage_runs) == 9
    for r in stage_runs:
        assert r.status == "SUCCEEDED"
        assert r.duration_ms is not None
        assert r.duration_ms >= 0

    # 6. Assertions on GarmentEmbedding
    emb_res = await db_session.execute(
        select(GarmentEmbedding).where(GarmentEmbedding.garment_id == garment.id)
    )
    embedding_record = emb_res.scalars().first()
    assert embedding_record is not None
    assert embedding_record.dimension == 768
    assert len(embedding_record.embedding) == 768

    # 7. Test Idempotency: Second run without force should reuse all stages
    second_run_garment = await orchestrator.run(completed_garment, force=False)
    assert second_run_garment.status == GarmentState.COMPLETED.value
    all_runs_res = await db_session.execute(
        select(PipelineStageRun).where(PipelineStageRun.garment_id == garment.id)
    )
    # Stage run count should remain 9 (no duplicates created)
    assert len(all_runs_res.scalars().all()) == 9
