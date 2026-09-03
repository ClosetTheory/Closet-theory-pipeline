"""Integration tests for FastAPI REST endpoints."""

import io
import pytest
from httpx import AsyncClient
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.pipeline.state_machine import GarmentState


@pytest.mark.asyncio
async def test_image_upload_success(client: AsyncClient, sample_catalog_image_bytes):
    files = {"file": ("test.jpg", sample_catalog_image_bytes, "image/jpeg")}
    response = await client.post("/api/v1/wardrobe/images", files=files, data={"tenant_id": "t1", "member_id": "m1"})

    assert response.status_code == 201
    data = response.json()
    assert "image_id" in data
    assert data["mime_type"] == "image/jpeg"
    assert data["width"] == 800
    assert data["height"] == 800


@pytest.mark.asyncio
async def test_image_upload_unsupported_mime(client: AsyncClient):
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    response = await client.post("/api/v1/wardrobe/images", files=files)
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_garment_lifecycle_and_pipeline_api(
    client: AsyncClient,
    db_session,
    test_storage,
    sample_catalog_image_bytes,
):
    # 1. Upload image
    files = {"file": ("shirt.jpg", sample_catalog_image_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/wardrobe/images", files=files)
    assert upload_res.status_code == 201
    image_id = upload_res.json()["image_id"]

    # 2. Create garment
    garment_res = await client.post(
        "/api/v1/wardrobe/garments",
        json={"source_image_id": image_id, "tenant_id": "tenant_1", "member_id": "member_1"},
    )
    assert garment_res.status_code == 202
    garment_id = garment_res.json()["garment_id"]

    # 3. Retrieve garment
    get_res = await client.get(f"/api/v1/wardrobe/garments/{garment_id}")
    assert get_res.status_code == 200
    g_data = get_res.json()
    assert g_data["garment_id"] == garment_id

    # 4. Check pipeline audit status
    pipe_res = await client.get(f"/api/v1/wardrobe/garments/{garment_id}/pipeline")
    assert pipe_res.status_code == 200
    pipe_data = pipe_res.json()
    assert pipe_data["garment_id"] == garment_id
    assert "stages" in pipe_data

    # 5. Test retry endpoint
    retry_res = await client.post(
        f"/api/v1/wardrobe/garments/{garment_id}/retry",
        json={"stage": "STAGE_01_CLASSIFY", "force": True},
    )
    assert retry_res.status_code == 202

    # 6. Test review endpoint (APPROVE)
    review_res = await client.post(
        f"/api/v1/wardrobe/garments/{garment_id}/review",
        json={"decision": "APPROVE", "notes": "Approved by human operator"},
    )
    assert review_res.status_code == 200
    assert review_res.json()["decision"] == "APPROVE"


@pytest.mark.asyncio
async def test_compatibility_endpoint(
    client: AsyncClient,
    db_session,
    valid_attributes_dict,
):
    # Create two synthetic garments in database
    asset_a = ImageAsset(
        tenant_id="t1", member_id="m1", object_uri="test://a.jpg",
        mime_type="image/jpeg", width=800, height=800, sha256="sha_a"
    )
    asset_b = ImageAsset(
        tenant_id="t1", member_id="m1", object_uri="test://b.jpg",
        mime_type="image/jpeg", width=800, height=800, sha256="sha_b"
    )
    db_session.add_all([asset_a, asset_b])
    await db_session.commit()

    # Garment A: Oxford Shirt (TOP, base)
    garment_a = Garment(
        tenant_id="t1", member_id="m1", source_image_id=asset_a.id,
        category="TOP", subcategory="oxford_shirt",
        attributes_json=valid_attributes_dict,
        status=GarmentState.COMPLETED.value,
        quality_status="APPROVED",
    )
    # Garment B: Chinos (BOTTOM, standalone)
    chinos_attrs = {
        **valid_attributes_dict,
        "category": "pants",
        "subcategory": "chinos",
        "colour": ["navy"],
        "layering_role": "standalone",
    }
    garment_b = Garment(
        tenant_id="t1", member_id="m1", source_image_id=asset_b.id,
        category="BOTTOM", subcategory="chinos",
        attributes_json=chinos_attrs,
        status=GarmentState.COMPLETED.value,
        quality_status="APPROVED",
    )
    db_session.add_all([garment_a, garment_b])
    await db_session.commit()

    # Evaluate compatibility
    res = await client.post(
        "/api/v1/wardrobe/compatibility",
        json={
            "garment_a_id": garment_a.id,
            "garment_b_id": garment_b.id,
            "types": ["LAYERING", "STRUCTURAL", "VISUAL"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["garment_a_id"] == garment_a.id
    assert data["garment_b_id"] == garment_b.id
    assert data["overall_decision"] == "COMPATIBLE"
    assert len(data["results"]) == 3
