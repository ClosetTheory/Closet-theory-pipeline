"""Integration tests for live step-by-step pipeline endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_step_by_step_execution_flow(sample_catalog_image_bytes):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Upload
        files = {"file": ("item.jpg", sample_catalog_image_bytes, "image/jpeg")}
        up_resp = await client.post("/api/v1/wardrobe/images", files=files)
        assert up_resp.status_code == 201
        image_id = up_resp.json()["image_id"]

        # 2. Ingest
        gar_resp = await client.post("/api/v1/wardrobe/garments", json={"source_image_id": image_id})
        assert gar_resp.status_code == 202
        garment_id = gar_resp.json()["garment_id"]

        # 3. Execute Step 1 (Classifier)
        step1 = await client.post(f"/api/v1/wardrobe/garments/{garment_id}/step", json={"stage": "STAGE_01_CLASSIFY", "force": True})
        assert step1.status_code == 200
        data1 = step1.json()
        assert data1["stage"] == "STAGE_01_CLASSIFY"
        assert data1["status"] == "SUCCEEDED"
        assert "raw_image_url" in data1["visual_artifacts"]

        # 4. Execute Step 2 (Crop)
        step2 = await client.post(f"/api/v1/wardrobe/garments/{garment_id}/step", json={"stage": "STAGE_02_CROP", "force": True})
        assert step2.status_code == 200
        data2 = step2.json()
        assert data2["stage"] == "STAGE_02_CROP"
        assert data2["status"] == "SUCCEEDED"
        assert data2["visual_artifacts"]["annotated_overlay_url"] is not None

        # 5. Execute Step 3 (Attributes)
        step3 = await client.post(f"/api/v1/wardrobe/garments/{garment_id}/step", json={"stage": "STAGE_03_ATTRIBUTES", "force": True})
        assert step3.status_code == 200
        data3 = step3.json()
        assert data3["stage"] == "STAGE_03_ATTRIBUTES"
        assert data3["status"] == "SUCCEEDED"
        assert "subcategory" in data3["output_data"]
