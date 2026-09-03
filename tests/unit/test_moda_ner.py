"""Unit tests for Stage 3: MODA_NER attribute extraction (crop/catalog/fullbody tracks)."""

from unittest.mock import AsyncMock, patch

import pytest
from app.providers.attributes.moda_ner import ModaNerAttributeExtractorProvider
from app.providers.attributes.moda_ner_mapping import (
    map_catalog_track,
    map_crop_track,
    map_fullbody_track,
)
from app.providers.vlm.openrouter import OpenRouterGPTProvider
from app.schemas.attributes import validate_extracted_attributes


def test_map_crop_track_maps_known_values():
    raw = {
        "master_category": "top t-shirt sweatshirt",
        "category": "top t-shirt sweatshirt",
        "sub_category": "hoodie",
        "pattern": "stripe",
        "sleeve_length": "wrist-length",
        "silhouette": "regular",
    }
    out = map_crop_track(raw)
    assert out["subcategory"] == "hoodie"
    assert out["pattern"] == "striped"
    assert out["sleeve_length"] == "long"
    assert out["silhouette"] == "straight"


def test_map_crop_track_applies_subcategory_synonyms():
    assert map_crop_track({"sub_category": "puffer"})["subcategory"] == "puffer_jacket"
    assert map_crop_track({"sub_category": "trench"})["subcategory"] == "trench_coat"


def test_map_crop_track_omits_ungated_fields():
    # Low-confidence fields are simply absent from the model's own output (applicability
    # gating), not empty strings — the mapper must not invent values for missing keys.
    out = map_crop_track({"master_category": "accessories", "category": "top t-shirt sweatshirt"})
    assert "pattern" not in out
    assert "silhouette" not in out
    assert "subcategory" not in out


def test_map_catalog_track_maps_known_values():
    raw = {
        "category": "jean",
        "color": "navy",
        "fabric": "denim",
        "fit": "skinny",
        "pattern": "plain",
        "sleeve_length": "long",
        "pocket": "flap",
    }
    out = map_catalog_track(raw)
    assert out["subcategory"] == "jeans"
    assert out["colour"] == ["navy"]
    assert out["material"] == "denim"
    assert out["fit"] == "slim"
    assert out["pattern"] == "solid"
    assert out["sleeve_length"] == "long"
    assert out["pocket_detail"] == "flap"


def test_map_fullbody_track_has_no_category_or_subcategory():
    out = map_fullbody_track(
        {
            "upper_fabric": "cotton",
            "upper_pattern": "graphic",
            "sleeve_length": "long-sleeve",
        }
    )
    assert "category" not in out
    assert "subcategory" not in out
    assert out["material"] == "cotton"
    assert out["pattern"] == "graphic"
    assert out["sleeve_length"] == "long"


def test_map_fullbody_track_treats_na_as_absent():
    out = map_fullbody_track({"upper_fabric": "NA"})
    assert "material" not in out


@pytest.mark.asyncio
async def test_moda_ner_provider_merges_structural_and_topup(sample_catalog_image_bytes, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RUNPOD_ATTRIBUTE_ENDPOINT_ID", "test-endpoint")

    runpod_response = AsyncMock()
    runpod_response.raise_for_status = lambda: None
    runpod_response.json = lambda: {
        "status": "COMPLETED",
        "output": {
            "track": "catalog",
            "attributes": {
                "category": "jean",
                "color": "navy",
                "fabric": "denim",
                "fit": "skinny",
                "pattern": "plain",
                "sleeve_length": "long",
            },
        },
    }

    topup_fields = {
        # catalog track never supplies silhouette -- the real dynamic top-up would
        # request it too, since it's missing from the structural (Runpod) result.
        "silhouette": "straight",
        "occasion": ["casual"],
        "season": ["all_season"],
        "layering_role": "base",
        "warmth": 0.3,
        "versatility": 0.8,
    }

    provider = ModaNerAttributeExtractorProvider()
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=runpod_response)), patch.object(
        provider._topup, "extract_soft_attributes", new=AsyncMock(return_value=topup_fields)
    ):
        attributes = await provider.extract_attributes(sample_catalog_image_bytes, image_type="CATALOG")

    assert attributes.subcategory == "jeans"
    assert attributes.colour == ["navy"]
    assert attributes.material == "denim"
    assert attributes.fit.value == "slim"
    assert attributes.occasion[0].value == "casual"
    assert attributes.warmth == 0.3


@pytest.mark.asyncio
async def test_moda_ner_provider_falls_back_to_mock_on_runpod_failure(sample_catalog_image_bytes, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RUNPOD_ATTRIBUTE_ENDPOINT_ID", "test-endpoint")

    provider = ModaNerAttributeExtractorProvider()
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=RuntimeError("network down"))):
        attributes = await provider.extract_attributes(sample_catalog_image_bytes, image_type="CATALOG")

    # Falls back to the deterministic mock rather than raising.
    validated = validate_extracted_attributes(attributes.model_dump(mode="json"))
    assert validated.subcategory


@pytest.mark.asyncio
async def test_extract_soft_attributes_only_requests_missing_fields(sample_catalog_image_bytes):
    """A track that already supplied category/subcategory/colour/pattern/material/fit
    (e.g. catalog) should only be asked for the fields it structurally can't produce."""
    provider = OpenRouterGPTProvider(api_key="test-key")
    known = {
        "category": "jean",
        "subcategory": "jeans",
        "colour": ["navy"],
        "pattern": "solid",
        "material": "denim",
        "fit": "slim",
        # silhouette intentionally absent -- catalog track never supplies it
    }

    captured_prompt = {}

    async def fake_vision_chat_json(prompt_text, image_bytes, max_tokens=400):
        captured_prompt["text"] = prompt_text
        return '{"silhouette": "straight", "occasion": ["casual"], "season": ["all_season"], "layering_role": "base", "warmth": 0.3, "versatility": 0.8}'

    with patch.object(provider, "_vision_chat_json", new=fake_vision_chat_json):
        result = await provider.extract_soft_attributes(sample_catalog_image_bytes, known=known)

    assert result["silhouette"] == "straight"
    assert result["warmth"] == 0.3

    # Must not have asked for fields the track already supplied. Only check the
    # fill-schema block, since "Known attributes: {...}" legitimately echoes them back.
    schema_block = captured_prompt["text"].split("Known attributes")[0]
    assert '"category"' not in schema_block
    assert '"colour"' not in schema_block
    assert '"fit"' not in schema_block
    assert '"silhouette"' in schema_block
