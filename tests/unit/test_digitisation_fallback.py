"""The image-model ladder: what happens when the preferred model declines, and when all do.

These two behaviours replaced a local grabCut compositor that returned the member's own photo
pasted on a blank canvas and reported it as a 0.92-scoring success. Both are worth pinning down,
because the failure they replaced was invisible by construction.
"""

import pytest

from app.config import settings
from app.providers.base import ImageGenerationRefusedError
from app.providers.digitisation.gpt_digitiser import GPTStudioDigitisationProvider
from app.schemas.attributes import validate_extracted_attributes


def _provider(monkeypatch, result, record_declines=()):
    prov = GPTStudioDigitisationProvider(api_key="test-key")

    async def fake_call(prompt, crop_bytes):
        prov._last_generation_errors.extend(record_declines)
        return result

    monkeypatch.setattr(prov, "_call_openrouter_image_gen", fake_call)
    return prov


@pytest.mark.asyncio
async def test_preferred_model_records_no_fallback(
    monkeypatch, sample_catalog_image_bytes, valid_attributes_dict
):
    attrs = validate_extracted_attributes(valid_attributes_dict)
    prov = _provider(monkeypatch, (b"image-bytes", settings.OPENROUTER_IMAGE_MODEL))

    res = await prov.digitise(sample_catalog_image_bytes, attrs, attempt=1)

    assert settings.OPENROUTER_IMAGE_MODEL in res.model
    assert prov._last_fallback is None


@pytest.mark.asyncio
async def test_fallback_model_records_which_one_and_why(
    monkeypatch, sample_catalog_image_bytes, valid_attributes_dict
):
    attrs = validate_extracted_attributes(valid_attributes_dict)
    refusal = {
        "model": settings.OPENROUTER_IMAGE_MODEL,
        "status": 400,
        "reason": "Your request was rejected by the safety system.",
        "kind": "safety_refusal",
    }
    prov = _provider(
        monkeypatch, (b"image-bytes", "google/gemini-3-pro-image"), record_declines=[refusal]
    )

    res = await prov.digitise(sample_catalog_image_bytes, attrs, attempt=1)

    assert "google/gemini-3-pro-image" in res.model
    assert prov._last_fallback is not None
    assert prov._last_fallback["model"] == "google/gemini-3-pro-image"
    assert prov._last_fallback["preferred_model"] == settings.OPENROUTER_IMAGE_MODEL
    # The reason the preferred model gave has to survive into the run, or the only record of it
    # is a log line in a container that will be restarted.
    assert prov._last_fallback["kind"] == "safety_refusal"
    assert refusal in prov._last_fallback["declined"]


@pytest.mark.asyncio
async def test_every_model_declining_raises_rather_than_compositing(
    monkeypatch, sample_catalog_image_bytes, valid_attributes_dict
):
    attrs = validate_extracted_attributes(valid_attributes_dict)
    declines = [
        {"model": "openai/gpt-image-2", "status": 400, "reason": "rejected by the safety system",
         "kind": "safety_refusal"},
        {"model": "google/gemini-3-pro-image", "status": 400, "reason": "rejected",
         "kind": "rejected"},
    ]
    prov = _provider(monkeypatch, (None, ""), record_declines=declines)

    with pytest.raises(ImageGenerationRefusedError) as exc:
        await prov.digitise(sample_catalog_image_bytes, attrs, attempt=1)

    # Every model's own words, so a reviewer can tell a policy refusal from an outage.
    assert "openai/gpt-image-2" in str(exc.value)
    assert "google/gemini-3-pro-image" in str(exc.value)
    assert prov._last_generated_bytes is None


@pytest.mark.asyncio
async def test_missing_api_key_raises_instead_of_returning_a_stand_in(
    sample_catalog_image_bytes, valid_attributes_dict
):
    attrs = validate_extracted_attributes(valid_attributes_dict)
    prov = GPTStudioDigitisationProvider(api_key="test-key")
    # The constructor falls back to settings.OPENROUTER_API_KEY, so an unconfigured provider has
    # to be made after construction - otherwise this test quietly calls the live API.
    prov.api_key = ""

    with pytest.raises(ImageGenerationRefusedError):
        await prov.digitise(sample_catalog_image_bytes, attrs, attempt=1)
