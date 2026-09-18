"""A character's outfits must render on their own portrait, not the anonymous mannequin.

This pins down the fix for a real regression: characters' generated outfits (both the 3 daily
prompted looks and OOTD, since OOTD runs the same orchestrator underneath) were being rendered on
the same faceless mannequin used for ordinary members, even though each character has a real
portrait photo to render on. Two things are worth pinning independently:

  - the prompt actually changes shape based on whether a persona reference is present, and the
    reference photo is sent as an image-to-image input rather than just mentioned in text;
  - the lookup that decides whether a tenant IS a character is a real DB query, not a guess, and
    correctly returns None for an ordinary member (no Persona row) and for a character with no
    portrait yet (Persona row exists, portrait_image_id is None) — both must keep the mannequin
    path, only a portrait that actually exists should switch it.
"""

import base64

import pytest
from sqlalchemy import select

from app.models.image_asset import ImageAsset
from app.models.persona import Persona
from app.models.user import User
from app.providers.outfit_imaging.gpt_outfit_provider import GPTOutfitImageProvider
from app.schemas.styling import GarmentSummary
from app.styling.imaging import load_persona_portrait

SAMPLE_GARMENTS = [
    GarmentSummary(garment_id="garm_1", category="TOP", subcategory="shirt",
                   attributes={"colour": ["black"], "subcategory": "shirt"},
                   status="COMPLETED", quality_status="APPROVED"),
]


def test_prompt_without_persona_keeps_the_mannequin():
    provider = GPTOutfitImageProvider(api_key="test-key")
    prompt = provider._build_prompt(SAMPLE_GARMENTS, has_persona=False)
    assert "mannequin" in prompt.lower()
    assert "reference image" not in prompt.lower() or "same person" not in prompt.lower()


def test_prompt_with_persona_renders_on_that_person_not_a_mannequin():
    provider = GPTOutfitImageProvider(api_key="test-key")
    prompt = provider._build_prompt(SAMPLE_GARMENTS, has_persona=True)
    # "mannequin" is allowed to appear once, as a negative exclusion ("not a mannequin") — the
    # same pattern the digitisation prompt uses for what to avoid. What must NOT appear is the
    # mannequin-prompt's own framing sentence, which opens by naming it as the subject.
    assert "on a full matte black mannequin" not in prompt.lower()
    assert prompt.lower().count("mannequin") <= 1
    assert "same person" in prompt.lower()
    assert "same face" in prompt.lower()


@pytest.mark.asyncio
async def test_generate_puts_persona_reference_first_in_input_references(monkeypatch):
    provider = GPTOutfitImageProvider(api_key="test-key")
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(b"fake-image").decode()}]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers, json):
            captured["payload"] = json
            return FakeResponse()

    import app.providers.outfit_imaging.gpt_outfit_provider as mod
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda timeout=60.0: FakeClient())

    portrait_bytes = b"portrait-bytes"
    garment_bytes = b"garment-bytes"
    await provider.generate(SAMPLE_GARMENTS, [garment_bytes], persona_reference=portrait_bytes)

    refs = captured["payload"]["input_references"]
    assert len(refs) == 2
    first_b64 = refs[0]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(first_b64) == portrait_bytes
    assert "same person" in captured["payload"]["prompt"].lower()


@pytest.mark.asyncio
async def test_load_persona_portrait_is_none_for_an_ordinary_member(db_session, test_storage):
    result = await load_persona_portrait(db_session, test_storage, "tenant_ordinary_member")
    assert result is None


@pytest.mark.asyncio
async def test_load_persona_portrait_is_none_when_no_portrait_generated_yet(db_session, test_storage):
    from app.models.base import generate_uuid

    user_id = generate_uuid("user")
    db_session.add(User(
        id=user_id, email=f"{user_id}@test.local", display_name="No Portrait Yet",
        password_hash="!locked", tenant_id=user_id, member_id=user_id,
    ))
    db_session.add(Persona(
        user_id=user_id, slug=f"slug_{user_id}", display_name="No Portrait Yet",
        gender="women", city="Mumbai", country="India", body_shape="rectangle",
        skin_tone_monk=5,
        # portrait_image_id deliberately omitted — this is the "assigned but not yet portrait-ed"
        # state every character passes through before seeding, and it must still get the
        # mannequin, not a crash or a None-image reference.
    ))
    await db_session.commit()

    result = await load_persona_portrait(db_session, test_storage, user_id)
    assert result is None


@pytest.mark.asyncio
async def test_load_persona_portrait_returns_real_bytes_for_a_character(db_session, test_storage):
    from app.models.base import generate_uuid

    user_id = generate_uuid("user")
    portrait_bytes = b"a real portrait image, in bytes"
    object_uri = await test_storage.put_object(
        f"personas/{user_id}/portrait.jpg", portrait_bytes, content_type="image/jpeg"
    )
    asset_id = generate_uuid("img")
    db_session.add(ImageAsset(
        id=asset_id, tenant_id=user_id, member_id=user_id, object_uri=object_uri,
        mime_type="image/jpeg", width=1024, height=1536, sha256="deadbeef",
    ))
    db_session.add(User(
        id=user_id, email=f"{user_id}@test.local", display_name="Has A Portrait",
        password_hash="!locked", tenant_id=user_id, member_id=user_id,
    ))
    db_session.add(Persona(
        user_id=user_id, slug=f"slug_{user_id}", display_name="Has A Portrait",
        gender="men", city="Delhi", country="India", body_shape="triangle",
        skin_tone_monk=6, portrait_image_id=asset_id,
    ))
    await db_session.commit()

    result = await load_persona_portrait(db_session, test_storage, user_id)
    assert result == portrait_bytes
