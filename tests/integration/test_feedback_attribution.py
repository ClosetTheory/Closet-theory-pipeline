"""A vote with a reason is attributed: chips and comments become per-garment weights in the ledger,
lessons sharpen the style profile, and the notes reach Stage 2's context."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.config import settings
from app.models.garment import Garment
from app.models.outfit_vote import OutfitVote
from app.models.style_profile import StyleProfile
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.models.user import User
from app.styling.behavior import load_behavior_model, load_taste_notes


@pytest.fixture(autouse=True)
def _mock_extractor(monkeypatch):
    monkeypatch.setattr(settings, "STYLING_FEEDBACK_PROVIDER", "mock")


async def _owner(db_session) -> User:
    return (await db_session.execute(select(User).where(User.email == "test_user@example.com"))).scalars().one()


async def _outfit(db_session, owner: User) -> Outfit:
    request = StylingRequest(tenant_id=owner.tenant_id, member_id=owner.member_id, raw_text="brunch")
    db_session.add(request)
    await db_session.flush()
    outfit = Outfit(request_id=request.id, tenant_id=owner.tenant_id, member_id=owner.member_id, rank=1, final_score=0.7,
                    score_breakdown={"visual_harmony": 0.8})
    db_session.add(outfit)
    await db_session.flush()
    pieces = [("g_shirt", "TOP", "LINEN_SHIRT", "linen shirt", ["white"]),
              ("g_trousers", "BOTTOM", "TROUSERS", "floral trousers", ["green"]),
              ("g_sneakers", "FOOTWEAR", "SNEAKERS", "white sneakers", ["white"])]
    for gid, cat, sub, name, colours in pieces:
        gid = f"{gid}_{outfit.id[-6:]}"
        db_session.add(Garment(id=gid, tenant_id=owner.tenant_id, member_id=owner.member_id, source_image_id="src",
                               category=cat, subcategory=sub, status="COMPLETED", quality_status="APPROVED",
                               attributes_json={"casual_name": name, "colour": colours}))
        db_session.add(OutfitGarment(outfit_id=outfit.id, garment_id=gid, role=cat))
    await db_session.commit()
    return outfit


@pytest.mark.asyncio
async def test_comment_blames_the_named_garment_and_teaches_a_lesson(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit(db_session, owner)
    sfx = outfit.id[-6:]

    res = await client.post(
        f"/api/v1/wardrobe/styling/outfits/{outfit.id}/vote",
        json={"vote": "down", "comment": "The sneakers kill it. Also no florals for me.", "tags": ["colour_clash"]},
        headers=auth_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert "blamed white sneakers" in body["feedback_summary"]
    assert "colour clash" in body["feedback_summary"]
    assert "no floral" in body["feedback_summary"]

    row = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().one()
    assert row.comment.startswith("The sneakers")
    assert row.tags == ["colour_clash"]
    assert row.feedback["blamed_garment_ids"] == [f"g_sneakers_{sfx}"]
    assert row.garment_weights[f"g_sneakers_{sfx}"] > row.garment_weights[f"g_shirt_{sfx}"]

    model = await load_behavior_model(db_session, owner.tenant_id, owner.member_id)
    assert model.garment_score(f"g_sneakers_{sfx}") < model.garment_score(f"g_shirt_{sfx}") < 0.5

    profile = (await db_session.execute(select(StyleProfile).where(StyleProfile.tenant_id == owner.tenant_id))).scalars().one()
    assert profile.attribute_affinities["pattern"]["floral"]["score"] < 0

    notes = await load_taste_notes(db_session, owner.tenant_id, owner.member_id)
    assert any("The sneakers kill it" in n for n in notes)
    assert any("dislikes floral" in n for n in notes)


@pytest.mark.asyncio
async def test_reason_after_a_bare_vote_updates_the_same_row(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit(db_session, owner)
    url = f"/api/v1/wardrobe/styling/outfits/{outfit.id}/vote"

    first = await client.post(url, json={"vote": "down"}, headers=auth_headers)
    assert first.json()["feedback_summary"] is None
    second = await client.post(url, json={"vote": "down", "comment": "love the shirt, the rest is drab"}, headers=auth_headers)
    assert "praised linen shirt" in second.json()["feedback_summary"]

    rows = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().all()
    assert len(rows) == 1
    assert rows[0].counter_garment_ids == [f"g_shirt_{outfit.id[-6:]}"]


@pytest.mark.asyncio
async def test_review_comment_and_chips_reach_the_ledger(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit(db_session, owner)
    res = await client.put(
        f"/api/v1/wardrobe/styling/outfits/{outfit.id}/score",
        json={"rating": 2, "comment": "shirt with those trousers clashes", "tags": ["colour_clash", "not_a_real_tag"]},
        headers=auth_headers,
    )
    assert res.status_code == 200
    row = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().one()
    assert row.source == "own_review" and row.tags == ["colour_clash"]
    sfx = outfit.id[-6:]
    assert row.feedback["pairings"] == [[f"g_shirt_{sfx}", f"g_trousers_{sfx}"]]
    assert row.pair_weights[f"g_shirt_{sfx}|g_trousers_{sfx}"] == 2.0
