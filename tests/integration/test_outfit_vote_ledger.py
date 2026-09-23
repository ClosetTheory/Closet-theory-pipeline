"""The behaviour ledger: every 👍/👎 and every review star lands in outfit_votes, upserted per
(outfit, voter, source), and Stage 3 learns from it."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.garment import Garment
from app.models.outfit_vote import OutfitVote
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.models.user import User
from app.styling.behavior import load_behavior_model


async def _owner(db_session) -> User:
    return (await db_session.execute(select(User).where(User.email == "test_user@example.com"))).scalars().one()


async def _outfit_with_garments(db_session, owner: User, n: int = 3) -> Outfit:
    request = StylingRequest(tenant_id=owner.tenant_id, member_id=owner.member_id, raw_text="ledger test")
    db_session.add(request)
    await db_session.flush()
    outfit = Outfit(request_id=request.id, tenant_id=owner.tenant_id, member_id=owner.member_id, rank=1, final_score=0.7,
                    score_breakdown={"visual_harmony": 0.8})
    db_session.add(outfit)
    await db_session.flush()
    for i in range(n):
        g = Garment(id=f"garm_ledger_{outfit.id[-6:]}_{i}", tenant_id=owner.tenant_id, member_id=owner.member_id,
                    source_image_id="src", category=["TOP", "BOTTOM", "FOOTWEAR"][i % 3], subcategory="x",
                    status="COMPLETED", quality_status="APPROVED", attributes_json={"colour": ["navy"]})
        db_session.add(g)
        db_session.add(OutfitGarment(outfit_id=outfit.id, garment_id=g.id, role=g.category))
    await db_session.commit()
    return outfit


@pytest.mark.asyncio
async def test_thumbs_vote_is_recorded_once_per_voter_and_flips_in_place(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit_with_garments(db_session, owner)

    first = await client.post(f"/api/v1/wardrobe/styling/outfits/{outfit.id}/vote", json={"vote": "down"}, headers=auth_headers)
    assert first.status_code == 200
    body = first.json()
    assert sorted(body["garment_ids"]) == sorted(f"garm_ledger_{outfit.id[-6:]}_{i}" for i in range(3))
    assert body["ledger_votes"] == 1
    assert all(0 < s < 0.5 for s in body["garment_behavior_scores"].values()), "a dislike should score every garment below neutral"

    second = await client.post(f"/api/v1/wardrobe/styling/outfits/{outfit.id}/vote", json={"vote": "up"}, headers=auth_headers)
    assert second.status_code == 200

    rows = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().all()
    assert len(rows) == 1, "re-voting must flip the existing row, not stack a second one"
    assert rows[0].vote == "up" and rows[0].source == "styling_page" and rows[0].weight == 1.0
    assert len(rows[0].garment_ids) == 3

    model = await load_behavior_model(db_session, owner.tenant_id, owner.member_id)
    assert model.votes_considered == 1
    assert all(model.garment_score(g) > 0.5 for g in rows[0].garment_ids)


@pytest.mark.asyncio
async def test_own_review_stars_feed_the_ledger_and_a_three_withdraws(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit_with_garments(db_session, owner)
    url = f"/api/v1/wardrobe/styling/outfits/{outfit.id}/score"

    assert (await client.put(url, json={"rating": 1}, headers=auth_headers)).status_code == 200
    row = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().one()
    assert (row.source, row.vote, row.weight) == ("own_review", "down", 1.0)

    assert (await client.put(url, json={"rating": 4}, headers=auth_headers)).status_code == 200
    await db_session.refresh(row)
    assert (row.vote, row.weight) == ("up", 0.5)

    assert (await client.put(url, json={"rating": 3}, headers=auth_headers)).status_code == 200
    assert (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().all() == []


@pytest.mark.asyncio
async def test_thumbs_and_review_from_the_same_person_are_separate_rows(client: AsyncClient, db_session, auth_headers):
    owner = await _owner(db_session)
    outfit = await _outfit_with_garments(db_session, owner)
    await client.post(f"/api/v1/wardrobe/styling/outfits/{outfit.id}/vote", json={"vote": "down"}, headers=auth_headers)
    await client.put(f"/api/v1/wardrobe/styling/outfits/{outfit.id}/score", json={"rating": 2}, headers=auth_headers)
    rows = (await db_session.execute(select(OutfitVote).where(OutfitVote.outfit_id == outfit.id))).scalars().all()
    assert sorted(r.source for r in rows) == ["own_review", "styling_page"]
