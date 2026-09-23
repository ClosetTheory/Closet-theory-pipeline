"""Scoring one's own account's outfits with the character panel's five-dimension rubric.

An admin styling their own wardrobe ranks those outfits on /review exactly as a stylist ranks a
character's — but those rows cannot live in persona_outfit_reviews (its persona column is
required), so they take a sibling table and two endpoints under /wardrobe/styling.
"""

import pytest
from httpx import AsyncClient

from app.models.base import generate_uuid
from app.models.role import ROLE_ADMIN, ROLE_STYLIST, UserRole
from app.models.styling import Outfit, StylingRequest
from app.models.user import User


async def _make_user(session, email: str, role: str | None = None) -> User:
    from app.auth.security import hash_password

    user_id = generate_uuid("user")
    user = User(
        id=user_id, email=email, display_name=email.split("@")[0],
        password_hash=hash_password("testpassword123"),
        tenant_id=user_id, member_id=user_id,
    )
    session.add(user)
    if role:
        session.add(UserRole(user_id=user_id, role=role))
    await session.commit()
    return user


async def _make_outfit(session, owner: User, text: str = "own request") -> Outfit:
    request = StylingRequest(tenant_id=owner.tenant_id, member_id=owner.member_id, raw_text=text)
    session.add(request)
    await session.flush()
    outfit = Outfit(request_id=request.id, tenant_id=owner.tenant_id, member_id=owner.member_id, rank=1, final_score=0.8)
    session.add(outfit)
    await session.commit()
    return outfit


async def _headers(client: AsyncClient, email: str) -> dict:
    res = await client.post("/api/v1/auth/login", json={"email": email, "password": "testpassword123"})
    return {"Authorization": f"Bearer {res.json()['token']}"}


@pytest.mark.asyncio
async def test_admin_scores_their_own_outfit_and_sees_it_listed(client: AsyncClient, db_session):
    admin = await _make_user(db_session, "own-admin@example.com", ROLE_ADMIN)
    outfit = await _make_outfit(db_session, admin, text="smart casual dinner")
    headers = await _headers(client, "own-admin@example.com")

    listed = await client.get("/api/v1/wardrobe/styling/outfits/reviewable", headers=headers)
    assert listed.status_code == 200
    assert [it["outfit"]["outfit_id"] for it in listed.json()] == [outfit.id]
    assert listed.json()[0]["request_text"] == "smart casual dinner"
    assert listed.json()[0]["my_review"] is None

    saved = await client.put(
        f"/api/v1/wardrobe/styling/outfits/{outfit.id}/score",
        json={"rating": 4, "dimension_ratings": {"colour_harmony": 5, "persona_fit": 3}, "would_wear": True, "comment": "good"},
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json()["rating"] == 4
    assert saved.json()["dimension_ratings"] == {"colour_harmony": 5, "persona_fit": 3}

    # Revising updates the same row rather than adding another.
    again = await client.put(
        f"/api/v1/wardrobe/styling/outfits/{outfit.id}/score", json={"rating": 2}, headers=headers
    )
    assert again.json()["id"] == saved.json()["id"]
    assert again.json()["rating"] == 2

    relisted = await client.get("/api/v1/wardrobe/styling/outfits/reviewable", headers=headers)
    item = relisted.json()[0]
    assert item["my_review"]["id"] == saved.json()["id"]
    assert len(item["reviews"]) == 1


@pytest.mark.asyncio
async def test_own_score_rejects_another_accounts_outfit_and_unknown_dimensions(client: AsyncClient, db_session):
    admin = await _make_user(db_session, "own-admin2@example.com", ROLE_ADMIN)
    other = await _make_user(db_session, "own-other@example.com")
    foreign_outfit = await _make_outfit(db_session, other)
    headers = await _headers(client, "own-admin2@example.com")

    forbidden = await client.put(
        f"/api/v1/wardrobe/styling/outfits/{foreign_outfit.id}/score", json={"rating": 5}, headers=headers
    )
    assert forbidden.status_code == 403

    own = await _make_outfit(db_session, admin)
    bad = await client.put(
        f"/api/v1/wardrobe/styling/outfits/{own.id}/score",
        json={"rating": 5, "dimension_ratings": {"sparkle": 5}},
        headers=headers,
    )
    assert bad.status_code == 422

    # The other account sees only its own outfit, never the admin's.
    other_list = await client.get(
        "/api/v1/wardrobe/styling/outfits/reviewable", headers=await _headers(client, "own-other@example.com")
    )
    assert [it["outfit"]["outfit_id"] for it in other_list.json()] == [foreign_outfit.id]


@pytest.mark.asyncio
async def test_stylist_without_a_character_cannot_use_own_review(client: AsyncClient, db_session):
    """Stylists always work through a character; the own-account path is not a back door."""
    await _make_user(db_session, "own-stylist@example.com", ROLE_STYLIST)
    headers = await _headers(client, "own-stylist@example.com")
    res = await client.get("/api/v1/wardrobe/styling/outfits/reviewable", headers=headers)
    assert res.status_code == 403
