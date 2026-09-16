"""The evaluation-character panel: isolation, authorization and multi-reviewer scoring.

The load-bearing claim of this feature is that a stylist acting as a character writes into that
character's wardrobe and nowhere else, using the tenant checks the wardrobe endpoints already
perform. These tests assert that directly, because if it is wrong the failure is silent — garments
simply accumulate in the wrong account and nobody notices until a review session.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.base import generate_uuid
from app.models.garment import Garment
from app.models.persona import Persona, PersonaAssignment
from app.models.role import ROLE_ADMIN, ROLE_STYLIST, UserRole
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


async def _make_persona(session, slug: str) -> Persona:
    backing_id = generate_uuid("user")
    session.add(User(
        id=backing_id, email=f"{slug}@personas.closettheory.local", display_name=slug,
        password_hash="!locked-persona-account", gender="women",
        tenant_id=backing_id, member_id=backing_id,
    ))
    persona = Persona(
        slug=slug, user_id=backing_id, display_name=slug.replace("_", " ").title(),
        gender="women", city="mumbai", body_shape="rectangle", bio="test character",
    )
    session.add(persona)
    await session.commit()
    return persona


async def _token(client: AsyncClient, email: str) -> str:
    res = await client.post("/api/v1/auth/login", json={"email": email, "password": "testpassword123"})
    return res.json()["token"]


@pytest.mark.asyncio
async def test_stylist_upload_lands_on_the_character_not_the_stylist(
    client: AsyncClient, db_session, sample_catalog_image_bytes
):
    """The whole design in one assertion."""
    stylist = await _make_user(db_session, "s1@example.com", ROLE_STYLIST)
    persona = await _make_persona(db_session, "test_character_one")
    db_session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))
    await db_session.commit()

    headers = {
        "Authorization": f"Bearer {await _token(client, 's1@example.com')}",
        "X-Acting-As-Persona": persona.id,
    }
    upload = await client.post(
        "/api/v1/wardrobe/images",
        files={"file": ("t.jpg", sample_catalog_image_bytes, "image/jpeg")},
        headers=headers,
    )
    assert upload.status_code == 201

    created = await client.post(
        "/api/v1/wardrobe/garments",
        json={"source_image_id": upload.json()["image_id"], "auto_process": False},
        headers=headers,
    )
    assert created.status_code == 202

    garment = (await db_session.execute(
        select(Garment).where(Garment.id == created.json()["garment_id"])
    )).scalars().first()
    assert garment.tenant_id == persona.user_id, "garment must belong to the character"
    assert garment.tenant_id != stylist.id, "and must not belong to the stylist"


@pytest.mark.asyncio
async def test_a_stylist_cannot_act_as_an_unassigned_character(
    client: AsyncClient, db_session, sample_catalog_image_bytes
):
    await _make_user(db_session, "s2@example.com", ROLE_STYLIST)
    persona = await _make_persona(db_session, "test_character_two")  # assigned to nobody

    res = await client.post(
        "/api/v1/wardrobe/images",
        files={"file": ("t.jpg", sample_catalog_image_bytes, "image/jpeg")},
        headers={
            "Authorization": f"Bearer {await _token(client, 's2@example.com')}",
            "X-Acting-As-Persona": persona.id,
        },
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_a_revoked_assignment_stops_working(client: AsyncClient, db_session):
    stylist = await _make_user(db_session, "s3@example.com", ROLE_STYLIST)
    persona = await _make_persona(db_session, "test_character_three")
    db_session.add(PersonaAssignment(
        persona_id=persona.id, stylist_user_id=stylist.id, status="revoked"
    ))
    await db_session.commit()

    res = await client.get(
        f"/api/v1/personas/{persona.id}",
        headers={"Authorization": f"Bearer {await _token(client, 's3@example.com')}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_a_stylist_without_the_header_is_refused_rather_than_writing_to_their_own_wardrobe(
    client: AsyncClient, db_session, sample_catalog_image_bytes
):
    """Silently ingesting into the stylist's own account would be discovered far too late."""
    await _make_user(db_session, "s4@example.com", ROLE_STYLIST)

    res = await client.post(
        "/api/v1/wardrobe/images",
        files={"file": ("t.jpg", sample_catalog_image_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {await _token(client, 's4@example.com')}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_an_ordinary_member_is_completely_unaffected(
    client: AsyncClient, auth_headers, sample_catalog_image_bytes
):
    """No header, no role: behaviour must be exactly what it was before characters existed."""
    res = await client.post(
        "/api/v1/wardrobe/images",
        files={"file": ("t.jpg", sample_catalog_image_bytes, "image/jpeg")},
        headers=auth_headers,
    )
    assert res.status_code == 201


@pytest.mark.asyncio
async def test_admin_routes_reject_non_admins(client: AsyncClient, db_session, auth_headers):
    await _make_user(db_session, "s5@example.com", ROLE_STYLIST)
    stylist_headers = {"Authorization": f"Bearer {await _token(client, 's5@example.com')}"}

    for headers in (auth_headers, stylist_headers):
        assert (await client.get("/api/v1/admin/users", headers=headers)).status_code == 403
        assert (await client.get("/api/v1/admin/assignments", headers=headers)).status_code == 403


@pytest.mark.asyncio
async def test_two_stylists_can_review_the_same_outfit(client: AsyncClient, db_session):
    """The thing the old stylist_reviews table made impossible: its unique index on outfit_id
    allowed exactly one opinion per outfit, globally."""
    from app.models.styling import Outfit, StylingRequest

    persona = await _make_persona(db_session, "test_character_four")
    first = await _make_user(db_session, "r1@example.com", ROLE_STYLIST)
    second = await _make_user(db_session, "r2@example.com", ROLE_STYLIST)
    for stylist in (first, second):
        db_session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))

    request = StylingRequest(tenant_id=persona.user_id, member_id=persona.user_id, raw_text="test")
    db_session.add(request)
    await db_session.flush()
    outfit = Outfit(
        request_id=request.id, tenant_id=persona.user_id, member_id=persona.user_id,
        rank=1, final_score=0.8,
    )
    db_session.add(outfit)
    await db_session.commit()

    path = f"/api/v1/personas/{persona.id}/outfits/{outfit.id}/review"
    body = {"rating": 4, "dimension_ratings": {"colour_harmony": 5}, "comment": "works"}

    first_res = await client.put(
        path, json=body, headers={"Authorization": f"Bearer {await _token(client, 'r1@example.com')}"}
    )
    second_res = await client.put(
        path, json={"rating": 2, "comment": "disagree"},
        headers={"Authorization": f"Bearer {await _token(client, 'r2@example.com')}"},
    )
    assert first_res.status_code == 200
    assert second_res.status_code == 200
    assert first_res.json()["id"] != second_res.json()["id"], "two distinct reviews must survive"

    # Re-submitting updates in place rather than accumulating.
    again = await client.put(
        path, json={"rating": 5}, headers={"Authorization": f"Bearer {await _token(client, 'r1@example.com')}"}
    )
    assert again.json()["id"] == first_res.json()["id"]
    assert again.json()["rating"] == 5


@pytest.mark.asyncio
async def test_review_rejects_an_unknown_dimension(client: AsyncClient, db_session):
    """Dimensions are validated so a typo cannot silently create a metric nobody aggregates."""
    persona = await _make_persona(db_session, "test_character_five")
    stylist = await _make_user(db_session, "r3@example.com", ROLE_STYLIST)
    db_session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))
    await db_session.commit()

    res = await client.put(
        f"/api/v1/personas/{persona.id}/outfits/does-not-matter/review",
        json={"rating": 4, "dimension_ratings": {"vibes": 5}},
        headers={"Authorization": f"Bearer {await _token(client, 'r3@example.com')}"},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_a_stylist_sees_only_their_own_characters(client: AsyncClient, db_session):
    mine = await _make_persona(db_session, "test_character_mine")
    await _make_persona(db_session, "test_character_theirs")
    stylist = await _make_user(db_session, "s6@example.com", ROLE_STYLIST)
    db_session.add(PersonaAssignment(persona_id=mine.id, stylist_user_id=stylist.id))
    await db_session.commit()

    res = await client.get(
        "/api/v1/personas",
        headers={"Authorization": f"Bearer {await _token(client, 's6@example.com')}"},
    )
    assert res.status_code == 200
    slugs = [p["slug"] for p in res.json()]
    assert slugs == ["test_character_mine"]


@pytest.mark.asyncio
async def test_an_admin_sees_every_character(client: AsyncClient, db_session):
    await _make_persona(db_session, "test_character_a")
    await _make_persona(db_session, "test_character_b")
    await _make_user(db_session, "admin@example.com", ROLE_ADMIN)

    res = await client.get(
        "/api/v1/personas",
        headers={"Authorization": f"Bearer {await _token(client, 'admin@example.com')}"},
    )
    assert res.status_code == 200
    assert len(res.json()) == 2
