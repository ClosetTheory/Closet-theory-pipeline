"""GET /admin/metrics/stylists — admin only, and the rows it loads feed the pure aggregator."""

import pytest
from httpx import AsyncClient

from app.models.base import generate_uuid
from app.models.persona import Persona, PersonaAssignment
from app.models.persona_review import PersonaOutfitReview
from app.models.role import ROLE_ADMIN, ROLE_STYLIST, UserRole
from app.models.styling import Outfit, StylingRequest
from app.models.user import User


async def _make_user(session, email: str, role: str | None = None) -> User:
    from app.auth.security import hash_password
    user_id = generate_uuid("user")
    user = User(id=user_id, email=email, display_name=email.split("@")[0],
                password_hash=hash_password("testpassword123"), tenant_id=user_id, member_id=user_id)
    session.add(user)
    if role:
        session.add(UserRole(user_id=user_id, role=role))
    await session.commit()
    return user


async def _make_persona(session, slug: str) -> Persona:
    backing_id = generate_uuid("user")
    session.add(User(id=backing_id, email=f"{slug}@personas.closettheory.local", display_name=slug,
                     password_hash="!locked-persona-account", gender="women", tenant_id=backing_id, member_id=backing_id))
    persona = Persona(slug=slug, user_id=backing_id, display_name=slug.replace("_", " ").title(),
                      gender="women", city="mumbai", body_shape="rectangle", bio="test character")
    session.add(persona)
    await session.commit()
    return persona


async def _outfit_for(session, persona: Persona) -> Outfit:
    request = StylingRequest(tenant_id=persona.user_id, member_id=persona.user_id, raw_text="metrics test")
    session.add(request)
    await session.flush()
    outfit = Outfit(request_id=request.id, tenant_id=persona.user_id, member_id=persona.user_id, rank=1, final_score=0.7)
    session.add(outfit)
    await session.commit()
    return outfit


async def _token(client: AsyncClient, email: str) -> str:
    res = await client.post("/api/v1/auth/login", json={"email": email, "password": "testpassword123"})
    return res.json()["token"]


@pytest.mark.asyncio
async def test_stylists_are_refused_and_admins_get_the_matrix(client: AsyncClient, db_session):
    admin = await _make_user(db_session, "admin_m@example.com", ROLE_ADMIN)
    stylist = await _make_user(db_session, "stylist_m@example.com", ROLE_STYLIST)
    persona = await _make_persona(db_session, "metrics_character")
    db_session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))
    await db_session.commit()
    o1 = await _outfit_for(db_session, persona)
    await _outfit_for(db_session, persona)
    db_session.add(PersonaOutfitReview(
        outfit_id=o1.id, persona_id=persona.id, reviewer_user_id=stylist.id, tenant_id=persona.user_id,
        member_id=persona.user_id, rating=4, vote="like", dimension_ratings={"colour_harmony": 5},
        comment="nice", would_wear=True, tags=["great_pairing"],
    ))
    await db_session.commit()

    # By decision, stylists do not see their own metrics.
    denied = await client.get("/api/v1/admin/metrics/stylists",
                              headers={"Authorization": f"Bearer {await _token(client, 'stylist_m@example.com')}"})
    assert denied.status_code == 403

    res = await client.get("/api/v1/admin/metrics/stylists?weekly_target=20",
                           headers={"Authorization": f"Bearer {await _token(client, 'admin_m@example.com')}"})
    assert res.status_code == 200, res.text
    m = res.json()
    assert m["weekly_target_per_character"] == 20
    assert m["dimension_keys"] == ["colour_harmony", "fit_and_silhouette", "occasion_fit", "persona_fit"]

    s = next(x for x in m["stylists"] if x["stylist_user_id"] == stylist.id)
    assert s["characters_assigned"] == 1 and s["outfits_available"] == 2 and s["outfits_reviewed"] == 1
    assert s["coverage_pct"] == 50.0 and s["backlog"] == 1
    assert s["reviews_window"] == 1 and s["weekly_target"] == 20 and s["weekly_progress_pct"] == 5.0
    assert s["with_comment_pct"] == 100.0 and s["with_chips_pct"] == 100.0 and s["would_wear_pct"] == 100.0

    c = next(x for x in m["characters"] if x["persona_id"] == persona.id)
    assert c["assigned_to"] == ["stylist_m"] and c["outfits_total"] == 2 and c["backlog"] == 1
    assert c["dimension_averages"]["colour_harmony"] == 5.0
    assert c["median_hours_to_first_review"] is not None

    cell = next(x for x in m["cells"] if x["stylist_user_id"] == stylist.id and x["persona_id"] == persona.id)
    assert cell["assigned"] is True and cell["outfits_reviewed"] == 1 and cell["avg_rating"] == 4.0
    assert admin.id not in {x["stylist_user_id"] for x in m["stylists"]}, "an admin with no activity is not a stylist row"
