"""The guest ("vc") account: shares the owner's wardrobe, carries the vc role, and is refused the
destructive and judging endpoints server-side regardless of what the navbar hides."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.auth.security import hash_password
from app.models.base import generate_uuid
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.role import ROLE_ADMIN, ROLE_VC, UserRole
from app.models.user import User
from scripts.create_vc_account import ensure_vc_account

OWNER_EMAIL = "owner@closettheory.local"


async def _owner_with_one_garment(session) -> User:
    uid = generate_uuid("user")
    owner = User(id=uid, email=OWNER_EMAIL, display_name="Owner", password_hash=hash_password("testpassword123"),
                 tenant_id=uid, member_id=uid, gender="women")
    session.add(owner)
    asset = ImageAsset(id=f"img_{uid}", tenant_id=uid, member_id=uid, object_uri="raw/x.jpg", mime_type="image/jpeg",
                       width=10, height=10, sha256=f"sha_{uid}")
    session.add(asset)
    session.add(Garment(id=f"g_{uid}", tenant_id=uid, member_id=uid, source_image_id=asset.id, category="TOP",
                        subcategory="SHIRT", status="COMPLETED", quality_status="APPROVED", attributes_json={"colour": ["white"]}))
    await session.commit()
    return owner


async def _login(client: AsyncClient, email: str, password: str) -> dict:
    res = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['token']}"}


@pytest.mark.asyncio
async def test_guest_shares_the_owners_wardrobe_and_carries_only_the_vc_role(client: AsyncClient, db_session):
    owner = await _owner_with_one_garment(db_session)
    guest = await ensure_vc_account(db_session, "vc@closettheory.co", "vc1234", OWNER_EMAIL)
    assert (guest.tenant_id, guest.member_id) == (owner.tenant_id, owner.member_id)

    headers = await _login(client, "vc@closettheory.co", "vc1234")
    me = (await client.get("/api/v1/auth/me", headers=headers)).json()
    assert me["roles"] == [ROLE_VC] and me["display_name"] == "VC"

    garments = (await client.get("/api/v1/wardrobe/garments", headers=headers)).json()
    assert [g["garment_id"] for g in garments] == [f"g_{owner.id}"]


@pytest.mark.asyncio
async def test_guest_is_refused_deletion_and_reviews_server_side(client: AsyncClient, db_session):
    owner = await _owner_with_one_garment(db_session)
    await ensure_vc_account(db_session, "vc@closettheory.co", "vc1234", OWNER_EMAIL)
    headers = await _login(client, "vc@closettheory.co", "vc1234")
    gid = f"g_{owner.id}"

    assert (await client.delete(f"/api/v1/wardrobe/garments/{gid}", headers=headers)).status_code == 403
    assert (await client.put("/api/v1/wardrobe/styling/outfits/any/review", json={"vote": "like"}, headers=headers)).status_code == 403
    assert (await client.put("/api/v1/wardrobe/styling/outfits/any/score", json={"rating": 4}, headers=headers)).status_code == 403
    assert (await client.get("/api/v1/admin/users", headers=headers)).status_code == 403
    # and the garment is still there — the owner's wardrobe was never touched
    assert await db_session.get(Garment, gid) is not None

    # the owner is unaffected: the same delete succeeds for them
    owner_headers = await _login(client, OWNER_EMAIL, "testpassword123")
    assert (await client.delete(f"/api/v1/wardrobe/garments/{gid}", headers=owner_headers)).status_code == 204


@pytest.mark.asyncio
async def test_rerunning_resets_password_and_strips_admin(client: AsyncClient, db_session):
    await _owner_with_one_garment(db_session)
    guest = await ensure_vc_account(db_session, "vc@closettheory.co", "first", OWNER_EMAIL)
    db_session.add(UserRole(user_id=guest.id, role=ROLE_ADMIN))
    await db_session.commit()

    again = await ensure_vc_account(db_session, "vc@closettheory.co", "second", OWNER_EMAIL)
    assert again.id == guest.id
    roles = set((await db_session.execute(select(UserRole.role).where(UserRole.user_id == guest.id))).scalars().all())
    assert roles == {ROLE_VC}
    assert (await client.post("/api/v1/auth/login", json={"email": "vc@closettheory.co", "password": "first"})).status_code != 200
    await _login(client, "vc@closettheory.co", "second")


@pytest.mark.asyncio
async def test_guest_cannot_be_the_owner_or_share_a_missing_account(db_session):
    await _owner_with_one_garment(db_session)
    with pytest.raises(SystemExit):
        await ensure_vc_account(db_session, OWNER_EMAIL, "x", OWNER_EMAIL)
    with pytest.raises(SystemExit):
        await ensure_vc_account(db_session, "vc@closettheory.co", "x", "nobody@closettheory.local")
