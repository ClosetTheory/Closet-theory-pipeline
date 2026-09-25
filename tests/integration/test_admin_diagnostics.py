"""GET /admin/diagnostics — admin only; reports loop lag, host load and process state, reads only."""

import pytest
from httpx import AsyncClient

from app.models.base import generate_uuid
from app.models.role import ROLE_ADMIN, ROLE_STYLIST, UserRole
from app.models.user import User


async def _make_user(session, email: str, role: str) -> User:
    from app.auth.security import hash_password
    user_id = generate_uuid("user")
    user = User(id=user_id, email=email, display_name=email.split("@")[0],
                password_hash=hash_password("testpassword123"), tenant_id=user_id, member_id=user_id)
    session.add(user)
    session.add(UserRole(user_id=user_id, role=role))
    await session.commit()
    return user


async def _token(client: AsyncClient, email: str) -> str:
    res = await client.post("/api/v1/auth/login", json={"email": email, "password": "testpassword123"})
    return res.json()["token"]


@pytest.mark.asyncio
async def test_diagnostics_is_admin_only_and_measures_the_loop(client: AsyncClient, db_session):
    await _make_user(db_session, "admin_d@example.com", ROLE_ADMIN)
    await _make_user(db_session, "stylist_d@example.com", ROLE_STYLIST)

    denied = await client.get("/api/v1/admin/diagnostics",
                              headers={"Authorization": f"Bearer {await _token(client, 'stylist_d@example.com')}"})
    assert denied.status_code == 403

    res = await client.get("/api/v1/admin/diagnostics?lag_samples=3",
                           headers={"Authorization": f"Bearer {await _token(client, 'admin_d@example.com')}"})
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["event_loop"]["samples"] == 3 and d["event_loop"]["lag_ms_max"] >= 0
    assert d["process"]["pid"] > 0 and d["process"]["asyncio_tasks"] >= 1
    assert "use_in_memory_queue" in d["settings"] and "worker_concurrency" in d["settings"]
    assert "cpu_count" in d["host"]  # loadavg / meminfo may be absent off-Linux; keys still exist
    assert set(d.keys()) == {"now", "process", "event_loop", "host", "container_cgroup", "settings"}
