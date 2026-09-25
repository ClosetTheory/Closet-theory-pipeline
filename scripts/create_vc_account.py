"""Create (or refresh) a guest account that tours the product on another account's wardrobe.

Why a script and not registration: a registered account gets its own empty tenant, and the point
of this one is the opposite — it must open onto a wardrobe that already has garments, outfits and
styling history in it, so a visitor sees the product working rather than an empty catalogue.
That is done by giving the guest the *same* `tenant_id`/`member_id` as the wardrobe owner:
every read and write in this codebase scopes on those two columns, so the guest sees exactly
what the owner sees, with no per-endpoint special-casing.

What the guest can and cannot do is the `vc` role (app/models/role.py): the navbar hides every
tab except Catalogue, Ingestion, Styling, Pipeline Info and Profile, and the API refuses garment
deletion and stylist reviews server-side (`forbid_guest` in app/api/dependencies.py).

Idempotent on email: re-running resets the password, re-points the wardrobe and re-asserts the
role, and strips admin/stylist roles the account may have picked up — a guest is never also an
admin.

    python -m scripts.create_vc_account                       # vc@closettheory.co on admin's wardrobe
    python -m scripts.create_vc_account --password 'something'
    python -m scripts.create_vc_account --wardrobe-of someone@closettheory.co
"""

import argparse
import asyncio
import os
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import hash_password
from app.database import AsyncSessionLocal, DEMO_USER_EMAIL
from app.models.base import generate_uuid
from app.models.role import ROLE_ADMIN, ROLE_STYLIST, ROLE_VC, UserRole
from app.models.user import User

DEFAULT_EMAIL = "vc@closettheory.co"
# Same shape as the other panel accounts (admin1234, stylist1234): simple on purpose, and
# overridable with --password or CT_VC_PASSWORD before this runs anywhere that matters.
DEFAULT_PASSWORD = "vc1234"
DEFAULT_DISPLAY_NAME = "VC"


async def ensure_vc_account(
    session: AsyncSession,
    email: str = DEFAULT_EMAIL,
    password: str = DEFAULT_PASSWORD,
    wardrobe_of: str = DEMO_USER_EMAIL,
    display_name: str = DEFAULT_DISPLAY_NAME,
) -> User:
    """Creates or refreshes the guest account. Flushes and commits."""
    owner: Optional[User] = (
        await session.execute(select(User).where(User.email == wardrobe_of.lower()))
    ).scalars().first()
    if owner is None:
        raise SystemExit(f"No account with email {wardrobe_of!r} to share a wardrobe from.")
    if owner.email.lower() == email.lower():
        raise SystemExit("The guest account cannot be the wardrobe owner itself.")

    guest = (await session.execute(select(User).where(User.email == email.lower()))).scalars().first()
    if guest is None:
        guest = User(
            id=generate_uuid("user"),
            email=email.lower(),
            display_name=display_name,
            password_hash=hash_password(password),
            tenant_id=owner.tenant_id,
            member_id=owner.member_id,
            gender=owner.gender,
        )
        session.add(guest)
        action = "created"
    else:
        guest.password_hash = hash_password(password)
        guest.tenant_id = owner.tenant_id
        guest.member_id = owner.member_id
        guest.display_name = display_name
        if not guest.gender:
            guest.gender = owner.gender
        action = "updated"
    await session.flush()

    roles = (await session.execute(select(UserRole).where(UserRole.user_id == guest.id))).scalars().all()
    have = {r.role for r in roles}
    for r in roles:
        if r.role in (ROLE_ADMIN, ROLE_STYLIST):
            await session.delete(r)
    if ROLE_VC not in have:
        session.add(UserRole(user_id=guest.id, role=ROLE_VC))
    await session.commit()

    print(f"{action}: {guest.email} (id={guest.id}) -> wardrobe of {owner.email} "
          f"(tenant={owner.tenant_id}, member={owner.member_id}), role={ROLE_VC}")
    return guest


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=os.environ.get("CT_VC_PASSWORD") or DEFAULT_PASSWORD)
    parser.add_argument("--wardrobe-of", default=DEMO_USER_EMAIL, help="email of the account whose wardrobe the guest tours")
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    args = parser.parse_args()
    async with AsyncSessionLocal() as session:
        await ensure_vc_account(session, args.email, args.password, args.wardrobe_of, args.display_name)


if __name__ == "__main__":
    asyncio.run(main())
