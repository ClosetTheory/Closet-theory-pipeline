"""API dependencies for database, storage, auth and acting-as-character scope injection."""

from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Set
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.storage import get_storage_client, StorageClient


async def get_db_session(session: AsyncSession = Depends(get_db)) -> AsyncSession:
    return session


def get_storage() -> StorageClient:
    return get_storage_client()


async def get_current_user(
    authorization: str = Header(default=""),
    session: AsyncSession = Depends(get_db_session),
):
    from app.auth.security import verify_session_token
    from app.models.user import User

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header")

    user_id = verify_session_token(authorization.removeprefix("Bearer "))
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


# --- roles -----------------------------------------------------------------------------------
# Roles are read from the database on every request rather than carried in the session token.
# The token payload is only {sub, exp} and deliberately stays that way: a role baked into a
# 7-day token cannot be revoked for 7 days, and revoking a stylist's access to member wardrobes
# should take effect immediately.


async def get_user_roles(user, session: AsyncSession) -> Set[str]:
    from sqlalchemy import select
    from app.models.role import UserRole

    rows = await session.execute(select(UserRole.role).where(UserRole.user_id == user.id))
    return set(rows.scalars().all())


async def require_admin(
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    from app.models.role import ROLE_ADMIN

    if ROLE_ADMIN not in await get_user_roles(current_user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return current_user


async def require_stylist(
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """Admins pass this too — an admin reviewing alongside the stylists is a supported case, not
    a privilege escalation."""
    from app.models.role import ROLE_ADMIN, ROLE_STYLIST

    roles = await get_user_roles(current_user, session)
    if not roles & {ROLE_STYLIST, ROLE_ADMIN}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Stylist access required")
    return current_user


async def forbid_guest(
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
):
    """The guest ("vc") role tours a wardrobe that belongs to someone else. It may browse, ingest
    and run styling, but anything that destroys or formally judges that account's data — deleting
    a garment, leaving a stylist review or an own-outfit score — is refused server-side, so hiding
    the tab is not the only thing standing between a guest and the admin's records."""
    from app.models.role import ROLE_VC

    if ROLE_VC in await get_user_roles(current_user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Guest accounts cannot do this")
    return current_user


# --- acting as a character -------------------------------------------------------------------


@dataclass
class ActingScope:
    """Who the request is acting as, resolved once so nothing downstream has to decide.

    `actor` is always the authenticated human — a stylist or admin — and is what reviews and
    audit rows are attributed to. `tenant_id`/`member_id` are the *character's* when a character
    is in scope, which is what makes the existing wardrobe and styling endpoints operate on that
    character's data without any of them being aware characters exist.
    """

    actor: object                      # app.models.user.User
    tenant_id: str
    member_id: str
    persona_id: Optional[str] = None
    persona: Optional[object] = None   # app.models.persona.Persona
    roles: Set[str] = field(default_factory=set)

    @property
    def is_acting_as_persona(self) -> bool:
        return self.persona_id is not None


async def get_acting_scope(
    x_acting_as_persona: str = Header(default=""),
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ActingScope:
    """Resolves the scope a request operates in.

    Without the header the scope is the caller's own account, byte-identically to the behaviour
    before characters existed — which is what lets every existing client and the whole test suite
    keep working untouched.

    With the header, the caller must be an admin or hold an active assignment to that character.
    The tenant and member ids then come from the character's backing user row and never from the
    header, so a forged header cannot reach another character's wardrobe; the worst it can do is
    earn a 403. Note this runs *after* authentication, so it is never an auth bypass.
    """
    from app.models.persona import Persona, PersonaAssignment
    from app.models.role import ROLE_ADMIN, ROLE_STYLIST
    from app.models.user import User
    from sqlalchemy import select

    roles = await get_user_roles(current_user, session)
    persona_id = (x_acting_as_persona or "").strip()

    if not persona_id:
        # A stylist who forgets the header would otherwise ingest a character's wardrobe into
        # their own account and only discover it much later. Fail loudly instead. Admins and
        # ordinary members are unaffected: an admin has their own reasons to call these, and a
        # member acting on their own wardrobe is the normal case.
        if ROLE_STYLIST in roles and ROLE_ADMIN not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Stylist accounts must act as an assigned character. "
                    "Send the X-Acting-As-Persona header."
                ),
            )
        return ActingScope(
            actor=current_user,
            tenant_id=current_user.tenant_id,
            member_id=current_user.member_id,
            roles=roles,
        )

    persona = (
        await session.execute(select(Persona).where(Persona.id == persona_id))
    ).scalars().first()
    if not persona or persona.status == "archived":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")

    if ROLE_ADMIN not in roles:
        assignment = (
            await session.execute(
                select(PersonaAssignment).where(
                    PersonaAssignment.persona_id == persona.id,
                    PersonaAssignment.stylist_user_id == current_user.id,
                    PersonaAssignment.status == "active",
                )
            )
        ).scalars().first()
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This character is not assigned to you",
            )

    backing_user = await session.get(User, persona.user_id)
    if not backing_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Character has no backing account; re-run the seeder",
        )

    return ActingScope(
        actor=current_user,
        tenant_id=backing_user.tenant_id,
        member_id=backing_user.member_id,
        persona_id=persona.id,
        persona=persona,
        roles=roles,
    )
