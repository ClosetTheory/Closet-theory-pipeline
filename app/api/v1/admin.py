"""Administration: accounts, roles and the character/stylist assignment matrix.

Everything here requires the admin role. Roles live in `user_roles` rather than a column on
`users` — see app/models/role.py for why that distinction is load-bearing in a project with no
migrations.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session, require_admin
from app.auth.security import hash_password
from app.models.base import generate_uuid
from app.models.garment import Garment
from app.models.persona import Persona, PersonaAssignment
from app.models.persona_review import PersonaOutfitReview
from app.models.role import KNOWN_ROLES, ROLE_STYLIST, UserRole
from app.models.styling import Outfit
from app.models.user import User
from app.schemas.persona import UserListItem

router = APIRouter(prefix="/admin", tags=["Administration"])


@router.get("/users", response_model=List[UserListItem])
async def list_users(
    role: str = Query(default="", description="filter to users holding this role"),
    include_personas: bool = Query(
        default=False, description="include the synthetic accounts backing evaluation characters"
    ),
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """Lists accounts. Character accounts are excluded by default, so the list answers "who are
    our people" rather than being drowned by twenty-two synthetic members.

    `is_persona` is derived from the presence of a `personas` row rather than a flag on the user,
    so it cannot drift out of step with reality. Password hashes are never returned.
    """
    users = list((await session.execute(select(User).order_by(User.email))).scalars().all())
    roles_rows = (await session.execute(select(UserRole.user_id, UserRole.role))).all()
    roles_by_user: Dict[str, List[str]] = {}
    for user_id, role_name in roles_rows:
        roles_by_user.setdefault(user_id, []).append(role_name)

    persona_rows = (await session.execute(select(Persona.user_id, Persona.slug))).all()
    persona_by_user = {user_id: slug for user_id, slug in persona_rows}

    items: List[UserListItem] = []
    for user in users:
        is_persona = user.id in persona_by_user
        if is_persona and not include_personas:
            continue
        user_roles = sorted(roles_by_user.get(user.id, []))
        if role and role not in user_roles:
            continue
        items.append(UserListItem(
            user_id=user.id,
            email=user.email,
            display_name=user.display_name,
            gender=user.gender,
            roles=user_roles,
            is_persona=is_persona,
            persona_slug=persona_by_user.get(user.id),
        ))
    return items


@router.post("/users/{user_id}/roles", status_code=status.HTTP_201_CREATED)
async def grant_role(
    user_id: str,
    role: str = Body(..., embed=True),
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    if role not in KNOWN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown role {role!r}; valid: {sorted(KNOWN_ROLES)}",
        )
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    existing = (
        await session.execute(
            select(UserRole).where(UserRole.user_id == user_id, UserRole.role == role)
        )
    ).scalars().first()
    if not existing:
        session.add(UserRole(user_id=user_id, role=role, granted_by_user_id=admin.id))
        await session.commit()
    return {"user_id": user_id, "role": role, "granted": True}


@router.delete("/users/{user_id}/roles/{role}", status_code=status.HTTP_200_OK)
async def revoke_role(
    user_id: str,
    role: str,
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    if user_id == admin.id and role == "admin":
        # Removing your own last admin role locks everyone out of this router, and the only way
        # back is the ADMIN_EMAILS bootstrap plus a restart.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refusing to revoke your own admin role — ask another admin to do it",
        )
    existing = (
        await session.execute(
            select(UserRole).where(UserRole.user_id == user_id, UserRole.role == role)
        )
    ).scalars().first()
    if existing:
        await session.delete(existing)
        await session.commit()
    return {"user_id": user_id, "role": role, "granted": False}


@router.post("/stylists", status_code=status.HTTP_201_CREATED)
async def create_stylist(
    email: str = Body(...),
    display_name: str = Body(default=""),
    password: str = Body(...),
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """Creates an account and grants the stylist role in one call, because doing it in two leaves
    a window where the account exists but cannot see anything."""
    existing = (await session.execute(select(User).where(User.email == email))).scalars().first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="That email already exists")

    user_id = generate_uuid("user")
    user = User(
        id=user_id,
        email=email,
        display_name=display_name or email.split("@")[0],
        password_hash=hash_password(password),
        tenant_id=user_id,
        member_id=user_id,
    )
    session.add(user)
    session.add(UserRole(user_id=user_id, role=ROLE_STYLIST, granted_by_user_id=admin.id))
    await session.commit()
    return {"user_id": user_id, "email": email, "roles": [ROLE_STYLIST]}


@router.get("/assignments")
async def assignment_matrix(
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """The whole character x stylist picture in one response, which is what the panel renders."""
    rows = (
        await session.execute(
            select(
                Persona.id, Persona.slug, Persona.display_name, Persona.city, Persona.status,
                PersonaAssignment.stylist_user_id, PersonaAssignment.status,
                User.display_name, User.email,
            )
            .select_from(Persona)
            .outerjoin(
                PersonaAssignment,
                (PersonaAssignment.persona_id == Persona.id) & (PersonaAssignment.status == "active"),
            )
            .outerjoin(User, User.id == PersonaAssignment.stylist_user_id)
            .order_by(Persona.display_name)
        )
    ).all()

    personas: Dict[str, Dict[str, Any]] = {}
    for pid, slug, name, city, p_status, stylist_id, a_status, s_name, s_email in rows:
        entry = personas.setdefault(pid, {
            "persona_id": pid, "slug": slug, "display_name": name,
            "city": city, "status": p_status, "stylists": [],
        })
        if stylist_id:
            entry["stylists"].append({
                "stylist_user_id": stylist_id, "name": s_name or s_email, "email": s_email,
            })

    stylists = (
        await session.execute(
            select(User.id, User.display_name, User.email)
            .join(UserRole, UserRole.user_id == User.id)
            .where(UserRole.role == ROLE_STYLIST)
            .order_by(User.email)
        )
    ).all()

    return {
        "personas": list(personas.values()),
        "stylists": [
            {"stylist_user_id": sid, "name": name or email, "email": email}
            for sid, name, email in stylists
        ],
    }


@router.get("/personas/{persona_id}/summary")
async def persona_summary(
    persona_id: str,
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """Where a character has got to: how much wardrobe exists, how far through the pipeline it
    is, how many outfits came out and what the stylists made of them."""
    persona = (
        await session.execute(select(Persona).where(Persona.id == persona_id))
    ).scalars().first()
    if not persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")

    by_state = (
        await session.execute(
            select(Garment.status, func.count())
            .where(Garment.tenant_id == persona.user_id)
            .group_by(Garment.status)
        )
    ).all()
    outfit_count = (
        await session.execute(
            select(func.count()).select_from(Outfit).where(Outfit.tenant_id == persona.user_id)
        )
    ).scalar_one()
    review_stats = (
        await session.execute(
            select(func.count(), func.avg(PersonaOutfitReview.rating))
            .where(PersonaOutfitReview.persona_id == persona_id)
        )
    ).first()
    would_wear = (
        await session.execute(
            select(func.count()).where(
                PersonaOutfitReview.persona_id == persona_id,
                PersonaOutfitReview.would_wear.is_(True),
            )
        )
    ).scalar_one()

    review_count, avg_rating = review_stats or (0, None)
    return {
        "persona_id": persona_id,
        "slug": persona.slug,
        "display_name": persona.display_name,
        "garments_by_state": {state: count for state, count in by_state},
        "garment_total": sum(count for _, count in by_state),
        "outfit_count": outfit_count,
        "review_count": review_count or 0,
        "average_rating": round(float(avg_rating), 2) if avg_rating is not None else None,
        "would_wear_count": would_wear or 0,
    }
