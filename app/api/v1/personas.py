"""Evaluation characters: the roster a stylist works through, and the scores they give.

Two audiences on one router. An admin manages the roster and who is assigned what; a stylist sees
only their own assigned characters and rates the outfits generated for them. The same handler
serves both wherever the only difference is the filter.

Ingestion and styling are deliberately absent here. A stylist builds a character's wardrobe
through the existing wardrobe endpoints while acting as that character — see
app/api/dependencies.py — so there is exactly one ingestion path in the codebase rather than a
second one that drifts.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user,
    get_db_session,
    get_storage,
    get_user_roles,
    require_admin,
    require_stylist,
)
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.persona import Persona, PersonaAssignment
from app.models.persona_review import PersonaOutfitReview
from app.models.role import ROLE_ADMIN
from app.models.styling import Outfit
from app.models.user import User
from app.schemas.persona import (
    AssignmentRequest,
    PersonaOutfitReviewRequest,
    PersonaOutfitReviewResult,
    PersonaRead,
)
from app.storage.base import StorageClient

router = APIRouter(prefix="/personas", tags=["Evaluation Characters"])


# --- helpers ---------------------------------------------------------------------------------


async def _assigned_persona_ids(session: AsyncSession, user_id: str) -> List[str]:
    rows = await session.execute(
        select(PersonaAssignment.persona_id).where(
            PersonaAssignment.stylist_user_id == user_id, PersonaAssignment.status == "active"
        )
    )
    return list(rows.scalars().all())


async def _load_persona_or_404(session: AsyncSession, persona_id: str) -> Persona:
    persona = (
        await session.execute(select(Persona).where(Persona.id == persona_id))
    ).scalars().first()
    if not persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")
    return persona


async def _require_access(session: AsyncSession, user, persona: Persona) -> set:
    """An admin reaches every character; a stylist only their active assignments.

    Mirrors the check in get_acting_scope on purpose — this router does not go through that
    dependency, because reading a character is not the same as acting as one.
    """
    roles = await get_user_roles(user, session)
    if ROLE_ADMIN in roles:
        return roles
    if persona.id not in await _assigned_persona_ids(session, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This character is not assigned to you"
        )
    return roles


async def _persona_counts(session: AsyncSession, persona_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Garment, outfit and review counts for a set of characters, in three queries rather than
    three per character — the roster page asks for all of them at once."""
    if not persona_ids:
        return {}
    personas = (
        await session.execute(select(Persona.id, Persona.user_id).where(Persona.id.in_(persona_ids)))
    ).all()
    by_user = {user_id: persona_id for persona_id, user_id in personas}
    out: Dict[str, Dict[str, Any]] = {
        pid: {"garment_count": 0, "outfit_count": 0, "review_count": 0, "average_rating": None}
        for pid, _ in personas
    }

    garments = await session.execute(
        select(Garment.tenant_id, func.count())
        .where(Garment.tenant_id.in_(list(by_user)))
        .group_by(Garment.tenant_id)
    )
    for tenant_id, count in garments.all():
        out[by_user[tenant_id]]["garment_count"] = count

    outfits = await session.execute(
        select(Outfit.tenant_id, func.count())
        .where(Outfit.tenant_id.in_(list(by_user)))
        .group_by(Outfit.tenant_id)
    )
    for tenant_id, count in outfits.all():
        out[by_user[tenant_id]]["outfit_count"] = count

    reviews = await session.execute(
        select(PersonaOutfitReview.persona_id, func.count(), func.avg(PersonaOutfitReview.rating))
        .where(PersonaOutfitReview.persona_id.in_(persona_ids))
        .group_by(PersonaOutfitReview.persona_id)
    )
    for persona_id, count, avg in reviews.all():
        out[persona_id]["review_count"] = count
        out[persona_id]["average_rating"] = round(float(avg), 2) if avg is not None else None

    return out


async def _assigned_names(session: AsyncSession, persona_ids: List[str]) -> Dict[str, List[str]]:
    if not persona_ids:
        return {}
    rows = await session.execute(
        select(PersonaAssignment.persona_id, User.display_name, User.email)
        .join(User, User.id == PersonaAssignment.stylist_user_id)
        .where(PersonaAssignment.persona_id.in_(persona_ids), PersonaAssignment.status == "active")
    )
    out: Dict[str, List[str]] = {}
    for persona_id, display_name, email in rows.all():
        out.setdefault(persona_id, []).append(display_name or email)
    return out


def _to_read(persona: Persona, counts: Dict[str, Any], stylists: List[str]) -> PersonaRead:
    return PersonaRead(
        persona_id=persona.id,
        slug=persona.slug,
        display_name=persona.display_name,
        age=persona.age,
        gender=persona.gender,
        city=persona.city,
        state_region=persona.state_region,
        climate_zone=persona.climate_zone,
        occupation=persona.occupation,
        body_shape=persona.body_shape,
        height_cm=persona.height_cm,
        height_band=persona.height_band,
        build=persona.build,
        skin_tone_monk=persona.skin_tone_monk,
        color_analysis=persona.color_analysis or {},
        hair=persona.hair or {},
        ethnic_wear=persona.ethnic_wear or {},
        preferences=persona.preferences or {},
        hard_constraints=persona.hard_constraints or [],
        fit_pain_points=persona.fit_pain_points or [],
        budget_tier=persona.budget_tier,
        bio=persona.bio,
        styling_notes=persona.styling_notes,
        rationale=persona.rationale,
        portrait_url=f"/api/v1/personas/{persona.id}/portrait" if persona.portrait_image_id else None,
        status=persona.status,
        assigned_stylists=stylists,
        **counts,
    )


# --- reading the roster ----------------------------------------------------------------------


@router.get("", response_model=List[PersonaRead])
async def list_personas(
    mine_only: bool = Query(default=False, description="stylists always see only their own"),
    current_user: User = Depends(require_stylist),
    session: AsyncSession = Depends(get_db_session),
):
    """Admins see the whole roster; stylists see only what is assigned to them."""
    roles = await get_user_roles(current_user, session)
    stmt = select(Persona).where(Persona.status != "archived").order_by(Persona.display_name)

    if ROLE_ADMIN not in roles or mine_only:
        assigned = await _assigned_persona_ids(session, current_user.id)
        if not assigned:
            return []
        stmt = stmt.where(Persona.id.in_(assigned))

    personas = list((await session.execute(stmt)).scalars().all())
    ids = [p.id for p in personas]
    counts = await _persona_counts(session, ids)
    stylists = await _assigned_names(session, ids)
    return [_to_read(p, counts.get(p.id, {}), stylists.get(p.id, [])) for p in personas]


@router.get("/{persona_id}", response_model=PersonaRead)
async def get_persona_detail(
    persona_id: str,
    current_user: User = Depends(require_stylist),
    session: AsyncSession = Depends(get_db_session),
):
    persona = await _load_persona_or_404(session, persona_id)
    await _require_access(session, current_user, persona)
    counts = await _persona_counts(session, [persona.id])
    stylists = await _assigned_names(session, [persona.id])
    return _to_read(persona, counts.get(persona.id, {}), stylists.get(persona.id, []))


@router.get("/{persona_id}/portrait")
async def get_persona_portrait(
    persona_id: str,
    current_user: User = Depends(require_stylist),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    persona = await _load_persona_or_404(session, persona_id)
    await _require_access(session, current_user, persona)
    if not persona.portrait_image_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No portrait generated yet")

    asset = await session.get(ImageAsset, persona.portrait_image_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portrait asset missing")
    data = await storage.get_object(asset.object_uri)
    # Portraits are immutable once generated, so they cache hard.
    return Response(content=data, media_type=asset.mime_type, headers={"Cache-Control": "public, max-age=86400"})


# --- reviewing the outfits generated for a character -------------------------------------------


def _review_to_result(review: PersonaOutfitReview, reviewer_name: Optional[str] = None) -> PersonaOutfitReviewResult:
    return PersonaOutfitReviewResult(
        id=review.id,
        outfit_id=review.outfit_id,
        persona_id=review.persona_id,
        reviewer_user_id=review.reviewer_user_id,
        reviewer_name=reviewer_name,
        rating=review.rating,
        vote=review.vote,
        dimension_ratings=review.dimension_ratings or {},
        comment=review.comment,
        would_wear=review.would_wear,
        tags=review.tags or [],
        created_at=review.created_at.isoformat(),
        updated_at=review.updated_at.isoformat(),
    )


@router.get("/{persona_id}/outfits")
async def list_persona_outfits(
    persona_id: str,
    limit: int = Query(default=50, le=200),
    current_user: User = Depends(require_stylist),
    session: AsyncSession = Depends(get_db_session),
):
    """Every outfit generated for this character, newest first, with my review and everyone
    else's alongside it — so a stylist can see where they disagree with a colleague."""
    from app.styling.replay import build_outfit_result

    persona = await _load_persona_or_404(session, persona_id)
    await _require_access(session, current_user, persona)

    outfits = list((
        await session.execute(
            select(Outfit)
            .where(Outfit.tenant_id == persona.user_id)
            .order_by(Outfit.created_at.desc())
            .limit(limit)
        )
    ).scalars().all())
    if not outfits:
        return []

    outfit_ids = [o.id for o in outfits]
    review_rows = (
        await session.execute(
            select(PersonaOutfitReview, User.display_name, User.email)
            .join(User, User.id == PersonaOutfitReview.reviewer_user_id)
            .where(PersonaOutfitReview.outfit_id.in_(outfit_ids))
        )
    ).all()
    by_outfit: Dict[str, List[PersonaOutfitReviewResult]] = {}
    for review, display_name, email in review_rows:
        by_outfit.setdefault(review.outfit_id, []).append(
            _review_to_result(review, display_name or email)
        )

    items = []
    for outfit in outfits:
        reviews = by_outfit.get(outfit.id, [])
        items.append({
            "outfit": (await build_outfit_result(session, outfit)).model_dump(mode="json"),
            "generated_at": outfit.created_at.isoformat(),
            "my_review": next(
                (r.model_dump(mode="json") for r in reviews if r.reviewer_user_id == current_user.id),
                None,
            ),
            "reviews": [r.model_dump(mode="json") for r in reviews],
        })
    return items


@router.put("/{persona_id}/outfits/{outfit_id}/review", response_model=PersonaOutfitReviewResult)
async def upsert_persona_outfit_review(
    persona_id: str,
    outfit_id: str,
    request: PersonaOutfitReviewRequest,
    current_user: User = Depends(require_stylist),
    session: AsyncSession = Depends(get_db_session),
):
    """One review per (outfit, reviewer), updated in place on a re-submit.

    Keyed on the reviewer rather than the outfit alone, which is the whole reason this does not
    reuse `stylist_reviews`: three stylists scoring the same outfit is the point of the panel.
    """
    persona = await _load_persona_or_404(session, persona_id)
    await _require_access(session, current_user, persona)

    outfit = await session.get(Outfit, outfit_id)
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Outfit not found")
    if outfit.tenant_id != persona.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="That outfit belongs to another character"
        )

    review = (
        await session.execute(
            select(PersonaOutfitReview).where(
                PersonaOutfitReview.outfit_id == outfit_id,
                PersonaOutfitReview.reviewer_user_id == current_user.id,
            )
        )
    ).scalars().first()

    if review is None:
        review = PersonaOutfitReview(
            outfit_id=outfit_id,
            persona_id=persona.id,
            reviewer_user_id=current_user.id,
            tenant_id=persona.user_id,
            member_id=persona.user_id,
            rating=request.rating,
        )
        session.add(review)

    review.rating = request.rating
    review.vote = request.vote
    review.dimension_ratings = request.dimension_ratings
    review.comment = request.comment
    review.would_wear = request.would_wear
    review.tags = request.tags
    await session.commit()
    await session.refresh(review)
    return _review_to_result(review, current_user.display_name or current_user.email)


# --- admin: roster and assignment --------------------------------------------------------------


@router.post("/{persona_id}/assignments", status_code=status.HTTP_201_CREATED)
async def assign_persona(
    persona_id: str,
    request: AssignmentRequest,
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """Assigns a character to a stylist. Exclusive by default: two stylists independently
    building the same wardrobe is nearly always an accident rather than an intent."""
    persona = await _load_persona_or_404(session, persona_id)
    stylist = await session.get(User, request.stylist_user_id)
    if not stylist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stylist not found")

    if request.exclusive:
        others = (
            await session.execute(
                select(PersonaAssignment).where(
                    PersonaAssignment.persona_id == persona.id,
                    PersonaAssignment.stylist_user_id != stylist.id,
                    PersonaAssignment.status == "active",
                )
            )
        ).scalars().all()
        for other in others:
            other.status = "revoked"
            other.revoked_at = func.now()

    existing = (
        await session.execute(
            select(PersonaAssignment).where(
                PersonaAssignment.persona_id == persona.id,
                PersonaAssignment.stylist_user_id == stylist.id,
            )
        )
    ).scalars().first()
    if existing:
        existing.status = "active"
        existing.revoked_at = None
    else:
        session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))

    await session.commit()
    return {"persona_id": persona.id, "stylist_user_id": stylist.id, "status": "active"}


@router.delete("/{persona_id}/assignments/{stylist_user_id}", status_code=status.HTTP_200_OK)
async def revoke_persona_assignment(
    persona_id: str,
    stylist_user_id: str,
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """Revokes rather than deletes, so "who was working on this in August" stays answerable."""
    assignment = (
        await session.execute(
            select(PersonaAssignment).where(
                PersonaAssignment.persona_id == persona_id,
                PersonaAssignment.stylist_user_id == stylist_user_id,
            )
        )
    ).scalars().first()
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment.status = "revoked"
    await session.commit()
    return {"persona_id": persona_id, "stylist_user_id": stylist_user_id, "status": "revoked"}


@router.post("/{persona_id}/portrait", status_code=status.HTTP_200_OK)
async def regenerate_persona_portrait(
    persona_id: str,
    force: bool = Query(default=False),
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    from app.personas.portraits import generate_and_persist_portrait

    persona = await _load_persona_or_404(session, persona_id)
    image_id, result = await generate_and_persist_portrait(session, storage, persona, force=force)
    await session.commit()
    if result == "failed":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Portrait generation failed"
        )
    return {"persona_id": persona.id, "portrait_image_id": image_id, "result": result}
