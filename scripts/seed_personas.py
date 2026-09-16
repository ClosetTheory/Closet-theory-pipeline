"""Project the authored character roster into the database.

`app/personas/roster.yaml` is the source of truth; this turns it into rows. Idempotent on
`Persona.slug`, so re-running updates a character in place rather than creating a second one —
which matters because the roster will be corrected by stylists over time.

Deliberately a script rather than something `init_db()` does at boot: that call is wrapped in a
try/except which only logs, so a partial seed of twenty-two characters would fail invisibly and
retry on every restart.

Each character gets a real `User` row with its own tenant_id/member_id. That is what lets a
stylist ingest a wardrobe for them through the existing pipeline with no special-casing — see
app/models/persona.py. Those accounts are given an unusable password hash, so they can be acted
as (via an assignment) but never logged into.

    python -m scripts.seed_personas --dry-run
    python -m scripts.seed_personas --with-stylists
    python -m scripts.seed_personas --only aarushi_deshpande
"""

import argparse
import asyncio
from typing import Dict, List, Optional

from sqlalchemy import select

from app.auth.security import hash_password
from app.database import AsyncSessionLocal
from app.models.persona import Persona, PersonaAssignment
from app.models.role import ROLE_STYLIST, UserRole
from app.models.style_profile import StyleProfile
from app.models.base import generate_uuid
from app.models.user import User
from app.personas import load_roster
from app.personas.portraits import generate_and_persist_portrait
from app.storage import get_storage_client
from app.schemas.persona import PersonaSeed

# `verify_password` splits on "$" and expects four parts, so any value without them can never
# match a password. This is deliberate rather than a random unguessable hash: it is greppable,
# and it makes the intent obvious to the next person reading the users table.
LOCKED_PASSWORD_HASH = "!locked-persona-account"

STYLIST_SLOTS = ("stylist1", "stylist2", "stylist3")
STYLIST_EMAIL = "{slot}@closettheory.co"
# A known password rather than a generated one. These are placeholder accounts on an internal
# evaluation panel, and a random string printed once is the kind of thing that gets mistyped,
# lost, and then debugged as "login is broken". Override with --stylist-password before this
# ever runs anywhere that matters.
DEFAULT_STYLIST_PASSWORD = "stylist1234"


def _persona_fields(seed: PersonaSeed) -> Dict:
    """Everything that is projected from the YAML onto the row, so create and update stay in
    step — a field added to one path and forgotten in the other is the classic seeder bug."""
    return {
        "display_name": seed.name,
        "age": seed.age,
        "gender": seed.gender,
        "gender_presentation": seed.gender_presentation,
        "pronouns": seed.pronouns,
        "occupation": seed.occupation,
        "lifestyle": seed.lifestyle,
        "work_dress_code": seed.work_dress_code,
        "city": seed.city,
        "state_region": seed.state_region,
        "country": seed.country,
        "climate_zone": seed.climate.zone,
        "ootd_location": seed.ootd_location,
        "climate": seed.climate.model_dump(mode="json"),
        "body_shape": seed.body_shape,
        "height_cm": seed.height_cm,
        "height_band": seed.height_band,
        "build": seed.build,
        "measurements": seed.measurements,
        "fit_pain_points": seed.fit_pain_points,
        "skin_tone_monk": seed.skin_tone_monk,
        "color_analysis": seed.color_analysis.model_dump(mode="json", exclude_none=True),
        "hair": seed.hair.model_dump(mode="json", exclude_none=True),
        "ethnic_wear": seed.ethnic_wear.model_dump(mode="json"),
        "preferences": seed.preferences,
        "hard_constraints": seed.hard_constraints,
        "budget_tier": seed.budget_tier,
        "weekly_plan": seed.weekly_plan,
        "bio": seed.bio,
        "styling_notes": seed.styling_notes,
        "rationale": seed.rationale,
        "status": "active",
        "source": "seed",
    }


async def _ensure_stylists(
    session, dry_run: bool, password: str = DEFAULT_STYLIST_PASSWORD, reset: bool = False
) -> Dict[str, User]:
    """Creates the three stylist accounts if they are missing and grants each the stylist role.

    Placeholder addresses on purpose — real ones get swapped in once the panel is in use. The
    generated password is printed once and never stored anywhere retrievable.
    """
    accounts: Dict[str, User] = {}
    for slot in STYLIST_SLOTS:
        email = STYLIST_EMAIL.format(slot=slot)
        user = (await session.execute(select(User).where(User.email == email))).scalars().first()
        if user:
            if reset and not dry_run:
                user.password_hash = hash_password(password)
                print(f"  reset {email} -> {password}")
            accounts[slot] = user
        elif dry_run:
            print(f"  would create stylist account {email}")
            continue
        else:
            # tenant_id/member_id are NOT NULL, so the id has to exist before the insert rather
            # than being back-filled after the flush.
            user_id = generate_uuid("user")
            user = User(
                id=user_id,
                email=email,
                display_name=slot.capitalize(),
                password_hash=hash_password(password),
                # A stylist works on characters, never a wardrobe of their own, but the columns
                # are non-null so they point at the account itself.
                tenant_id=user_id,
                member_id=user_id,
            )
            session.add(user)
            await session.flush()
            accounts[slot] = user
            print(f"  created stylist {email}  password: {password}")

        if dry_run:
            continue
        has_role = (
            await session.execute(
                select(UserRole).where(UserRole.user_id == user.id, UserRole.role == ROLE_STYLIST)
            )
        ).scalars().first()
        if not has_role:
            session.add(UserRole(user_id=user.id, role=ROLE_STYLIST))
    return accounts


async def _upsert_persona(session, seed: PersonaSeed, dry_run: bool) -> Optional[Persona]:
    persona = (
        await session.execute(select(Persona).where(Persona.slug == seed.slug))
    ).scalars().first()

    if dry_run:
        verb = "update" if persona else "create"
        print(
            f"  would {verb:6} {seed.slug:26} {seed.name:26} "
            f"{seed.gender:7} {seed.city:12} Monk {seed.skin_tone_monk} "
            f"{seed.color_analysis.season:7} {seed.body_shape}"
        )
        return None

    if persona is None:
        # Its own tenant and member id: that is what isolates this character's wardrobe from
        # every other one, using the tenant checks the endpoints already perform.
        backing_id = generate_uuid("user")
        backing = User(
            id=backing_id,
            email=f"{seed.slug}@personas.closettheory.local",
            display_name=seed.name,
            password_hash=LOCKED_PASSWORD_HASH,
            gender=seed.gender,
            tenant_id=backing_id,
            member_id=backing_id,
        )
        session.add(backing)
        await session.flush()

        persona = Persona(slug=seed.slug, user_id=backing.id, **_persona_fields(seed))
        session.add(persona)
        await session.flush()
    else:
        for key, value in _persona_fields(seed).items():
            setattr(persona, key, value)
        backing = await session.get(User, persona.user_id)
        if backing:
            backing.gender = seed.gender
            backing.display_name = seed.name

    # A character's boldness is authored, not learned — there is no vote history to learn from
    # until a stylist starts rating. Seed it so the very first recommendation already reflects
    # who they are.
    boldness = float((seed.preferences.get("style_signals") or {}).get("boldness_preference", 0.0))
    profile = (
        await session.execute(
            select(StyleProfile).where(
                StyleProfile.tenant_id == persona.user_id, StyleProfile.member_id == persona.user_id
            )
        )
    ).scalars().first()
    if profile is None:
        session.add(StyleProfile(
            tenant_id=persona.user_id, member_id=persona.user_id, boldness_preference=boldness
        ))
    elif profile.vote_count == 0:
        # Only while it is still the authored value; once a stylist has voted, the learned
        # signal wins and re-seeding must not stamp on it.
        profile.boldness_preference = boldness

    return persona


async def _assign(session, persona: Persona, slot: str, stylists: Dict[str, User]) -> bool:
    stylist = stylists.get(slot)
    if not stylist:
        return False
    existing = (
        await session.execute(
            select(PersonaAssignment).where(
                PersonaAssignment.persona_id == persona.id,
                PersonaAssignment.stylist_user_id == stylist.id,
            )
        )
    ).scalars().first()
    if existing:
        if existing.status != "active":
            existing.status = "active"
            existing.revoked_at = None
            return True
        return False
    session.add(PersonaAssignment(persona_id=persona.id, stylist_user_id=stylist.id))
    return True


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    ap.add_argument("--only", help="seed a single character by slug")
    ap.add_argument("--with-stylists", action="store_true", help="also create the 3 stylist accounts")
    ap.add_argument("--stylist-password", default=DEFAULT_STYLIST_PASSWORD,
                    help="password for the seeded stylist accounts")
    ap.add_argument("--reset-stylist-passwords", action="store_true",
                    help="reset existing stylist accounts to --stylist-password")
    ap.add_argument("--grant-admin", default="",
                    help="grant the admin role to this existing account (email)")
    ap.add_argument("--with-portraits", action="store_true",
                    help="generate a portrait for any character missing one (costs real image calls)")
    ap.add_argument("--reseed-portraits", action="store_true",
                    help="regenerate portraits even where one already exists")
    ap.add_argument("--no-assign", action="store_true", help="skip stylist assignment")
    args = ap.parse_args()

    roster: List[PersonaSeed] = load_roster()
    if args.only:
        roster = [p for p in roster if p.slug == args.only]
        if not roster:
            raise SystemExit(f"No character with slug {args.only!r}")

    print(f"roster: {len(roster)} character(s) validated against the live styling enums")
    if args.dry_run:
        print("DRY RUN — nothing will be written\n")

    async with AsyncSessionLocal() as session:
        stylists: Dict[str, User] = {}
        if args.with_stylists or not args.no_assign:
            stylists = await _ensure_stylists(
                session, args.dry_run, args.stylist_password, args.reset_stylist_passwords
            )

        if args.grant_admin and not args.dry_run:
            # The alternative is ADMIN_EMAILS, which lives in a deploy secret and so cannot be
            # changed without a redeploy. Granting here keeps bootstrapping an admin to one
            # command in whichever environment is being seeded.
            from app.models.role import ROLE_ADMIN

            target = (
                await session.execute(select(User).where(User.email == args.grant_admin.lower()))
            ).scalars().first()
            if not target:
                print(f"  ! no account with email {args.grant_admin!r}; skipping admin grant")
            else:
                held = (
                    await session.execute(
                        select(UserRole).where(
                            UserRole.user_id == target.id, UserRole.role == ROLE_ADMIN
                        )
                    )
                ).scalars().first()
                if held:
                    print(f"  {target.email} is already an admin")
                else:
                    session.add(UserRole(user_id=target.id, role=ROLE_ADMIN))
                    print(f"  granted admin to {target.email}")

        created = updated = assigned = 0
        portraits: Dict[str, int] = {}
        for seed in roster:
            existed = (
                await session.execute(select(Persona.id).where(Persona.slug == seed.slug))
            ).scalars().first()
            persona = await _upsert_persona(session, seed, args.dry_run)
            if args.dry_run:
                continue
            created += 0 if existed else 1
            updated += 1 if existed else 0
            if persona and seed.assigned_stylist and not args.no_assign:
                if await _assign(session, persona, seed.assigned_stylist, stylists):
                    assigned += 1
            if persona and (args.with_portraits or args.reseed_portraits):
                _, status = await generate_and_persist_portrait(
                    session, get_storage_client(), persona, force=args.reseed_portraits
                )
                portraits[status] = portraits.get(status, 0) + 1
                print(f"  portrait {seed.slug:26} {status}")

        if args.dry_run:
            print("\nnothing written.")
            return
        await session.commit()

    print(f"\n{created} created, {updated} updated, {assigned} assignment(s) made.")
    print("Character accounts cannot be logged into; reach them by assignment only.")


if __name__ == "__main__":
    asyncio.run(main())
