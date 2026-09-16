"""Evaluation characters ("personas"): synthetic members used to measure styling quality.

Production cannot answer "how good is the styling?" — 163 of its 164 members are in one city,
`body_shape` is set on 18 accounts, the body-scan table is empty, and no member has ever accepted,
rejected or bookmarked an outfit. So the engine is being judged on nothing.

A character fixes that by being a deliberately specified member: a known body shape, colour
season, climate, wardrobe need and set of hard constraints. A stylist is assigned several, builds
each wardrobe by hand, generates outfits and scores them.

The load-bearing design decision is that **a character is backed by a real `User` row** with its
own `tenant_id`/`member_id`. Every ingestion and styling write in this codebase already takes its
scope from the authenticated caller, and every read already filters on `tenant_id` — so making a
character a user means the entire upload -> 9-stage pipeline -> styling -> review stack works on
it unchanged, and tenant isolation becomes character isolation for free. The alternative (a
`persona_id` threaded through every query) would also have collided with the
`UniqueConstraint("tenant_id", "member_id")` on `style_profiles` and `ootd_subscriptions`,
collapsing every character of one stylist onto a single learned profile.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now


class Persona(Base):
    """One row per evaluation character. `slug` is the stable seed key, so re-running the seeder
    updates a character in place instead of creating a second one."""

    __tablename__ = "personas"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("persona"))
    # The backing account. Unique because a user is either one character or not a character.
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    # --- identity ---
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Mirrors User.gender / GarmentEnum: it is written straight onto the backing account and is
    # what defaults StylingIntent.gender, so it must use that vocabulary, not a descriptive one.
    gender: Mapped[str] = mapped_column(String(16), nullable=False)  # "women" | "men" | "unisex"
    # How the person describes themselves, which is a different question from the styling filter
    # above — an androgynous-presenting person may still want womenswear retrieved.
    gender_presentation: Mapped[str] = mapped_column(String(24), nullable=False, default="unspecified")
    pronouns: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    occupation: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    lifestyle: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    work_dress_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # --- geography. Production is 163/164 "mumbai", so this is entirely new coverage, and it is
    # the axis that changes styling most: a Gangtok January and a Chennai May are different
    # products. `ootd_location` is passed verbatim to the real weather lookup. ---
    city: Mapped[str] = mapped_column(String(64), nullable=False)
    state_region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    country: Mapped[str] = mapped_column(String(64), nullable=False, default="India")
    climate_zone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    ootd_location: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    climate: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    # --- body ---
    body_shape: Mapped[str] = mapped_column(String(32), nullable=False)
    height_cm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height_band: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # petite|average|tall
    build: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # Deliberately sparse. Production's body-scan table has zero rows, so characters must not
    # assume a measurement-rich world — carry only what a stylist would ask on a call.
    measurements: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # The most useful stylist-facing field: "every kurta is 4 inches too long", "UK11 feet and
    # Indian retail stops at 10".
    fit_pain_points: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    # --- colour. Shape mirrors production's consumer_interaction_profiles.color_analysis exactly
    # ({season, season_sub, undertone, depth, temperature, contrast, palette, palette_bins,
    # summary}) so a character row and a real member row are directly diffable. NOTE the colour
    # season lives inside this blob and is never hoisted to a bare `season` column: garments
    # already have a `season` meaning weather, and the two sharing a name is a live bug waiting
    # to happen. ---
    color_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    skin_tone_monk: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-10, prod's scale

    # --- hair. Not vanity: hair-to-skin contrast is half of what decides whether someone can
    # carry a high-contrast outfit, and head coverings are a hard styling constraint. ---
    hair: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    # --- wardrobe need ---
    # share_of_wardrobe and share_of_occasions are deliberately separate: a Mumbai marketer owns
    # three sarees but attends one saree occasion a year, while a Chennai principal owns 0.75 and
    # wears 0.70. An engine reading only ownership gets both of them wrong.
    ethnic_wear: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # Production's consumer_profiles.preferences shape, verbatim, including the camelCase inside
    # styling_context.
    preferences: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # Flows into the existing StylingContext.hard_constraints: no_sleeveless, modest_coverage,
    # turban_colour_coordination, no_leather, quick_dry_only...
    hard_constraints: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    budget_tier: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    weekly_plan: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    # --- narrative + provenance ---
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    styling_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Why this character exists — the axis it is the sole carrier of. Keeps the roster from
    # drifting back into twenty variations of one person.
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    portrait_image_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("image_assets.id", ondelete="SET NULL"), nullable=True
    )
    portrait_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # draft|active|archived
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="seed")  # seed|manual

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class PersonaAssignment(Base):
    """One row per (character, stylist). The schema allows many-to-many; the assign endpoint
    defaults to exclusive and revokes other active rows, because two stylists independently
    building the same wardrobe is almost always a mistake rather than an intent.

    Revocation sets `status` rather than deleting, so "who was working on this in August" stays
    answerable. A partial unique index on status='active' would express exclusivity in DDL but
    isn't portable to the SQLite test database, so it is enforced in application code.
    """

    __tablename__ = "persona_assignments"
    __table_args__ = (UniqueConstraint("persona_id", "stylist_user_id", name="uq_persona_assignment"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("passign"))
    persona_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("personas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stylist_user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assigned_by_user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # active|revoked
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class PersonaGarmentUpload(Base):
    """Audit trail: which stylist put this garment into which character's wardrobe.

    A column on `garments` would be the natural home, but adding one is exactly the unsafe
    operation this schema cannot perform (see app/models/role.py). A side table costs one insert
    per ingestion and carries the same information.
    """

    __tablename__ = "persona_garment_uploads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("pgup"))
    garment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("garments.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    persona_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("personas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    uploaded_by_user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
