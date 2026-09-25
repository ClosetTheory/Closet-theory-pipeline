"""Loads the *stated* profile Stage 2 builds its context from: colour analysis, onboarding
preferences, weekly plan, hard constraints and climate for the member being styled.

Shape follows production's tables one-to-one — `consumer_interaction_profiles.color_analysis`,
`consumer_profiles.preferences`, `consumer_interaction_profiles.weekly_plan` — so the production
port of this pipeline can implement the same loader over those tables and the rest of Stage 2
does not change. In this deployment the only members with a stated profile are the evaluation
characters, whose Persona row already carries every field in exactly that shape (see
app/models/persona.py), keyed by the character's backing user id, which is also its tenant_id.

Ordinary accounts get an empty MemberSignals and Stage 2 behaves as before.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.persona import Persona


class MemberSignals(BaseModel):
    source: str = "none"  # "persona" | "none"
    display_name: Optional[str] = None
    color_analysis: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    weekly_plan: Dict[str, Any] = Field(default_factory=dict)
    hard_constraints: List[str] = Field(default_factory=list)
    climate: Dict[str, Any] = Field(default_factory=dict)
    body_shape: Optional[str] = None
    fit_pain_points: List[str] = Field(default_factory=list)
    styling_notes: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.source == "none"


async def load_member_signals(session: AsyncSession, tenant_id: str, member_id: str) -> MemberSignals:
    """A character is backed by a user whose id is both its tenant_id and member_id, so a
    Persona row matching the tenant is the member being styled. Anything else is an ordinary
    account with no stated profile."""
    persona = (
        await session.execute(select(Persona).where(Persona.user_id == tenant_id))
    ).scalars().first()
    if persona is None or persona.user_id != member_id:
        return MemberSignals()
    return MemberSignals(
        source="persona",
        display_name=persona.display_name,
        color_analysis=dict(persona.color_analysis or {}),
        preferences=dict(persona.preferences or {}),
        weekly_plan=dict(persona.weekly_plan or {}),
        hard_constraints=list(persona.hard_constraints or []),
        climate=dict(persona.climate or {}),
        body_shape=persona.body_shape,
        fit_pain_points=list(persona.fit_pain_points or []),
        styling_notes=persona.styling_notes,
    )
