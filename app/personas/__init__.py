"""Evaluation characters: loading and validating the authored roster.

`roster.yaml` is the source of truth; the database is a projection of it. This module is the only
thing that reads the file, so validation happens in exactly one place and nothing downstream has
to wonder whether a character is well-formed.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import yaml
from app.schemas.persona import PersonaSeed

ROSTER_PATH = Path(__file__).parent / "roster.yaml"


@lru_cache(maxsize=1)
def load_roster() -> List[PersonaSeed]:
    """Parses and validates every character. Raises on the first invalid one, with the slug in
    the message — a roster that half-loads is worse than one that refuses to."""
    raw = yaml.safe_load(ROSTER_PATH.read_text(encoding="utf-8")) or {}
    entries = raw.get("personas") or []
    roster = [PersonaSeed(**entry) for entry in entries]

    slugs = [p.slug for p in roster]
    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
    if duplicates:
        raise ValueError(f"roster.yaml has duplicate slugs: {duplicates}")
    return roster


def get_persona(slug: str) -> PersonaSeed:
    for persona in load_roster():
        if persona.slug == slug:
            return persona
    raise KeyError(f"No character with slug {slug!r}")


def roster_by_stylist() -> Dict[str, List[PersonaSeed]]:
    grouped: Dict[str, List[PersonaSeed]] = {}
    for persona in load_roster():
        grouped.setdefault(persona.assigned_stylist or "unassigned", []).append(persona)
    return grouped
