"""Validation for evaluation characters.

The point of this module is that it imports the engine's *live* enums. A character that declares
an occasion or season the styling pipeline cannot actually filter on fails to load, loudly, at
seed time — rather than being accepted and then quietly producing empty recommendations three
weeks later. That is what makes "described in fields our system can consume" enforced rather than
aspirational.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from app.schemas.attributes import GenderEnum, OccasionEnum, SeasonEnum

# --- vocabularies -----------------------------------------------------------------------------
# Body shapes are lowercased single vocabulary. Production holds `rectangle`, `Rectangle` and
# `Triangle` in one column across 18 rows, which is three spellings of two values.
WOMEN_BODY_SHAPES = ("rectangle", "triangle", "inverted_triangle", "hourglass", "oval")
MEN_BODY_SHAPES = ("rectangle", "trapezoid", "triangle", "inverted_triangle", "oval")
ALL_BODY_SHAPES = tuple(sorted(set(WOMEN_BODY_SHAPES) | set(MEN_BODY_SHAPES)))

# Four-season names only. Production's `season` field currently mixes seasonal names with bare
# temperature words (`cool`/`warm`/`neutral`) even though a separate `temperature` field already
# holds exactly those — 19 of 57 rows have the answer in the wrong field. `season_sub` carries
# the 12-season refinement, because four seasons genuinely under-resolve Indian colouring where
# most people land in "warm + deep".
COLOUR_SEASONS = ("Spring", "Summer", "Autumn", "Winter")
# `olive` is not in production's vocabulary and should be. Its own gold-standard summary says
# "olive-brown skin" while undertone says `warm` — so the most common South Asian undertone
# collapses into `warm` and loses the distinction that matters, since olive skin goes sallow
# against the yellow-golds that flatter true-warm skin.
UNDERTONES = ("warm", "cool", "neutral", "olive")
DEPTHS = ("light", "medium", "deep")
TEMPERATURES = ("warm", "cool", "neutral")
CONTRASTS = ("low", "medium", "high")

HEIGHT_BANDS = ("petite", "average", "tall")
BUILDS = ("slight", "lean", "athletic", "average", "solid", "full", "plus")
HAIR_LENGTHS = ("shaved", "cropped", "short", "chin", "shoulder", "mid_back", "waist", "hip")
HAIR_TEXTURES = ("straight", "wavy", "curly", "coily")
HEAD_COVERINGS = ("none", "turban", "hijab", "dupatta_over_head", "cap", "stole")
BUDGET_TIERS = ("value", "mid", "premium", "luxury")
DRAPE_COMPETENCE = ("none", "learning", "comfortable", "expert")

OCCASION_VALUES = {e.value for e in OccasionEnum}
SEASON_VALUES = {e.value for e in SeasonEnum}
GENDER_VALUES = {e.value for e in GenderEnum}


class ColorAnalysis(BaseModel):
    """Mirrors production's consumer_interaction_profiles.color_analysis, key for key, so a
    character row and a real member row are directly comparable."""

    season: Literal[COLOUR_SEASONS]  # type: ignore[valid-type]
    season_sub: Optional[str] = None  # "Deep Autumn", "Soft Summer", ...
    undertone: Literal[UNDERTONES]  # type: ignore[valid-type]
    depth: Literal[DEPTHS]  # type: ignore[valid-type]
    temperature: Literal[TEMPERATURES]  # type: ignore[valid-type]
    contrast: Literal[CONTRASTS] = "medium"  # type: ignore[valid-type]
    palette: List[str] = Field(min_length=3, max_length=9)
    summary: str
    source: str = "panel_authored"
    vocabulary: str = "four_season_v1"

    @field_validator("palette")
    @classmethod
    def _hex_only(cls, v: List[str]) -> List[str]:
        bad = [c for c in v if not (c.startswith("#") and len(c) == 7)]
        if bad:
            raise ValueError(f"palette entries must be #rrggbb hex: {bad}")
        return [c.lower() for c in v]


class Hair(BaseModel):
    length: Literal[HAIR_LENGTHS]  # type: ignore[valid-type]
    texture: Literal[HAIR_TEXTURES]  # type: ignore[valid-type]
    colour: str
    facial_hair: Optional[str] = None
    # Not cosmetic: a turban is the largest block of colour in an outfit and a hijab changes the
    # neckline calculus, so this feeds hard_constraints, not just the portrait.
    head_covering: Literal[HEAD_COVERINGS] = "none"  # type: ignore[valid-type]


class EthnicWear(BaseModel):
    # Ownership and need are deliberately separate numbers. A Mumbai marketer owns three sarees
    # (0.15) but attends one saree occasion a year (0.04); a Chennai principal owns 0.75 and
    # wears 0.70. An engine reading only ownership over-proposes to the first and under-proposes
    # to the second.
    share_of_wardrobe: float = Field(ge=0.0, le=1.0)
    share_of_occasions: float = Field(ge=0.0, le=1.0)
    daily_ethnic: bool = False
    drape_competence: Literal[DRAPE_COMPETENCE] = "comfortable"  # type: ignore[valid-type]
    garment_classes: List[str] = Field(default_factory=list)
    regional_garments: List[str] = Field(default_factory=list)
    festival_calendar: List[str] = Field(default_factory=list)


class Climate(BaseModel):
    zone: str
    summer_severity: Literal["mild", "hot", "extreme"] = "hot"
    winter_severity: Literal["none", "mild", "moderate", "cold"] = "mild"
    humidity: Literal["dry", "moderate", "humid", "very_humid"] = "moderate"
    monsoon_intensity: Literal["low", "moderate", "heavy", "extreme"] = "moderate"
    monsoon_months: List[int] = Field(default_factory=list)


class PersonaSeed(BaseModel):
    """One character as authored in roster.yaml."""

    slug: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    name: str
    age: int = Field(ge=16, le=99)
    gender: str
    gender_presentation: str = "unspecified"
    pronouns: Optional[str] = None
    occupation: Optional[str] = None
    lifestyle: Optional[str] = None
    work_dress_code: Optional[str] = None

    city: str
    state_region: Optional[str] = None
    country: str = "India"
    ootd_location: str  # passed verbatim to the real weather lookup, so every character is
    #                     immediately runnable through /ootd
    climate: Climate

    body_shape: str
    height_cm: Optional[int] = Field(default=None, ge=120, le=220)
    height_band: Literal[HEIGHT_BANDS] = "average"  # type: ignore[valid-type]
    build: Literal[BUILDS] = "average"  # type: ignore[valid-type]
    measurements: Dict[str, Any] = Field(default_factory=dict)
    fit_pain_points: List[str] = Field(default_factory=list)

    skin_tone_monk: int = Field(ge=1, le=10)
    color_analysis: ColorAnalysis
    hair: Hair

    ethnic_wear: EthnicWear
    preferences: Dict[str, Any] = Field(default_factory=dict)
    hard_constraints: List[str] = Field(default_factory=list)
    budget_tier: Literal[BUDGET_TIERS] = "mid"  # type: ignore[valid-type]
    weekly_plan: Dict[str, Any] = Field(default_factory=dict)

    bio: str
    styling_notes: Optional[str] = None
    # Why this character exists — the axis it is the sole carrier of. Required, because without
    # it a roster drifts back into twenty variations of the same person.
    rationale: str
    assigned_stylist: Optional[str] = None

    @field_validator("gender")
    @classmethod
    def _known_gender(cls, v: str) -> str:
        if v not in GENDER_VALUES:
            raise ValueError(f"gender must be one of {sorted(GENDER_VALUES)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _shape_matches_gender(self) -> "PersonaSeed":
        allowed = (
            WOMEN_BODY_SHAPES if self.gender == "women"
            else MEN_BODY_SHAPES if self.gender == "men"
            else ALL_BODY_SHAPES
        )
        if self.body_shape not in allowed:
            raise ValueError(
                f"{self.slug}: body_shape {self.body_shape!r} is not valid for gender "
                f"{self.gender!r}; expected one of {sorted(allowed)}"
            )
        return self

    @model_validator(mode="after")
    def _occasion_mix_is_real(self) -> "PersonaSeed":
        """Every key must be an occasion the engine can actually filter on, and the mix must sum
        to 1. This is the check that makes the roster executable rather than decorative."""
        mix = (self.preferences or {}).get("occasion_mix") or {}
        unknown = sorted(set(mix) - OCCASION_VALUES)
        if unknown:
            raise ValueError(
                f"{self.slug}: occasion_mix has keys the styling engine cannot filter on: "
                f"{unknown}. Valid: {sorted(OCCASION_VALUES)}"
            )
        if mix:
            total = sum(float(v) for v in mix.values())
            if abs(total - 1.0) > 0.02:
                raise ValueError(f"{self.slug}: occasion_mix sums to {total:.2f}, expected 1.00")
        return self


class PersonaRead(BaseModel):
    """What the admin and stylist panels render."""

    persona_id: str
    slug: str
    display_name: str
    age: Optional[int] = None
    gender: str
    city: str
    state_region: Optional[str] = None
    climate_zone: Optional[str] = None
    occupation: Optional[str] = None
    body_shape: str
    height_cm: Optional[int] = None
    height_band: Optional[str] = None
    build: Optional[str] = None
    skin_tone_monk: Optional[int] = None
    color_analysis: Dict[str, Any] = Field(default_factory=dict)
    hair: Dict[str, Any] = Field(default_factory=dict)
    ethnic_wear: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    hard_constraints: List[str] = Field(default_factory=list)
    fit_pain_points: List[str] = Field(default_factory=list)
    budget_tier: Optional[str] = None
    bio: Optional[str] = None
    styling_notes: Optional[str] = None
    rationale: Optional[str] = None
    portrait_url: Optional[str] = None
    status: str
    assigned_stylists: List[str] = Field(default_factory=list)
    garment_count: int = 0
    outfit_count: int = 0
    review_count: int = 0
    average_rating: Optional[float] = None


class AssignmentRequest(BaseModel):
    stylist_user_id: str
    # Two stylists independently building the same wardrobe is nearly always an accident, so
    # assigning defaults to revoking other active holders.
    exclusive: bool = True


class PersonaOutfitReviewRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    vote: Optional[Literal["like", "dislike"]] = None
    dimension_ratings: Dict[str, int] = Field(default_factory=dict)
    comment: Optional[str] = Field(default=None, max_length=4000)
    would_wear: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)

    @field_validator("dimension_ratings")
    @classmethod
    def _known_dimensions(cls, v: Dict[str, int]) -> Dict[str, int]:
        from app.models.persona_review import REVIEW_DIMENSIONS

        unknown = sorted(set(v) - set(REVIEW_DIMENSIONS))
        if unknown:
            raise ValueError(f"unknown review dimensions {unknown}; valid: {list(REVIEW_DIMENSIONS)}")
        out_of_range = {k: n for k, n in v.items() if not 1 <= int(n) <= 5}
        if out_of_range:
            raise ValueError(f"dimension ratings must be 1-5: {out_of_range}")
        return {k: int(n) for k, n in v.items()}


class PersonaOutfitReviewResult(BaseModel):
    id: str
    outfit_id: str
    persona_id: str
    reviewer_user_id: str
    reviewer_name: Optional[str] = None
    rating: int
    vote: Optional[str] = None
    dimension_ratings: Dict[str, int] = Field(default_factory=dict)
    comment: Optional[str] = None
    would_wear: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class UserListItem(BaseModel):
    user_id: str
    email: str
    display_name: Optional[str] = None
    gender: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    # Derived from the presence of a personas row rather than a flag column, so it cannot drift.
    is_persona: bool = False
    persona_slug: Optional[str] = None
