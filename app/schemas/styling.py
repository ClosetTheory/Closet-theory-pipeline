"""Schemas for the Styling Pipeline (outfit recommendation)."""

from enum import Enum
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.weather import WeatherSnapshot

_NULL_STRINGS = {"null", "none", "n/a", ""}


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class StylingIntent(BaseModel):
    """Structured styling intent produced by Stage 1 (Request Normalisation)."""

    occasion: Optional[str] = None
    formality: Optional[str] = None
    colors: List[str] = Field(default_factory=list)
    style_direction: Optional[str] = None
    gender: Optional[str] = None
    weather: Optional[str] = None
    time_context: Optional[str] = None
    anchor_garment_id: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)

    @field_validator("occasion", "formality", "style_direction", "gender", "weather", "time_context", mode="before")
    @classmethod
    def _coerce_literal_null_strings(cls, value: Any) -> Any:
        """LLM normalizers occasionally emit the literal string "null"/"none" instead of JSON
        null — treat those the same as an actual missing value rather than a real signal."""
        if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
            return None
        return value


class StylingContext(BaseModel):
    """Stage 2 output: normalised intent + factual application context."""

    intent: StylingIntent
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    behavioral_signals: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    allowed_categories: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    # Persisted here (inside StylingRequest.context, an existing JSON column) rather than as a
    # new column on StylingRequest -- this project has no migrations, and this is the cheapest
    # way for the Review tab to know which pipeline produced a given outfit without altering an
    # existing table. See StylingOrchestrator.run().
    used_hopit: bool = False


def describe_member_profile(context: "StylingContext") -> str:
    """The Stage 2 profile signals as prompt lines, for the LLM stages (semantic validation,
    aesthetic scoring). Empty string when Stage 2 found nothing stated, so prompts for ordinary
    members are byte-identical to before. Everything here is data about the member, never an
    instruction — the constraints line is the one place a hard rule is stated as a rule."""
    prefs = context.user_preferences or {}
    env = context.environment or {}
    lines: List[str] = []
    palette = prefs.get("palette") or {}
    if palette:
        season = palette.get("season_sub") or palette.get("season")
        bits = [b for b in (
            str(season) if season else None,
            f"{palette['temperature']}-toned" if palette.get("temperature") else None,
            f"{palette['undertone']} undertone" if palette.get("undertone") else None,
            f"{palette['depth']} depth" if palette.get("depth") else None,
            f"{palette['contrast']} contrast" if palette.get("contrast") else None,
        ) if b]
        summary = str(palette.get("summary") or "").strip()
        lines.append("Member's colour analysis: " + ", ".join(bits) + (f". {summary}" if summary else ""))
    love, avoid = prefs.get("colour_love") or [], prefs.get("colour_avoid") or []
    if love or avoid:
        lines.append(
            "Member's stated colours: "
            + (f"loves {', '.join(map(str, love))}" if love else "")
            + ("; " if love and avoid else "")
            + (f"avoids {', '.join(map(str, avoid))}" if avoid else "")
        )
    fits, leanings = prefs.get("fits_loved") or [], prefs.get("aesthetic_leanings") or []
    if fits or leanings:
        lines.append(
            "Member's stated style: "
            + (f"prefers {', '.join(str(f).replace('_', ' ') for f in fits)} fits" if fits else "")
            + ("; " if fits and leanings else "")
            + (f"leans {', '.join(str(a).replace('_', ' ') for a in leanings)}" if leanings else "")
        )
    if prefs.get("body_shape"):
        lines.append(f"Member's body shape: {str(prefs['body_shape']).replace('_', ' ')}")
    if context.hard_constraints:
        from app.rules.member_signals import constraint_label  # local: rules import schemas
        lines.append(
            "Member's HARD constraints — an outfit that breaks one of these is wrong for them regardless of how good it looks: "
            + "; ".join(constraint_label(c) for c in context.hard_constraints)
        )
    weather = env.get("weather") or {}
    if weather:
        w = f"{weather.get('temp_c')}°C"
        if weather.get("feels_like_c") is not None:
            w += f" (feels like {weather['feels_like_c']}°C)"
        w += f", {weather.get('condition')}"
        if weather.get("humidity_pct") is not None:
            w += f", humidity {weather['humidity_pct']}%"
        if weather.get("is_rainy"):
            w += ", rain likely"
        lines.append(f"Actual weather at the member's location: {w}")
    today = env.get("today") or {}
    if today and (today.get("tags") or today.get("note")):
        tags = ", ".join(str(t).replace("_", " ") for t in (today.get("tags") or []))
        lines.append("Member's plan for today: " + (tags or "unspecified") + (f" — {today['note']}" if today.get("note") else ""))
    return "\n".join(lines)


class GarmentSummary(BaseModel):
    """Lightweight, list/embed-friendly garment representation (real DB row)."""

    garment_id: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    garment_class: Optional[str] = None
    role: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    canonical_image_url: Optional[str] = None
    status: str
    quality_status: str
    created_at: Optional[str] = None


class CandidateGarment(BaseModel):
    garment_id: str
    category: Optional[str] = None
    role: Optional[str] = None
    behavior_score: float = 0.5
    retrieval_score: float = 0.5


class ScoreBreakdown(BaseModel):
    request_match: float = 0.0
    compatibility: float = 0.0
    user_preference: float = 0.0
    occasion_fit: float = 0.0
    visual_harmony: float = 0.0
    aesthetic_score: float = 0.0
    wardrobe_behavior: float = 0.0
    weather_fit: float = 0.0
    attribute_affinity: float = 0.0
    novelty: float = 0.0
    final_score: float = 0.0


class OutfitCandidate(BaseModel):
    outfit_id: Optional[str] = None
    garment_ids: List[str]
    roles: Dict[str, str] = Field(default_factory=dict)
    compatibility_reason: Optional[str] = None
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)


class ValidationResult(BaseModel):
    status: ValidationStatus
    confidence: float = 0.5
    issues: List[str] = Field(default_factory=list)
    reason: str = ""
    model: Optional[str] = None
    model_version: Optional[str] = None


class AestheticScoreResult(BaseModel):
    """Holistic stylist-style judgment of a candidate outfit as a single composition —
    distinct from compatibility (won't clash) and semantic validation (fits the request):
    this is specifically "would a stylist call this genuinely well put-together," judged
    once per outfit rather than averaged from pairwise checks."""

    score: float = 0.5
    reasoning: str = ""
    model: Optional[str] = None
    model_version: Optional[str] = None


def validate_styling_intent(raw_input: Any) -> StylingIntent:
    """Parses/validates LLM normalizer output into a StylingIntent, tolerant of extra keys."""
    data = raw_input
    if isinstance(raw_input, str):
        data = json.loads(raw_input)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for StylingIntent, got {type(data)}")
    return StylingIntent.model_validate(data)


def validate_validation_result(raw_input: Any, model: Optional[str] = None, model_version: Optional[str] = None) -> ValidationResult:
    """Parses/validates LLM validator output (semantic or visual) into a ValidationResult."""
    data = raw_input
    if isinstance(raw_input, str):
        data = json.loads(raw_input)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for ValidationResult, got {type(data)}")
    data.setdefault("model", model)
    data.setdefault("model_version", model_version)
    return ValidationResult.model_validate(data)


class VisualGateResult(BaseModel):
    """SPEC.md Section 34: the Visual Gate evaluates the actual generated outfit image.

    Output is a 0-10 quality score plus structured feedback across the spec's
    named evaluation areas — this is a quality score, not the styling decision.
    """

    score: float = Field(default=5.0, ge=0.0, le=10.0)
    feedback: Dict[str, str] = Field(default_factory=dict)
    model: Optional[str] = None
    model_version: Optional[str] = None


class SemanticGateResult(BaseModel):
    """SPEC.md Section 35: the Semantic Gate validates the generated result against the
    original request/context/selected garments — binary pass/fail + violations + feedback.
    """

    status: str = "PASS"  # "PASS" | "FAIL"
    violations: List[str] = Field(default_factory=list)
    feedback: str = ""
    model: Optional[str] = None
    model_version: Optional[str] = None


class GateAggregationResult(BaseModel):
    """Aggregates the Visual Gate + Semantic Gate (run in parallel on the generated image)
    into a single pass/feedback decision (SPEC.md Section 36)."""

    passed: bool
    visual: VisualGateResult
    semantic: SemanticGateResult


def validate_visual_gate_result(raw_input: Any, model: Optional[str] = None, model_version: Optional[str] = None) -> VisualGateResult:
    data = raw_input
    if isinstance(raw_input, str):
        data = json.loads(raw_input)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for VisualGateResult, got {type(data)}")
    data.setdefault("model", model)
    data.setdefault("model_version", model_version)
    return VisualGateResult.model_validate(data)


def validate_semantic_gate_result(raw_input: Any, model: Optional[str] = None, model_version: Optional[str] = None) -> SemanticGateResult:
    data = raw_input
    if isinstance(raw_input, str):
        data = json.loads(raw_input)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for SemanticGateResult, got {type(data)}")
    data.setdefault("model", model)
    data.setdefault("model_version", model_version)
    return SemanticGateResult.model_validate(data)


class StageTrace(BaseModel):
    """One entry in the Styling Pipeline's step-by-step execution trace (for the detail UI)."""

    stage: str
    title: str
    status: str = "SUCCEEDED"
    duration_ms: float = 0.0
    summary: Dict[str, Any] = Field(default_factory=dict)


class OutfitResult(BaseModel):
    """A single ranked outfit returned to the client — real garments + full provenance."""

    outfit_id: str
    rank: int
    garments: List[GarmentSummary]
    roles: Dict[str, str] = Field(default_factory=dict)
    scores: ScoreBreakdown
    compatibility_reason: Optional[str] = None
    semantic_validation: Optional[ValidationResult] = None
    generated_image_url: Optional[str] = None
    visual_gate: Optional[VisualGateResult] = None
    generation_semantic_gate: Optional[SemanticGateResult] = None


class StylingRecommendationRequest(BaseModel):
    request_text: Optional[str] = None
    anchor_garment_ids: Optional[List[str]] = None
    top_k: int = Field(default=3, ge=1, le=10)
    use_hopit: bool = Field(
        default=False,
        description="Route Stage 5 (retrieval) and Stage 6 (compatibility) through Hopit's "
        "hosted /v1/outfits:rank instead of our own retrieval+combinator+ranking. Every other "
        "stage runs unchanged on whatever candidates that stage produces.",
    )
    boldness_preference: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 = strongly favour conventional, naturally-matching combinations (default when "
            "omitted). 1.0 = favour bolder, less conventional combinations. Intended to be "
            "supplied from a learned per-user preference signal once behavioral history exists; "
            "until then, callers may pass it explicitly."
        ),
    )
    weather: Optional[WeatherSnapshot] = Field(
        default=None,
        description=(
            "Real weather for the member's location, when the caller has it (Outfit-of-the-Day "
            "always does). Stage 2 places it in StylingContext.environment and Stage 7 scores "
            "warmth against the actual feels-like temperature instead of the request's weather "
            "word. Omit it and weather_fit falls back to whatever the request text implied."
        ),
    )


class StylingRecommendationResponse(BaseModel):
    request_id: str
    intent: StylingIntent
    outfits: List[OutfitResult]
    trace: List[StageTrace] = Field(default_factory=list)
    # Populated only when `outfits` is empty — a plain-language summary of why every candidate
    # combination was rejected (drawn from Stage 8/10's real rejection reasons), so the frontend
    # can show the caller something actionable instead of a bare empty result.
    no_outfit_reason: Optional[str] = None


class StylingRunProgressResponse(BaseModel):
    """A checkpointed snapshot of an in-flight (or just-finished) streaming styling run —
    what GET /wardrobe/styling/runs/active and /runs/{id} return, so a client that switched
    tabs mid-run can rebuild the stage grid from `trace` and pick up polling from
    `current_stage`/`status` instead of starting over."""

    id: str
    status: str
    current_stage: Optional[str] = None
    trace: List[StageTrace] = Field(default_factory=list)
    styling_request_id: Optional[str] = None
    error_message: Optional[str] = None
    request_payload: Dict[str, Any] = Field(default_factory=dict)


class OutfitVoteRequest(BaseModel):
    vote: Literal["up", "down"]
    # Optional reason: chips from app.rules.feedback.REASON_TAGS and/or free text. Turned into
    # per-garment attribution in the behaviour ledger; re-posting the same vote with a reason
    # updates the earlier row rather than adding a second vote.
    comment: Optional[str] = Field(default=None, max_length=2000)
    tags: List[str] = Field(default_factory=list)


class OutfitVoteResponse(BaseModel):
    outfit_id: str
    vote: Literal["up", "down"]
    outfit_boldness: float
    boldness_preference: float
    vote_count: int
    # The behaviour ledger's view after this vote (see app.rules.wardrobe_behavior): which garments
    # the vote touched and what each now scores, so the UI can react immediately — e.g. dim other
    # shortlisted outfits that share a just-disliked garment.
    garment_ids: List[str] = Field(default_factory=list)
    garment_behavior_scores: Dict[str, float] = Field(default_factory=dict)
    ledger_votes: int = 0
    # How the reason (chips + comment) was understood, e.g. "blamed white sneakers · no floral".
    feedback_summary: Optional[str] = None


class AttributeAffinityValue(BaseModel):
    value: str
    score: float  # raw EMA score, [-1, 1]
    count: int
    confidence: float  # [0, 1] — how much the score should be trusted given vote volume


class StyleProfileResponse(BaseModel):
    boldness_preference: float
    vote_count: int
    attribute_affinities: Dict[str, List[AttributeAffinityValue]]


# --- Outfit of the Day ---
# A daily, weather-aware pick — wraps the existing /recommendations pipeline with a
# synthesized request_text combining real weather and a context paragraph auto-derived from
# the member's actual styling history and wardrobe (app/styling/member_context.py), rather
# than a hand-picked "persona". Cached per (member, calendar date, location) in
# app/models/ootd.py::OutfitOfTheDay.

class OutfitOfTheDayRequest(BaseModel):
    location: str = Field(..., description="Free-text place name for a real weather lookup, e.g. 'Mumbai' — can be anywhere")
    extra_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional freeform addition on top of the auto-derived context, e.g. 'there's a "
            "client meeting today'. Not required — the styling context is derived automatically "
            "from this member's real styling history and wardrobe even if this is omitted."
        ),
    )
    force_regenerate: bool = Field(
        default=False,
        description="Re-run generation even if today's pick for this member/location already exists.",
    )
    use_hopit: bool = Field(
        default=False,
        description="Same meaning as StylingRecommendationRequest.use_hopit — Stage 5/6 routed "
        "through Hopit. A Hopit pick is never cached in outfits_of_the_day (that table has no "
        "column to distinguish which pipeline produced a day's pick, and this project has no "
        "migrations to add one), so it's regenerated live on every call.",
    )


class OutfitOfTheDayResponse(BaseModel):
    date: str = Field(..., description="Calendar date (YYYY-MM-DD) this pick is for")
    location: str
    context_used: str = Field(..., description="The auto-derived member context (plus any extra_hint) actually used to generate this pick")
    weather: WeatherSnapshot
    styling: StylingRecommendationResponse
    cached: bool = Field(..., description="True if this was an already-generated pick for today, not a fresh run")
    generation_source: str = Field(default="on_demand", description="'on_demand' or 'scheduled'")


class OOTDSubscriptionRequest(BaseModel):
    location: str
    extra_hint: Optional[str] = Field(default=None, description="Optional recurring addition to the auto-derived context")
    enabled: bool = True


class OOTDSubscriptionResponse(BaseModel):
    location: Optional[str] = None
    extra_hint: Optional[str] = None
    enabled: bool = False


# --- Outfit garment swap ---
# Replace one role (TOP/BOTTOM/OUTERWEAR/FOOTWEAR/ONE_PIECE/ACCESSORY) in an already-generated
# outfit — either by naming the exact replacement, or by describing what's wanted in free text
# (app/styling/swap.py interprets it and picks a real match). Never invents a garment; always a
# real, member-owned, COMPLETED wardrobe item, re-checked for compatibility with what remains.

class SwapGarmentRequest(BaseModel):
    role: str = Field(..., description="Which role to replace, e.g. 'FOOTWEAR' — see roles on the outfit's `roles` field")
    new_garment_id: str = Field(..., description="A real garment_id from this member's own wardrobe")


class ChatSwapRequest(BaseModel):
    instruction: str = Field(..., description="Free text, e.g. \"swap the shoes for something more casual\"")


class SwapCandidateSummary(BaseModel):
    garment_id: str
    subcategory: Optional[str] = None
    canonical_image_url: Optional[str] = None


# --- Internal stylist QA review panel (see app/static/review.html) ---
# Separate from OutfitVoteRequest/Response above: that mechanism updates the member's learned
# StyleProfile and keeps no durable per-outfit record. This is purely an internal record of a
# reviewer's like/dislike + optional comment on one of their own account's generated outfits,
# with no effect on ranking or learning.

class StylistReviewRequest(BaseModel):
    vote: Literal["like", "dislike"]
    comment: Optional[str] = Field(default=None, max_length=2000)


class StylistReviewResult(BaseModel):
    id: str
    outfit_id: str
    vote: str
    comment: Optional[str] = None
    created_at: str
    updated_at: str


class OutfitReviewQueueItem(BaseModel):
    outfit: OutfitResult
    request_text: Optional[str] = None
    generated_at: str
    review: Optional[StylistReviewResult] = None


# --- Scoring one's own outfits with the character panel's five-dimension rubric ---
# Same request shape as PersonaOutfitReviewRequest (and the same dimension validation), so the
# /review page can drive both with one form; only the storage table differs (see
# app.models.persona_review.OwnOutfitReview for why).

class OwnOutfitReviewResult(BaseModel):
    id: str
    outfit_id: str
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


class OwnOutfitReviewItem(BaseModel):
    """One of this account's generated outfits with every five-dimension score attached — the
    same item shape /personas/{id}/outfits returns, so the review page renders both alike."""
    outfit: OutfitResult
    generated_at: str
    request_id: Optional[str] = None
    request_text: Optional[str] = None
    used_hopit: bool = False
    my_review: Optional[OwnOutfitReviewResult] = None
    reviews: List[OwnOutfitReviewResult] = Field(default_factory=list)
