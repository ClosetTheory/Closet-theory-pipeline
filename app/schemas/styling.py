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


class StylingRecommendationResponse(BaseModel):
    request_id: str
    intent: StylingIntent
    outfits: List[OutfitResult]
    trace: List[StageTrace] = Field(default_factory=list)
    # Populated only when `outfits` is empty — a plain-language summary of why every candidate
    # combination was rejected (drawn from Stage 8/10's real rejection reasons), so the frontend
    # can show the caller something actionable instead of a bare empty result.
    no_outfit_reason: Optional[str] = None


class OutfitVoteRequest(BaseModel):
    vote: Literal["up", "down"]


class OutfitVoteResponse(BaseModel):
    outfit_id: str
    vote: Literal["up", "down"]
    outfit_boldness: float
    boldness_preference: float
    vote_count: int


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
