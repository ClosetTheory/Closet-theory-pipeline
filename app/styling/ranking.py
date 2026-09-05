"""Styling Stage 7 (part 2): Outfit Ranking via the weighted scorer.

Builds an OutfitCandidate (with full score breakdown) for every surviving
combination, using rule-based signals wherever possible and only the neutral
stubs (wardrobe_behavior, novelty) where no real data exists yet. user_preference
rewards conventional/well-matched combinations by default, shiftable toward
bolder combinations via StylingContext.user_preferences["boldness_preference"].
attribute_affinity rewards colour/pattern values the member has upvoted before,
via StylingContext.user_preferences["attribute_affinities"] (see app/rules/style_profile.py).
"""

from typing import Dict, List, Optional, Tuple
from app.models.garment import Garment
from app.rules.scoring import DEFAULT_STYLING_WEIGHTS, apply_diversity_penalty, compute_outfit_score
from app.rules.style_profile import attribute_affinity_score
from app.rules.taxonomy import bundle_category
from app.rules.visual import FORMALITY_SCORES, NEUTRAL_COLORS, _get_max_formality
from app.rules.wardrobe_behavior import score_wardrobe_behavior
from app.schemas.styling import OutfitCandidate, ScoreBreakdown, StylingContext, StylingIntent
from app.styling.compatibility import evaluate_outfit_compatibility
from app.styling.retrieval import resolve_role


def _request_match(garments: List[Garment], intent: StylingIntent) -> float:
    if not intent.formality and not intent.colors and not intent.occasion:
        return 0.7  # no explicit request signal (e.g. anchor-only styling) — neutral

    score = 0.5
    if intent.formality:
        target = intent.formality.lower()
        if any(target in [o.lower() for o in (g.attributes_json or {}).get("occasion", [])] for g in garments):
            score += 0.25

    if intent.colors:
        wanted = {c.lower() for c in intent.colors}
        garment_colors = {c for g in garments for c in (g.attributes_json or {}).get("colour", [])}
        if wanted & garment_colors:
            score += 0.25
        elif "dark" in wanted and garment_colors & {"black", "navy", "charcoal", "brown"}:
            score += 0.25
        elif "neutral" in wanted and garment_colors & NEUTRAL_COLORS:
            score += 0.25

    return max(0.0, min(1.0, score))


DRESSY_CONTEXTS = {"party", "formal", "evening"}


def _occasion_fit(garments: List[Garment], intent: StylingIntent) -> float:
    signals = {(intent.occasion or "").lower(), (intent.formality or "").lower()}
    is_dressy_request = bool(signals & DRESSY_CONTEXTS)

    if not intent.formality and not is_dressy_request:
        return 0.7

    # A dressy request (party/formal/evening) reads as an actual "look" when it's a single
    # cohesive ONE_PIECE garment, not a separately-matched top+bottom — reward that directly,
    # independent of the formality-score averaging below, which alone can't distinguish a
    # well-matched top+bottom pairing from a genuine dress/jumpsuit silhouette.
    silhouette_bonus = 0.0
    if is_dressy_request:
        roles = {resolve_role(g) for g in garments}
        silhouette_bonus = 0.15 if "ONE_PIECE" in roles else 0.0

    if not intent.formality:
        return max(0.0, min(1.0, 0.7 + silhouette_bonus))

    target_score = FORMALITY_SCORES.get(intent.formality.lower(), 3)
    garment_scores = [
        _get_max_formality((g.attributes_json or {}).get("occasion", [])) for g in garments
    ]
    if not garment_scores:
        return 0.5
    avg_diff = sum(abs(target_score - s) for s in garment_scores) / len(garment_scores)
    base = max(0.0, 1.0 - (avg_diff / 5.0))
    return max(0.0, min(1.0, base + silhouette_bonus))


# Target warmth (0=lightest, 1=warmest) implied by each recognised weather term. Garments
# carry a "warmth" attribute in the same 0-1 range (set at ingestion time), so this lets an
# outfit's actual warmth be compared against what the request's weather context calls for.
WEATHER_WARMTH_TARGET = {
    "hot": 0.1,
    "warm": 0.2,
    "sunny": 0.25,
    "mild": 0.4,
    "cool": 0.55,
    "rainy": 0.6,
    "windy": 0.6,
    "cold": 0.8,
    "snowy": 0.9,
    "freezing": 0.95,
}


def _weather_fit(garments: List[Garment], intent: StylingIntent) -> float:
    """How well the outfit's warmth matches the request's weather (if any was given).
    No weather signal in the request — neutral score, same as the other "no signal" stubs."""
    if not intent.weather:
        return 0.7

    target = WEATHER_WARMTH_TARGET.get(intent.weather.lower())
    if target is None:
        return 0.7

    warmth_values = [float((g.attributes_json or {}).get("warmth", 0.5)) for g in garments]
    if not warmth_values:
        return 0.5
    avg_warmth = sum(warmth_values) / len(warmth_values)
    return max(0.0, 1.0 - abs(target - avg_warmth))


async def _visual_harmony(garments: List[Garment]) -> float:
    from itertools import combinations
    from app.rules.visual import evaluate_visual_rules
    from app.styling.compatibility import _attrs

    if len(garments) < 2:
        return 1.0
    scores = []
    for a, b in combinations(garments, 2):
        _confident, _decision, score, _reason, _ver = evaluate_visual_rules(_attrs(a), _attrs(b))
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.7


def _user_preference(visual_harmony_score: float, context: StylingContext) -> float:
    """Rewards outfits whose "conventionality" (naturally-matching vs. bold/clashing, using the
    already-computed visual_harmony score as that signal) matches the user's boldness
    preference. Defaults to strongly favouring conventional, well-matched combinations
    (boldness_preference=0.0) when no preference has been supplied — this is the tunable knob
    for someone who prefers bolder, less conventional outfits: raising boldness_preference
    (0.0-1.0) shifts reward toward lower-harmony/more-distinctive combinations instead of
    penalizing them. Intended to eventually be set from a learned per-user signal (derived from
    which recommendations a user actually picks/saves over time) once behavioral history is
    tracked; until then it's an explicit input (see StylingRecommendationRequest.boldness_preference)."""
    boldness = context.user_preferences.get("boldness_preference", 0.0)
    boldness = max(0.0, min(1.0, float(boldness)))
    target_harmony = 1.0 - boldness
    return max(0.0, 1.0 - abs(target_harmony - visual_harmony_score))


class RankingTrace:
    """Bundles ranking-stage telemetry alongside the diversified outfits, for the pipeline detail view."""

    def __init__(self, outfits: List[OutfitCandidate], total_evaluated: int, total_compatible: int, pairing_rejected: int = 0):
        self.outfits = outfits
        self.total_evaluated = total_evaluated
        self.total_compatible = total_compatible
        self.pairing_rejected = pairing_rejected


LAYER_IMPROVEMENT_EPSILON = 0.01  # minimum final_score gain required to suggest an optional layer


async def rank_combinations(
    combinations_with_retrieval: List[tuple],
    intent: StylingIntent,
    context: StylingContext,
    top_k: int,
) -> RankingTrace:
    """Evaluates compatibility + scores every combination, drops INCOMPATIBLE ones, returns
    diversified top_k. Combos tagged with a base_combo_key (an optional outerwear layer added
    on top of a base combo — see combinator.py) are only kept if their final_score actually
    improves on their paired base combo's score; otherwise the layer isn't worth suggesting
    and only the base combo survives."""
    scored: List[OutfitCandidate] = []
    scored_by_key: Dict[frozenset, float] = {}
    pending_layered: List[Tuple[OutfitCandidate, Optional[frozenset]]] = []
    pairing_rejected = 0

    for entry in combinations_with_retrieval:
        garments, _retrieval_sum, base_combo_key = entry if len(entry) == 3 else (*entry, None)

        decision, compatibility_score, reason, hard_reject_source = await evaluate_outfit_compatibility(garments)
        if decision == "INCOMPATIBLE":
            if hard_reject_source == "pairing":
                pairing_rejected += 1
            continue

        visual_harmony_score = await _visual_harmony(garments)
        components = {
            "request_match": _request_match(garments, intent),
            "compatibility": compatibility_score,
            "user_preference": _user_preference(visual_harmony_score, context),
            "occasion_fit": _occasion_fit(garments, intent),
            "visual_harmony": visual_harmony_score,
            "wardrobe_behavior": sum(score_wardrobe_behavior(g) for g in garments) / len(garments),
            "weather_fit": _weather_fit(garments, intent),
            "attribute_affinity": attribute_affinity_score(
                [g.attributes_json or {} for g in garments], context.user_preferences.get("attribute_affinities", {})
            ),
            "novelty": 1.0,  # stub: no WearLog history to compare against yet
        }
        final_score = compute_outfit_score(components, DEFAULT_STYLING_WEIGHTS)

        roles = {g.id: (resolve_role(g) or "UNKNOWN") for g in garments}
        candidate = OutfitCandidate(
            garment_ids=[g.id for g in garments],
            roles=roles,
            compatibility_reason=reason,
            scores=ScoreBreakdown(**components, final_score=final_score),
        )

        if base_combo_key is None:
            scored_by_key[frozenset(g.id for g in garments)] = final_score
            scored.append(candidate)
        else:
            pending_layered.append((candidate, base_combo_key))

    for candidate, base_combo_key in pending_layered:
        base_score = scored_by_key.get(base_combo_key)
        if base_score is not None and candidate.scores.final_score <= base_score + LAYER_IMPROVEMENT_EPSILON:
            continue  # the extra layer doesn't improve on the base outfit — not worth suggesting
        scored.append(candidate)

    diversified = apply_diversity_penalty(scored)
    return RankingTrace(
        outfits=diversified[:top_k],
        total_evaluated=len(combinations_with_retrieval),
        total_compatible=len(scored),
        pairing_rejected=pairing_rejected,
    )
