"""Styling Stage 7 (part 2): Outfit Ranking via the weighted scorer.

Builds an OutfitCandidate (with full score breakdown) for every surviving
combination, using rule-based signals wherever possible and only the neutral
stubs (wardrobe_behavior, user_preference, novelty) where no real data exists yet.
"""

from typing import List
from app.models.garment import Garment
from app.rules.scoring import DEFAULT_STYLING_WEIGHTS, apply_diversity_penalty, compute_outfit_score
from app.rules.taxonomy import bundle_category
from app.rules.visual import FORMALITY_SCORES, NEUTRAL_COLORS, _get_max_formality
from app.rules.wardrobe_behavior import score_wardrobe_behavior
from app.schemas.styling import OutfitCandidate, ScoreBreakdown, StylingIntent
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


def _occasion_fit(garments: List[Garment], intent: StylingIntent) -> float:
    if not intent.formality:
        return 0.7

    target_score = FORMALITY_SCORES.get(intent.formality.lower(), 3)
    garment_scores = [
        _get_max_formality((g.attributes_json or {}).get("occasion", [])) for g in garments
    ]
    if not garment_scores:
        return 0.5
    avg_diff = sum(abs(target_score - s) for s in garment_scores) / len(garment_scores)
    return max(0.0, 1.0 - (avg_diff / 5.0))


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


class RankingTrace:
    """Bundles ranking-stage telemetry alongside the diversified outfits, for the pipeline detail view."""

    def __init__(self, outfits: List[OutfitCandidate], total_evaluated: int, total_compatible: int, pairing_rejected: int = 0):
        self.outfits = outfits
        self.total_evaluated = total_evaluated
        self.total_compatible = total_compatible
        self.pairing_rejected = pairing_rejected


async def rank_combinations(
    combinations_with_retrieval: List[tuple],
    intent: StylingIntent,
    top_k: int,
) -> RankingTrace:
    """Evaluates compatibility + scores every combination, drops INCOMPATIBLE ones, returns diversified top_k."""
    scored: List[OutfitCandidate] = []
    pairing_rejected = 0

    for garments, _retrieval_sum in combinations_with_retrieval:
        decision, compatibility_score, reason, hard_reject_source = await evaluate_outfit_compatibility(garments)
        if decision == "INCOMPATIBLE":
            if hard_reject_source == "pairing":
                pairing_rejected += 1
            continue

        components = {
            "request_match": _request_match(garments, intent),
            "compatibility": compatibility_score,
            "user_preference": 0.5,  # stub: no StyleProfile data yet
            "occasion_fit": _occasion_fit(garments, intent),
            "visual_harmony": await _visual_harmony(garments),
            "wardrobe_behavior": sum(score_wardrobe_behavior(g) for g in garments) / len(garments),
            "novelty": 1.0,  # stub: no WearLog history to compare against yet
        }
        final_score = compute_outfit_score(components, DEFAULT_STYLING_WEIGHTS)

        roles = {g.id: (resolve_role(g) or "UNKNOWN") for g in garments}
        scored.append(
            OutfitCandidate(
                garment_ids=[g.id for g in garments],
                roles=roles,
                compatibility_reason=reason,
                scores=ScoreBreakdown(**components, final_score=final_score),
            )
        )

    diversified = apply_diversity_penalty(scored)
    return RankingTrace(
        outfits=diversified[:top_k],
        total_evaluated=len(combinations_with_retrieval),
        total_compatible=len(scored),
        pairing_rejected=pairing_rejected,
    )
