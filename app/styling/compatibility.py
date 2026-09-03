"""Styling Stage 6: Candidate Compatibility Analysis.

Reuses the same deterministic rule functions as app/api/v1/compatibility.py,
pairwise across every garment in a candidate outfit combination, with VLM
fallback only when the visual rules are inconclusive (mirrors compatibility.py's
rules-first cascade exactly).
"""

from itertools import combinations
from typing import List, Tuple
from app.models.garment import Garment
from app.providers.vlm import get_vlm_provider
from app.rules.layering import evaluate_layering_compatibility
from app.rules.structural import evaluate_structural_compatibility
from app.rules.visual import evaluate_visual_rules


def _attrs(garment: Garment) -> dict:
    return {**(garment.attributes_json or {}), "category": garment.category}


async def evaluate_pair_compatibility(garment_a: Garment, garment_b: Garment) -> Tuple[str, float, str]:
    """Returns (decision, score, reason) for a single garment pair, worst-decision-wins across rule types."""
    attrs_a, attrs_b = _attrs(garment_a), _attrs(garment_b)
    reasons = []
    scores = []
    worst_decision = "COMPATIBLE"

    for decision, score, reason, _ver in (
        evaluate_layering_compatibility(attrs_a, attrs_b),
        evaluate_structural_compatibility(attrs_a, attrs_b),
    ):
        reasons.append(reason)
        scores.append(score)
        if decision == "INCOMPATIBLE":
            worst_decision = "INCOMPATIBLE"
        elif decision == "REVIEW_REQUIRED" and worst_decision != "INCOMPATIBLE":
            worst_decision = "REVIEW_REQUIRED"

    confident, decision, score, reason, _ver = evaluate_visual_rules(attrs_a, attrs_b)
    if not confident:
        vlm = get_vlm_provider()
        decision, score, reason = await vlm.evaluate_visual_compatibility(None, None, attrs_a, attrs_b)
    reasons.append(reason)
    scores.append(score)
    if decision == "INCOMPATIBLE":
        worst_decision = "INCOMPATIBLE"
    elif decision == "REVIEW_REQUIRED" and worst_decision != "INCOMPATIBLE":
        worst_decision = "REVIEW_REQUIRED"

    avg_score = sum(scores) / len(scores) if scores else 0.5
    return worst_decision, avg_score, " | ".join(reasons)


async def evaluate_outfit_compatibility(garments: List[Garment]) -> Tuple[str, float, str]:
    """
    Evaluates all pairs in a candidate outfit combination.
    Any INCOMPATIBLE pair hard-rejects the whole combination (PRD Section 12).
    """
    if len(garments) < 2:
        return "COMPATIBLE", 1.0, "Single-garment outfit; no pairwise compatibility to evaluate."

    pair_scores = []
    pair_reasons = []
    overall = "COMPATIBLE"

    for garment_a, garment_b in combinations(garments, 2):
        decision, score, reason = await evaluate_pair_compatibility(garment_a, garment_b)
        pair_scores.append(score)
        pair_reasons.append(f"{garment_a.subcategory}+{garment_b.subcategory}: {reason}")
        if decision == "INCOMPATIBLE":
            return "INCOMPATIBLE", score, reason
        if decision == "REVIEW_REQUIRED" and overall != "INCOMPATIBLE":
            overall = "REVIEW_REQUIRED"

    avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
    return overall, avg_score, "; ".join(pair_reasons)
