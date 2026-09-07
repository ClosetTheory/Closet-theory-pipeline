"""Styling Stage 6: Candidate Compatibility Analysis.

Reuses the same deterministic rule functions as app/api/v1/compatibility.py,
pairwise across every garment in a candidate outfit combination, with VLM
fallback only when the visual rules are inconclusive (mirrors compatibility.py's
rules-first cascade exactly).

SPEC.md Section 16/17: Structural and Layering (and, per this implementation's
Pairing addition — see app/rules/pairing.py) failures hard-reject a combination.
Visual dissociation must only ever produce a soft score penalty, never a hard
reject — this is enforced here by never letting the visual rule's decision set
`worst_decision = INCOMPATIBLE`.
"""

from itertools import combinations
from typing import List, Optional, Tuple
from app.models.garment import Garment
from app.pipeline.image_resolution import resolve_garment_image_bytes
from app.providers.vlm import get_vlm_provider
from app.rules.layering import evaluate_layering_compatibility
from app.rules.pairing import evaluate_pairing_compatibility
from app.rules.structural import evaluate_structural_compatibility
from app.rules.visual import evaluate_visual_rules
from app.storage import get_storage_client


def _attrs(garment: Garment) -> dict:
    return {**(garment.attributes_json or {}), "category": garment.category}


async def evaluate_pair_compatibility(garment_a: Garment, garment_b: Garment) -> Tuple[str, float, str, Optional[str]]:
    """
    Returns (decision, score, reason, hard_reject_source) for a single garment pair.
    hard_reject_source names which check ("pairing"/"layering"/"structural") caused an
    INCOMPATIBLE decision, or None if the pair wasn't hard-rejected. Visual never sets it.
    """
    attrs_a, attrs_b = _attrs(garment_a), _attrs(garment_b)
    reasons = []
    scores = []
    worst_decision = "COMPATIBLE"
    hard_reject_source: Optional[str] = None

    for source, (decision, score, reason, _ver) in (
        ("pairing", evaluate_pairing_compatibility(attrs_a, attrs_b)),
        ("layering", evaluate_layering_compatibility(attrs_a, attrs_b)),
        ("structural", evaluate_structural_compatibility(attrs_a, attrs_b)),
    ):
        reasons.append(reason)
        scores.append(score)
        if decision == "INCOMPATIBLE":
            worst_decision = "INCOMPATIBLE"
            hard_reject_source = source
        elif decision == "REVIEW_REQUIRED" and worst_decision != "INCOMPATIBLE":
            worst_decision = "REVIEW_REQUIRED"

    # Visual dissociation is a soft signal only (SPEC.md Section 16) — it can never
    # escalate worst_decision to INCOMPATIBLE, only to REVIEW_REQUIRED at most.
    confident, visual_decision, visual_score, visual_reason, _ver = evaluate_visual_rules(attrs_a, attrs_b)
    if not confident:
        storage = get_storage_client()
        image_a_bytes = await resolve_garment_image_bytes(storage, garment_a)
        image_b_bytes = await resolve_garment_image_bytes(storage, garment_b)
        vlm = get_vlm_provider()
        visual_decision, visual_score, visual_reason = await vlm.evaluate_visual_compatibility(
            image_a_bytes, image_b_bytes, attrs_a, attrs_b
        )
    reasons.append(visual_reason)
    scores.append(visual_score)
    if visual_decision in ("INCOMPATIBLE", "REVIEW_REQUIRED") and worst_decision != "INCOMPATIBLE":
        worst_decision = "REVIEW_REQUIRED"

    avg_score = sum(scores) / len(scores) if scores else 0.5

    # Co-ord override: these two garments came from the SAME source photo (Stage 2 spawns one
    # Garment per detected item in a single full-body shot — see stage_02_crop.py), meaning
    # they're not a hypothetical pairing but a fact — the person was actually photographed
    # wearing both together. A soft visual-dissociation flag is moot once that's known, so
    # REVIEW_REQUIRED from visual disagreement alone is downgraded to COMPATIBLE with a high
    # floor score. A genuine hard reject (pairing/layering/structural — e.g. two bottoms) still
    # stands: co-ingestion doesn't override a physically impossible combination, it can only
    # mean the detector mis-split one garment into two.
    if garment_a.source_image_id == garment_b.source_image_id and worst_decision != "INCOMPATIBLE":
        worst_decision = "COMPATIBLE"
        avg_score = max(avg_score, 0.95)
        reasons.insert(0, "Ingested together from the same photo — known to have been worn as a set.")

    return worst_decision, avg_score, " | ".join(reasons), hard_reject_source


async def evaluate_outfit_compatibility(garments: List[Garment]) -> Tuple[str, float, str, Optional[str]]:
    """
    Evaluates all pairs in a candidate outfit combination.
    Any pairing/layering/structural INCOMPATIBLE pair hard-rejects the whole combination;
    visual dissociation only ever lowers the score (SPEC.md Section 17).
    Returns (decision, score, reason, hard_reject_source).
    """
    if len(garments) < 2:
        return "COMPATIBLE", 1.0, "Single-garment outfit; no pairwise compatibility to evaluate.", None

    pair_scores = []
    pair_reasons = []
    overall = "COMPATIBLE"

    for garment_a, garment_b in combinations(garments, 2):
        decision, score, reason, hard_reject_source = await evaluate_pair_compatibility(garment_a, garment_b)
        pair_scores.append(score)
        pair_reasons.append(f"{garment_a.subcategory}+{garment_b.subcategory}: {reason}")
        if decision == "INCOMPATIBLE":
            return "INCOMPATIBLE", score, reason, hard_reject_source
        if decision == "REVIEW_REQUIRED" and overall != "INCOMPATIBLE":
            overall = "REVIEW_REQUIRED"

    avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
    return overall, avg_score, "; ".join(pair_reasons), None
