"""Weighted Outfit Scorer and Shortlist Diversity (Styling Pipeline Stage 7)."""

from typing import Dict, List
from app.config import settings
from app.schemas.styling import OutfitCandidate

STYLING_SCORER_VERSION = settings.STYLING_SCORER_VERSION

DEFAULT_STYLING_WEIGHTS: Dict[str, float] = {
    "request_match": 0.25,
    "compatibility": 0.20,
    "user_preference": 0.15,
    "occasion_fit": 0.15,
    "visual_harmony": 0.10,
    "wardrobe_behavior": 0.10,
    "novelty": 0.05,
}


def compute_outfit_score(components: Dict[str, float], weights: Dict[str, float] = DEFAULT_STYLING_WEIGHTS) -> float:
    """Weighted sum of score components, clamped to [0.0, 1.0]."""
    total = sum(weights.get(key, 0.0) * value for key, value in components.items())
    return max(0.0, min(1.0, total))


def _garment_set_similarity(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two outfits' garment ID sets."""
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def apply_diversity_penalty(outfits: List[OutfitCandidate]) -> List[OutfitCandidate]:
    """
    Greedily re-ranks outfits so the shortlist isn't near-duplicate combinations.
    final_score -= similarity_to_already_selected_outfits (PRD Section 15).
    """
    remaining = sorted(outfits, key=lambda o: o.scores.final_score, reverse=True)
    selected: List[OutfitCandidate] = []

    while remaining:
        if not selected:
            best = remaining.pop(0)
            selected.append(best)
            continue

        best_idx, best_adjusted = None, -1.0
        for idx, candidate in enumerate(remaining):
            max_similarity = max(
                _garment_set_similarity(candidate.garment_ids, s.garment_ids) for s in selected
            )
            adjusted = candidate.scores.final_score - max_similarity
            if adjusted > best_adjusted:
                best_idx, best_adjusted = idx, adjusted

        chosen = remaining.pop(best_idx)
        selected.append(chosen)

    return selected
