"""Wardrobe Behaviour Scoring (Styling Pipeline Stage 3).

V1 stub: no WearLog/StyleProfile data exists yet, so every garment gets a neutral
score. Kept as its own function/module so a real wear-history-based scorer can be
dropped in later without reshaping the ranking pipeline (PRD Section 8/32 Phase 10).
"""

from app.models.garment import Garment

NEUTRAL_BEHAVIOR_SCORE = 0.5


def score_wardrobe_behavior(garment: Garment) -> float:
    """
    Intended weighting once wear history exists:
        0.30 * wear_affinity + 0.25 * preference_affinity + 0.20 * historical_selection
        + 0.15 * freshness + 0.10 * style_affinity
    For V1 (no WearLog/StyleProfile tables), returns a constant neutral score.
    """
    return NEUTRAL_BEHAVIOR_SCORE
