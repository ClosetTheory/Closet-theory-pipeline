"""Boldness-preference learning: EMA update from outfit up/downvotes.

An outfit's "boldness" is derived from its visual_harmony score (already computed in
app/styling/ranking.py and persisted on Outfit.score_breakdown) — a highly conventional,
well-matched outfit has harmony near 1.0 and so boldness near 0.0; a clashing/distinctive
one has low harmony and high boldness. Voting nudges the member's learned
StyleProfile.boldness_preference toward (upvote) or away from (downvote) the voted
outfit's boldness.
"""

BOLDNESS_LEARNING_RATE = 0.2


def outfit_boldness(visual_harmony_score: float) -> float:
    return max(0.0, min(1.0, 1.0 - visual_harmony_score))


def update_boldness_preference(previous: float, outfit_boldness_value: float, vote: str) -> float:
    """
    Exponential-moving-average update: pulled toward the voted outfit's boldness on an
    upvote, pushed away from it on a downvote. `previous`, `outfit_boldness_value`, and the
    return value are all in [0, 1].
    """
    if vote not in ("up", "down"):
        raise ValueError(f"Unknown vote type: {vote!r} (expected 'up' or 'down')")

    delta = outfit_boldness_value - previous
    direction = 1.0 if vote == "up" else -1.0
    updated = previous + direction * BOLDNESS_LEARNING_RATE * delta
    return max(0.0, min(1.0, updated))
