"""Style-preference learning from outfit up/downvotes: one numeric scalar (boldness) plus
per-value affinities for categorical attributes (colour, pattern, material, fit, silhouette,
sleeve_length, garment_class) — both updated by the same vote signal, just with different math.

An outfit's "boldness" is derived from its visual_harmony score (already computed in
app/styling/ranking.py and persisted on Outfit.score_breakdown) — a highly conventional,
well-matched outfit has harmony near 1.0 and so boldness near 0.0; a clashing/distinctive
one has low harmony and high boldness. Voting nudges the member's learned
StyleProfile.boldness_preference toward (upvote) or away from (downvote) the voted
outfit's boldness.

Colour and pattern have no natural numeric scale, so instead of one scalar each we track a
per-*value* running score and vote count (e.g. affinities["colour"]["black"] = {"score": 0.34,
"count": 12}): an upvote nudges every attribute value present in that outfit toward +1, a
downvote nudges them toward -1. The count matters as much as the score — a value seen once
carries little weight (could easily be coincidence: liked *despite* that color, not because of
it), while a value confirmed across many votes should dominate. Two places use the count:

1. Confidence shrinkage: a value's own score is trusted more as its count grows (Bayesian-style
   shrinkage toward neutral for low counts), so one lucky/unlucky vote can't swing the profile.
2. Weighted combination: when an outfit has several tracked attribute values, the ones with the
   most accumulated votes dominate the outfit-level score — the "bulk of likes and dislikes"
   is what identifies which value is actually driving the preference, not just whichever one
   happens to be present.
"""

from typing import Any, Dict, List, Tuple

BOLDNESS_LEARNING_RATE = 0.2
ATTRIBUTE_LEARNING_RATE = 0.2

# How many votes a value needs before its learned score is trusted close to face value.
# confidence = count / (count + CONFIDENCE_PRIOR) -> ~50% trust at CONFIDENCE_PRIOR votes,
# asymptotically approaching full trust as votes accumulate.
CONFIDENCE_PRIOR = 5.0

# Which categorical GarmentAttributes fields get a learned per-value affinity. Every value here
# is a genuine style-preference signal (how something looks/feels), independent of the request
# itself — deliberately excludes fields that describe *fit-for-purpose* rather than preference:
# occasion/season (already driven by the request's formality/weather, see _occasion_fit and
# _weather_fit), layering_role (governed by the deterministic layering-compatibility rules, not
# taste), and brand_label/subcategory (too sparse or too fine-grained — would fragment votes
# across near-duplicate values instead of accumulating confidence on a shared one).
TRACKED_CATEGORICAL_ATTRIBUTES = ["colour", "pattern", "material", "fit", "silhouette", "sleeve_length", "garment_class"]


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


def _attribute_values(attrs: Dict[str, Any], field: str) -> List[str]:
    """GarmentAttributes fields are either a single string (e.g. pattern) or a list (colour)."""
    value = attrs.get(field)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).lower() for v in value if v]
    return [str(value).lower()]


def update_attribute_affinities(
    previous: Dict[str, Dict[str, Dict[str, float]]],
    garments_attrs: List[Dict[str, Any]],
    vote: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns a new affinities dict with every tracked categorical value seen across the
    voted outfit's garments nudged toward +1 (upvote) or -1 (downvote), and its vote count
    incremented. Shape: {field: {value: {"score": float in [-1,1], "count": int}}}."""
    if vote not in ("up", "down"):
        raise ValueError(f"Unknown vote type: {vote!r} (expected 'up' or 'down')")

    target = 1.0 if vote == "up" else -1.0
    updated = {field: {v: dict(stats) for v, stats in values.items()} for field, values in previous.items()}

    for field in TRACKED_CATEGORICAL_ATTRIBUTES:
        seen_values = {v for attrs in garments_attrs for v in _attribute_values(attrs, field)}
        if not seen_values:
            continue
        field_affinities = updated.setdefault(field, {})
        for value in seen_values:
            stats = field_affinities.get(value, {"score": 0.0, "count": 0})
            new_score = stats["score"] + ATTRIBUTE_LEARNING_RATE * (target - stats["score"])
            field_affinities[value] = {
                "score": max(-1.0, min(1.0, new_score)),
                "count": stats["count"] + 1,
            }

    return updated


def confidence_for_count(count: int) -> float:
    """How much a value's raw score should be trusted given how many votes back it — public so
    API/display layers (e.g. the profile page) can show it without duplicating the math."""
    return count / (count + CONFIDENCE_PRIOR)


def _confident_score(stats: Dict[str, float]) -> Tuple[float, float]:
    """Shrinks a value's raw score toward neutral (0) based on how many votes back it —
    returns (shrunk_score, count) so the caller can also weight by count when combining."""
    count = stats.get("count", 0)
    return stats.get("score", 0.0) * confidence_for_count(count), count


def attribute_affinity_score(garments_attrs: List[Dict[str, Any]], affinities: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    """Combines the learned, confidence-shrunk affinity of every tracked-attribute value
    present across the outfit's garments — weighted by each value's vote count, so values
    backed by more accumulated likes/dislikes dominate the combined signal — mapped from
    [-1, 1] to a [0, 1] score. No learned affinities yet (cold start) — neutral 0.5, same
    convention as the other "no signal" stubs in ranking.py."""
    if not affinities:
        return 0.5

    weighted_sum = 0.0
    total_weight = 0.0
    for field in TRACKED_CATEGORICAL_ATTRIBUTES:
        field_affinities = affinities.get(field)
        if not field_affinities:
            continue
        for attrs in garments_attrs:
            for value in _attribute_values(attrs, field):
                stats = field_affinities.get(value)
                if not stats:
                    continue
                shrunk_score, count = _confident_score(stats)
                weighted_sum += shrunk_score * count
                total_weight += count

    if total_weight == 0:
        return 0.5
    avg_affinity = weighted_sum / total_weight
    return (avg_affinity + 1.0) / 2.0
