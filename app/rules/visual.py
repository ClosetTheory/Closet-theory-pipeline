"""Deterministic Visual Compatibility Rules with VLM Fallback Trigger (Stage 9).

Evaluates visual compatibility using deterministic rules for color harmony,
pattern density, and formality. Triggers VLM fallback only when rules are not confident.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from app.config import settings

VISUAL_RULE_VERSION = "visual_v1"

NEUTRAL_COLORS: Set[str] = {
    "black", "white", "gray", "grey", "navy", "beige",
    "cream", "khaki", "brown", "tan", "charcoal", "ivory", "olive"
}

BUSY_PATTERNS: Set[str] = {
    "floral", "plaid", "checkered", "animal_print", "geometric", "abstract", "polka_dot"
}

FORMALITY_SCORES: Dict[str, int] = {
    "formal": 5,
    "evening": 5,
    "business_casual": 4,
    "work": 4,
    "smart_casual": 3,
    "party": 3,
    "casual": 2,
    "lounge": 1,
    "activewear": 1,
}


def _get_max_formality(occasions: List[str]) -> int:
    if not occasions:
        return 2  # default casual
    scores = [FORMALITY_SCORES.get(occ.lower(), 2) for occ in occasions]
    return max(scores)


def evaluate_visual_rules(
    garment_a_attrs: Dict[str, Any],
    garment_b_attrs: Dict[str, Any],
) -> Tuple[bool, str, float, str, str]:
    """
    Evaluates deterministic visual compatibility rules.
    Returns: (confident: bool, decision: str, score: float, reason: str, rule_version: str)

    If confident is False, the pipeline must invoke the VLM provider fallback.
    """
    pattern_a = garment_a_attrs.get("pattern", "solid").lower()
    pattern_b = garment_b_attrs.get("pattern", "solid").lower()

    colors_a = [c.lower() for c in garment_a_attrs.get("colour", [])]
    colors_b = [c.lower() for c in garment_b_attrs.get("colour", [])]

    occasions_a = garment_a_attrs.get("occasion", [])
    occasions_b = garment_b_attrs.get("occasion", [])

    # Check 1: Severe Pattern Clash
    if pattern_a in BUSY_PATTERNS and pattern_b in BUSY_PATTERNS and pattern_a != pattern_b:
        return (
            True,
            "INCOMPATIBLE",
            0.20,
            f"Pattern collision: Competing high-density patterns ({pattern_a} vs {pattern_b}) create visual discord.",
            VISUAL_RULE_VERSION,
        )

    # Check 2: Formality Clash (e.g. gym shorts with tuxedo jacket)
    formality_a = _get_max_formality(occasions_a)
    formality_b = _get_max_formality(occasions_b)
    if abs(formality_a - formality_b) >= 3:
        return (
            True,
            "INCOMPATIBLE",
            0.25,
            f"Severe dress-code mismatch: Level {formality_a} formality clashes with Level {formality_b} piece.",
            VISUAL_RULE_VERSION,
        )

    # Check 3: Solid Anchor with Patterned Piece
    if (pattern_a == "solid" and pattern_b in BUSY_PATTERNS) or (pattern_b == "solid" and pattern_a in BUSY_PATTERNS):
        # If at least one color is neutral, it's a confident match
        has_neutral = any(c in NEUTRAL_COLORS for c in colors_a + colors_b)
        if has_neutral:
            return (
                True,
                "COMPATIBLE",
                0.92,
                "Balanced composition: Solid neutral piece grounds the statement patterned garment.",
                VISUAL_RULE_VERSION,
            )

    # Check 4: Neutral Color Harmony
    all_colors_a_neutral = all(c in NEUTRAL_COLORS for c in colors_a) if colors_a else False
    all_colors_b_neutral = all(c in NEUTRAL_COLORS for c in colors_b) if colors_b else False
    if all_colors_a_neutral or all_colors_b_neutral:
        return (
            True,
            "COMPATIBLE",
            0.94,
            "Versatile neutral palette: Neutral base provides seamless chromatic pairing.",
            VISUAL_RULE_VERSION,
        )

    # Check 5: Monochrome / Tone-on-tone (Shared primary non-neutral color)
    shared_colors = set(colors_a).intersection(set(colors_b))
    if shared_colors and pattern_a == "solid" and pattern_b == "solid":
        return (
            True,
            "COMPATIBLE",
            0.88,
            f"Monochromatic tonal harmony: Shared color palette ({', '.join(shared_colors)}).",
            VISUAL_RULE_VERSION,
        )

    # If none of the deterministic rules are sufficiently conclusive (e.g. two vibrant contrasting colors),
    # trigger VLM fallback
    return (
        False,
        "REVIEW_REQUIRED",
        0.50,
        "Color/texture combination requires nuanced visual perception beyond deterministic rules.",
        VISUAL_RULE_VERSION,
    )
