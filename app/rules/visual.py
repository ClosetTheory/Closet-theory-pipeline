"""Deterministic Visual Compatibility Rules with VLM Fallback Trigger (Stage 9).

Evaluates visual compatibility using deterministic rules for color harmony,
pattern density, and formality. Triggers VLM fallback only when rules are not confident.

v2 adds real color-theory signals that v1 was missing entirely — v1 could only
confidently call a pairing "good" when one side was a flat neutral or the two shared an
identical color; every other non-clashing combination (e.g. burgundy + forest green,
navy + mustard, orange + turquoise) fell through to a flat 0.50 "not confident" score
regardless of how genuinely well those colors work together. That's the direct mechanism
behind outfits reading as "safe" rather than "styled" — the scorer had no way to recognize
an intentional, elevated color pairing as anything other than uncertain.

Two new signals, checked only for pairs the v1 rules didn't already resolve:
  1. Hue-distance color theory (COLOR_HUES) — recognizes analogous (close hues, e.g. teal +
     forest green) and complementary (near-opposite hues, e.g. burgundy + forest green)
     relationships for the subset of common fashion color names with a well-defined hue.
  2. colour_temperature / colour_saturation attributes — already extracted at ingestion
     (Stage 3) into every garment's attributes_json but never previously read by this
     module. These are enum-constrained (warm/cool/neutral, muted/mid/vivid/fluorescent),
     so unlike free-text color names they're reliably present, giving broader coverage
     than the hue table for colors it doesn't recognize by name (a muted+muted pairing is
     almost always harmonious regardless of exact hue; an intentional vivid-against-muted
     contrast reads as styled, two competing vivids in opposing temperatures reads as
     genuinely uncertain and is correctly left for VLM judgment, same as before).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from app.config import settings

VISUAL_RULE_VERSION = "visual_v2"

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


# Approximate HSV hue (degrees, 0-360) for common fashion color names that aren't already
# flat neutrals (those are handled by NEUTRAL_COLORS instead — a neutral has no meaningful
# "hue relationship" in the styling sense). Deliberately not exhaustive: a color name absent
# here just means the hue-based check below is skipped for it, falling through to the
# temperature/saturation check or, failing that, the same VLM fallback v1 always had — this
# table only ever adds confidence, it never removes it.
COLOR_HUES: Dict[str, float] = {
    "red": 0, "maroon": 350, "burgundy": 345, "wine": 340, "crimson": 5,
    "orange": 30, "rust": 18, "coral": 16, "peach": 25, "terracotta": 15,
    "yellow": 55, "mustard": 50, "gold": 48,
    "lime": 90, "green": 120, "forest green": 130, "emerald": 145, "sage": 95,
    "mint": 155, "dark green": 130,
    "teal": 175, "turquoise": 174, "cyan": 185,
    "blue": 220, "navy blue": 225, "royal blue": 222, "sky blue": 200,
    "cobalt": 215, "denim blue": 210,
    "purple": 270, "lavender": 260, "violet": 280, "plum": 300, "lilac": 265,
    "magenta": 320, "pink": 335, "hot pink": 328, "fuchsia": 315,
    "dusty rose": 350, "light pink": 335, "rose": 345,
}

ANALOGOUS_MAX_DIFF = 40.0  # hues within this distance read as a deliberate tonal family
COMPLEMENTARY_MIN_DIFF = 150.0  # hues at least this far apart read as intentional contrast


def _hue_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _hue_relationship(colors_a: List[str], colors_b: List[str]) -> Optional[Tuple[str, float]]:
    """Returns (relationship, best_hue_distance) for the closest-matching pair of named hues
    across the two garments' color lists, or None if neither side has a recognized hue."""
    hues_a = [COLOR_HUES[c] for c in colors_a if c in COLOR_HUES]
    hues_b = [COLOR_HUES[c] for c in colors_b if c in COLOR_HUES]
    if not hues_a or not hues_b:
        return None

    best_diff = min(_hue_distance(ha, hb) for ha in hues_a for hb in hues_b)
    if best_diff <= ANALOGOUS_MAX_DIFF:
        return "analogous", best_diff
    if best_diff >= COMPLEMENTARY_MIN_DIFF:
        return "complementary", best_diff
    return None


def _temperature_saturation_harmony(
    attrs_a: Dict[str, Any], attrs_b: Dict[str, Any]
) -> Optional[Tuple[bool, float, str]]:
    """Uses the colour_temperature/colour_saturation attributes (populated at ingestion, enum-
    constrained so reliably present even when the free-text color name isn't in COLOR_HUES) to
    judge a pairing the hue table couldn't resolve. Returns (is_good, score, reason) or None if
    either garment is missing this data."""
    temp_a, temp_b = attrs_a.get("colour_temperature"), attrs_b.get("colour_temperature")
    sat_a, sat_b = attrs_a.get("colour_saturation"), attrs_b.get("colour_saturation")
    if not temp_a or not temp_b or not sat_a or not sat_b:
        return None

    muted_a, muted_b = sat_a in ("muted", "mid"), sat_b in ("muted", "mid")
    vivid_a, vivid_b = sat_a in ("vivid", "fluorescent"), sat_b in ("vivid", "fluorescent")

    if muted_a and muted_b:
        return True, 0.90, "Both muted/mid-saturation tones — an inherently low-risk, sophisticated pairing."

    same_temperature = temp_a == temp_b and temp_a != "neutral"
    if same_temperature and not (vivid_a and vivid_b):
        return True, 0.86, f"Shared {temp_a} colour temperature with at least one softened tone — reads as a cohesive palette."

    opposite_temperature = {temp_a, temp_b} == {"warm", "cool"}
    if opposite_temperature and (muted_a != muted_b):
        # exactly one side vivid, the other muted/mid: a deliberate accent against a calm base
        return True, 0.84, "Warm/cool contrast balanced by one muted tone — reads as an intentional accent, not a clash."

    if opposite_temperature and vivid_a and vivid_b:
        return False, 0.50, "Two vivid colours in opposing temperatures — genuinely could clash or pop depending on exact shades."

    if same_temperature and vivid_a and vivid_b:
        return False, 0.55, "Two vivid tones in the same family — risk of visually competing rather than complementing."

    return None


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

    # Check 6 (new in v2): Hue-distance color theory — analogous or complementary relationship
    # between two non-neutral, non-identical named colors that the checks above couldn't
    # already resolve (this only runs once neutral/monochrome are ruled out).
    hue_relationship = _hue_relationship(colors_a, colors_b)
    if hue_relationship:
        relationship, diff = hue_relationship
        if relationship == "analogous":
            return (
                True,
                "COMPATIBLE",
                0.87,
                f"Analogous colour palette: Adjacent hues ({diff:.0f}\N{DEGREE SIGN} apart) read as a deliberate tonal family.",
                VISUAL_RULE_VERSION,
            )
        return (
            True,
            "COMPATIBLE",
            0.83,
            f"Complementary colour contrast: Near-opposite hues ({diff:.0f}\N{DEGREE SIGN} apart) create an intentional, styled pop of contrast.",
            VISUAL_RULE_VERSION,
        )

    # Check 7 (new in v2): colour_temperature/colour_saturation harmony — covers colors not in
    # the named-hue table above, since these two attributes are enum-constrained and populated
    # at ingestion regardless of the exact free-text color name.
    temp_sat_result = _temperature_saturation_harmony(garment_a_attrs, garment_b_attrs)
    if temp_sat_result:
        is_good, score, reason = temp_sat_result
        if is_good:
            return (True, "COMPATIBLE", score, reason, VISUAL_RULE_VERSION)
        # Still genuinely uncertain even with temperature/saturation data (e.g. two vivid,
        # opposing-temperature colors) — same honest "defer to VLM" outcome as v1, just with
        # a more specific reason attached instead of a generic catch-all message.
        return (False, "REVIEW_REQUIRED", score, reason, VISUAL_RULE_VERSION)

    # If none of the deterministic rules are sufficiently conclusive (e.g. two vibrant contrasting colors),
    # trigger VLM fallback
    return (
        False,
        "REVIEW_REQUIRED",
        0.50,
        "Color/texture combination requires nuanced visual perception beyond deterministic rules.",
        VISUAL_RULE_VERSION,
    )
