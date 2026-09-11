"""Unit tests for visual.py v2's new color-theory signals (Option B) — hue-distance
analogous/complementary relationships and colour_temperature/colour_saturation harmony,
covering the class of non-neutral, non-identical pairings that v1 could only ever call
"not confident" regardless of how genuinely well the colors worked together.
"""

from app.rules.visual import evaluate_visual_rules


def test_analogous_hues_are_confident_compatible():
    # Teal + forest green: adjacent hues (175 vs 130 = 45... use a closer pair)
    teal = {"pattern": "solid", "colour": ["teal"], "occasion": ["casual"]}
    turquoise = {"pattern": "solid", "colour": ["turquoise"], "occasion": ["casual"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(teal, turquoise)
    assert confident is True
    assert decision == "COMPATIBLE"
    assert score >= 0.85
    assert "Analogous" in reason


def test_complementary_hues_are_confident_compatible():
    # Burgundy (345) + forest green (130): hue distance ~145... need >=150 for complementary.
    # Use navy blue (225) + mustard (50): distance = |225-50| = 175 -> complementary.
    navy_blue = {"pattern": "solid", "colour": ["navy blue"], "occasion": ["work"]}
    mustard = {"pattern": "solid", "colour": ["mustard"], "occasion": ["work"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(navy_blue, mustard)
    assert confident is True
    assert decision == "COMPATIBLE"
    assert score >= 0.80
    assert "Complementary" in reason


def test_hue_gap_between_analogous_and_complementary_stays_uncertain():
    # Orange (30) + turquoise (174): distance 144 sits between the analogous (<=40) and
    # complementary (>=150) windows on purpose — a genuinely ambiguous case that should
    # still defer to VLM judgment rather than guess confidently either way.
    orange = {"pattern": "solid", "colour": ["orange"], "occasion": ["party"]}
    turquoise = {"pattern": "solid", "colour": ["turquoise"], "occasion": ["party"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(orange, turquoise)
    assert confident is False
    assert decision == "REVIEW_REQUIRED"


def test_both_muted_saturation_is_confident_compatible_even_without_hue_match():
    # Colors not in COLOR_HUES at all, but both explicitly muted -> temperature/saturation
    # signal should still resolve this confidently.
    a = {"pattern": "solid", "colour": ["dusty blue"], "occasion": ["work"],
         "colour_temperature": "cool", "colour_saturation": "muted"}
    b = {"pattern": "solid", "colour": ["sage green"], "occasion": ["work"],
         "colour_temperature": "cool", "colour_saturation": "muted"}

    confident, decision, score, reason, _ = evaluate_visual_rules(a, b)
    assert confident is True
    assert decision == "COMPATIBLE"
    assert score >= 0.85


def test_vivid_accent_against_muted_base_is_confident_compatible():
    a = {"pattern": "solid", "colour": ["dusty blue"], "occasion": ["party"],
         "colour_temperature": "cool", "colour_saturation": "muted"}
    b = {"pattern": "solid", "colour": ["hot coral"], "occasion": ["party"],
         "colour_temperature": "warm", "colour_saturation": "vivid"}

    confident, decision, score, reason, _ = evaluate_visual_rules(a, b)
    assert confident is True
    assert decision == "COMPATIBLE"
    assert "accent" in reason.lower()


def test_two_vivid_opposing_temperatures_stays_uncertain():
    a = {"pattern": "solid", "colour": ["electric blue"], "occasion": ["party"],
         "colour_temperature": "cool", "colour_saturation": "vivid"}
    b = {"pattern": "solid", "colour": ["hot coral"], "occasion": ["party"],
         "colour_temperature": "warm", "colour_saturation": "vivid"}

    confident, decision, score, reason, _ = evaluate_visual_rules(a, b)
    assert confident is False
    assert decision == "REVIEW_REQUIRED"


def test_no_color_theory_data_falls_back_to_old_behavior():
    # No recognized hue, no temperature/saturation attrs at all -> same generic fallback v1 had.
    a = {"pattern": "solid", "colour": ["mystery shade one"], "occasion": ["casual"]}
    b = {"pattern": "solid", "colour": ["mystery shade two"], "occasion": ["casual"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(a, b)
    assert confident is False
    assert decision == "REVIEW_REQUIRED"
    assert score == 0.50
