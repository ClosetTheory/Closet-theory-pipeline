"""Unit tests for Stage 9: Visual Compatibility Rules and VLM Fallback Trigger."""

from app.rules.visual import evaluate_visual_rules, VISUAL_RULE_VERSION


def test_pattern_clash():
    garment_a = {"pattern": "floral", "colour": ["red"], "occasion": ["casual"]}
    garment_b = {"pattern": "plaid", "colour": ["green"], "occasion": ["casual"]}

    confident, decision, score, reason, ver = evaluate_visual_rules(garment_a, garment_b)
    assert confident is True
    assert decision == "INCOMPATIBLE"
    assert "Pattern collision" in reason
    assert ver == VISUAL_RULE_VERSION


def test_formality_mismatch():
    tux = {"pattern": "solid", "colour": ["black"], "occasion": ["formal"]}
    sweatpants = {"pattern": "solid", "colour": ["grey"], "occasion": ["lounge", "activewear"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(tux, sweatpants)
    assert confident is True
    assert decision == "INCOMPATIBLE"
    assert "formality" in reason.lower()


def test_solid_neutral_anchor_with_pattern():
    shirt = {"pattern": "striped", "colour": ["blue", "white"], "occasion": ["casual"]}
    chinos = {"pattern": "solid", "colour": ["beige"], "occasion": ["casual"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(shirt, chinos)
    assert confident is True
    assert decision == "COMPATIBLE"
    assert score >= 0.90


def test_ambiguous_combination_triggers_vlm():
    # Non-neutral, non-pattern-clashing vibrant colors (e.g. orange and turquoise)
    item_a = {"pattern": "solid", "colour": ["orange"], "occasion": ["party"]}
    item_b = {"pattern": "solid", "colour": ["turquoise"], "occasion": ["party"]}

    confident, decision, score, reason, _ = evaluate_visual_rules(item_a, item_b)
    # Must declare not confident to trigger VLM
    assert confident is False
    assert decision == "REVIEW_REQUIRED"
