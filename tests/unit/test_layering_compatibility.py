"""Unit tests for Stage 7: Layering Compatibility Rules."""

from app.rules.layering import evaluate_layering_compatibility, LAYERING_RULE_VERSION


def test_base_plus_mid():
    garment_a = {"layering_role": "base", "subcategory": "tshirt"}
    garment_b = {"layering_role": "mid", "subcategory": "cardigan"}

    decision, score, reason, ver = evaluate_layering_compatibility(garment_a, garment_b)
    assert decision == "COMPATIBLE"
    assert score >= 0.90
    assert ver == LAYERING_RULE_VERSION


def test_mid_plus_outer():
    garment_a = {"layering_role": "mid", "subcategory": "sweater"}
    garment_b = {"layering_role": "outer", "subcategory": "overcoat"}

    decision, score, _, _ = evaluate_layering_compatibility(garment_a, garment_b)
    assert decision == "COMPATIBLE"
    assert score >= 0.90


def test_outer_plus_outer_clash():
    garment_a = {"layering_role": "outer", "subcategory": "puffer_jacket"}
    garment_b = {"layering_role": "outer", "subcategory": "leather_jacket"}

    decision, score, reason, _ = evaluate_layering_compatibility(garment_a, garment_b)
    assert decision == "INCOMPATIBLE"
    assert score < 0.30
    assert "bulk" in reason.lower()


def test_accessory_compatibility():
    garment_a = {"layering_role": "accessory", "subcategory": "scarf"}
    garment_b = {"layering_role": "outer", "subcategory": "coat"}

    decision, score, _, _ = evaluate_layering_compatibility(garment_a, garment_b)
    assert decision == "COMPATIBLE"
    assert score == 1.0
