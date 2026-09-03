"""Unit tests for Stage 8: Structural Compatibility Rules."""

from app.rules.structural import evaluate_structural_compatibility, STRUCTURAL_RULE_VERSION


def test_top_and_bottom_complement():
    top = {"category": "TOP", "fit": "oversized", "silhouette": "straight", "layering_role": "base"}
    bottom = {"category": "BOTTOM", "fit": "slim", "silhouette": "straight", "layering_role": "standalone"}

    decision, score, reason, ver = evaluate_structural_compatibility(top, bottom)
    assert decision == "COMPATIBLE"
    assert score >= 0.90
    assert ver == STRUCTURAL_RULE_VERSION


def test_bottom_and_bottom_conflict():
    bottom_1 = {"category": "BOTTOM", "fit": "regular"}
    bottom_2 = {"category": "BOTTOM", "fit": "slim"}

    decision, score, reason, _ = evaluate_structural_compatibility(bottom_1, bottom_2)
    assert decision == "INCOMPATIBLE"
    assert "Slot conflict" in reason


def test_one_piece_with_bottom_conflict():
    dress = {"category": "ONE_PIECE", "layering_role": "standalone"}
    pants = {"category": "BOTTOM", "layering_role": "standalone"}

    decision, score, reason, _ = evaluate_structural_compatibility(dress, pants)
    assert decision == "INCOMPATIBLE"
    assert "overlaps" in reason.lower()


def test_footwear_with_apparel():
    shoes = {"category": "FOOTWEAR"}
    jeans = {"category": "BOTTOM"}

    decision, score, _, _ = evaluate_structural_compatibility(shoes, jeans)
    assert decision == "COMPATIBLE"
    assert score >= 0.90
