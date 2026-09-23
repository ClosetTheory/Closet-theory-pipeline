"""Unit tests for anchors-only outfit assembly (styling Stage 6/7, combinator.py).

When a stylist anchors garments, the outfit must be built from those garments and nothing
else — no slot may be filled from the wardrobe-wide retrieval pool.
"""

from app.models.garment import Garment
from app.styling.combinator import anchors_cover_body, build_outfit_combinations


def _g(gid: str, category: str, subcategory: str = "") -> Garment:
    return Garment(id=gid, tenant_id="t", member_id="m", source_image_id="src", category=category,
                   subcategory=subcategory or category.lower(), status="COMPLETED", quality_status="APPROVED")


def _ids(combo) -> set:
    return {g.id for g in combo[0]}


# A wardrobe-wide retrieval pool that must never leak into anchors-only combinations.
WARDROBE_POOL = {
    "TOP": [(_g("w_top", "TOP"), 0.9)],
    "BOTTOM": [(_g("w_bottom", "BOTTOM"), 0.9)],
    "FOOTWEAR": [(_g("w_shoes", "FOOTWEAR"), 0.9)],
    "OUTERWEAR": [(_g("w_coat", "OUTERWEAR"), 0.9)],
}


def test_anchors_only_never_uses_wardrobe_pool():
    anchors = [_g("a_top", "TOP"), _g("a_bottom", "BOTTOM")]
    combos = build_outfit_combinations(WARDROBE_POOL, anchors, anchors_only=True)

    assert len(combos) == 1
    assert _ids(combos[0]) == {"a_top", "a_bottom"}
    for combo in combos:
        assert not any(g.id.startswith("w_") for g in combo[0])


def test_default_mode_still_fills_missing_roles_from_wardrobe():
    """Regression guard for the non-anchored path: without anchors_only the old behaviour
    (anchors lock their own role, other slots come from retrieval) is unchanged."""
    anchors = [_g("a_top", "TOP")]
    combos = build_outfit_combinations(WARDROBE_POOL, anchors)

    assert combos
    assert all("a_top" in _ids(c) for c in combos)
    assert any("w_bottom" in _ids(c) for c in combos)


def test_anchors_only_varies_across_multiple_anchors_per_role():
    anchors = [_g("top1", "TOP"), _g("top2", "TOP"), _g("bot1", "BOTTOM"), _g("bot2", "BOTTOM"), _g("shoes", "FOOTWEAR")]
    combos = build_outfit_combinations({}, anchors, anchors_only=True)

    assert {frozenset(_ids(c)) for c in combos} == {
        frozenset({"top1", "bot1", "shoes"}), frozenset({"top1", "bot2", "shoes"}),
        frozenset({"top2", "bot1", "shoes"}), frozenset({"top2", "bot2", "shoes"}),
    }


def test_anchored_outerwear_is_required_not_optional():
    anchors = [_g("top", "TOP"), _g("bottom", "BOTTOM"), _g("blazer", "OUTERWEAR")]
    combos = build_outfit_combinations({}, anchors, anchors_only=True)

    assert len(combos) == 1
    assert _ids(combos[0]) == {"top", "bottom", "blazer"}
    assert combos[0][2] is None, "anchored outerwear must not be tagged as a droppable layered variant"


def test_anchored_accessories_ride_along_on_every_combination():
    anchors = [_g("dress", "ONE_PIECE"), _g("top", "TOP"), _g("bottom", "BOTTOM"), _g("belt", "ACCESSORY")]
    combos = build_outfit_combinations({}, anchors, anchors_only=True)

    assert {frozenset(_ids(c)) for c in combos} == {frozenset({"dress", "belt"}), frozenset({"top", "bottom", "belt"})}


def test_anchors_that_cannot_dress_a_body_yield_nothing():
    assert build_outfit_combinations(WARDROBE_POOL, [_g("top", "TOP")], anchors_only=True) == []
    assert build_outfit_combinations(WARDROBE_POOL, [_g("shoes", "FOOTWEAR")], anchors_only=True) == []


def test_anchors_cover_body():
    assert anchors_cover_body([_g("t", "TOP"), _g("b", "BOTTOM")])
    assert anchors_cover_body([_g("d", "ONE_PIECE")])
    assert not anchors_cover_body([_g("t", "TOP")])
    assert not anchors_cover_body([_g("t", "TOP"), _g("s", "FOOTWEAR")])
    assert not anchors_cover_body([])
