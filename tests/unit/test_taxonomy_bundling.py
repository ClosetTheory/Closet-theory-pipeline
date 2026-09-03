"""Unit tests for Stage 6: Category Bundling Lookup Table."""

from app.rules.taxonomy import bundle_category, TAXONOMY_VERSION


def test_known_tops_bundling():
    cat, ver, review = bundle_category("oxford_shirt")
    assert cat == "TOP"
    assert ver == TAXONOMY_VERSION
    assert review is False

    cat, _, _ = bundle_category("tshirt")
    assert cat == "TOP"


def test_known_bottoms_bundling():
    cat, _, review = bundle_category("jeans")
    assert cat == "BOTTOM"
    assert review is False

    cat, _, _ = bundle_category("trousers")
    assert cat == "BOTTOM"


def test_outerwear_and_footwear():
    cat, _, _ = bundle_category("blazer")
    assert cat == "OUTERWEAR"

    cat, _, _ = bundle_category("sneakers")
    assert cat == "FOOTWEAR"


def test_unknown_subcategory_routes_to_review():
    cat, ver, review = bundle_category("unknown_alien_armor")
    assert cat is None
    assert ver == TAXONOMY_VERSION
    assert review is True
