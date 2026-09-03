"""Deterministic Category Bundling (Stage 6).

Implements a deterministic lookup table mapping subcategories to canonical categories.
No LLMs, fully versioned, unknown subcategories route to explicit fallback/review.
"""

from typing import Optional, Tuple
from app.config import settings

TAXONOMY_VERSION = "taxonomy_v1"

# Canonical high-level categories: TOP, BOTTOM, ONE_PIECE, OUTERWEAR, FOOTWEAR, ACCESSORY
CATEGORY_MAP = {
    # Tops
    "oxford_shirt": "TOP",
    "button_down_shirt": "TOP",
    "dress_shirt": "TOP",
    "flannel_shirt": "TOP",
    "tshirt": "TOP",
    "polo_shirt": "TOP",
    "henley": "TOP",
    "tank_top": "TOP",
    "crop_top": "TOP",
    "blouse": "TOP",
    "sweater": "TOP",
    "cardigan": "TOP",
    "hoodie": "TOP",
    "sweatshirt": "TOP",
    "vest": "TOP",
    # Bottoms
    "jeans": "BOTTOM",
    "trousers": "BOTTOM",
    "chinos": "BOTTOM",
    "dress_pants": "BOTTOM",
    "cargo_pants": "BOTTOM",
    "shorts": "BOTTOM",
    "sweatpants": "BOTTOM",
    "leggings": "BOTTOM",
    "skirt": "BOTTOM",
    "mini_skirt": "BOTTOM",
    "midi_skirt": "BOTTOM",
    "maxi_skirt": "BOTTOM",
    # One Piece
    "dress": "ONE_PIECE",
    "sundress": "ONE_PIECE",
    "maxi_dress": "ONE_PIECE",
    "jumpsuit": "ONE_PIECE",
    "romper": "ONE_PIECE",
    "overalls": "ONE_PIECE",
    # Outerwear
    "blazer": "OUTERWEAR",
    "suit_jacket": "OUTERWEAR",
    "coat": "OUTERWEAR",
    "trench_coat": "OUTERWEAR",
    "overcoat": "OUTERWEAR",
    "parka": "OUTERWEAR",
    "leather_jacket": "OUTERWEAR",
    "denim_jacket": "OUTERWEAR",
    "bomber_jacket": "OUTERWEAR",
    "puffer_jacket": "OUTERWEAR",
    "windbreaker": "OUTERWEAR",
    # Footwear
    "sneakers": "FOOTWEAR",
    "boots": "FOOTWEAR",
    "loafers": "FOOTWEAR",
    "oxfords": "FOOTWEAR",
    "derby": "FOOTWEAR",
    "sandals": "FOOTWEAR",
    "heels": "FOOTWEAR",
    "flats": "FOOTWEAR",
    # Accessories
    "belt": "ACCESSORY",
    "scarf": "ACCESSORY",
    "hat": "ACCESSORY",
    "cap": "ACCESSORY",
    "tie": "ACCESSORY",
    "gloves": "ACCESSORY",
    "bag": "ACCESSORY",
    "sunglasses": "ACCESSORY",
}


def bundle_category(subcategory: str) -> Tuple[Optional[str], str, bool]:
    """
    Looks up canonical category for a given subcategory.
    Returns: (canonical_category, taxonomy_version, requires_review)
    """
    normalized = subcategory.lower().strip().replace(" ", "_").replace("-", "_")
    category = CATEGORY_MAP.get(normalized)
    if category is None:
        # Unknown category routes to explicit fallback/review
        return None, TAXONOMY_VERSION, True
    return category, TAXONOMY_VERSION, False
