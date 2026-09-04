"""Canonical Garment Class Taxonomy (SPEC.md Section 8-10).

Two-level taxonomy: garment_class (fine-grained, controlled vocabulary) -> category
(canonical bundle). This module is the single source of truth; app/rules/taxonomy.py's
CATEGORY_MAP (keyed by today's existing `subcategory` strings) is derived from it so
there is exactly one place that decides what bundles into what.
"""

from typing import Dict, Optional, Set, Tuple

GARMENT_CLASS_TAXONOMY_VERSION = "garment_class_v1"

# --- SPEC.md Section 9: Fine-Grained Garment Classes ---

TOPS: Set[str] = {
    "T_SHIRT", "SHIRT", "BLOUSE", "POLO", "TANK_TOP", "CROP_TOP", "TUBE_TOP",
    "SWEATER", "SWEATSHIRT", "HOODIE", "CARDIGAN", "VEST", "TOP_OTHER",
}
BOTTOMS: Set[str] = {
    "JEANS", "TROUSERS", "CHINOS", "CARGO_PANTS", "SHORTS", "SKIRT",
    "LEGGINGS", "JOGGERS", "BOTTOM_OTHER",
}
ONE_PIECE: Set[str] = {
    "DRESS", "JUMPSUIT", "ROMPER", "OVERALLS", "ONE_PIECE_OTHER",
}
OUTERWEAR: Set[str] = {
    "BLAZER", "SUIT_JACKET", "JACKET", "COAT", "TRENCH_COAT", "BOMBER",
    "DENIM_JACKET", "OVERSHIRT", "OUTERWEAR_OTHER",
}
FOOTWEAR: Set[str] = {
    "SNEAKERS", "FORMAL_SHOES", "LOAFERS", "BOOTS", "SANDALS", "HEELS",
    "FLIP_FLOPS", "FOOTWEAR_OTHER",
}
ACCESSORIES: Set[str] = {
    "BELT", "HAT", "CAP", "SCARF", "TIE", "BAG", "WATCH", "JEWELLERY",
    "SUNGLASSES", "ACCESSORY_OTHER",
}
TRADITIONAL: Set[str] = {
    "SAREE", "DHOTI", "KURTA", "LEHENGA", "SHERWANI", "SALWAR", "DUPATTA",
    "TRADITIONAL_OTHER",
}
OTHER: Set[str] = {
    "INNERWEAR", "ACTIVEWEAR", "OTHER",
}

GARMENT_CLASSES: Set[str] = TOPS | BOTTOMS | ONE_PIECE | OUTERWEAR | FOOTWEAR | ACCESSORIES | TRADITIONAL | OTHER

# --- SPEC.md Section 10: Category Bundling ---

CLASS_TO_CATEGORY: Dict[str, str] = {
    **{cls: "TOP" for cls in TOPS},
    **{cls: "BOTTOM" for cls in BOTTOMS},
    **{cls: "ONE_PIECE" for cls in ONE_PIECE},
    **{cls: "OUTERWEAR" for cls in OUTERWEAR},
    **{cls: "FOOTWEAR" for cls in FOOTWEAR},
    **{cls: "ACCESSORY" for cls in ACCESSORIES},
    **{cls: "TRADITIONAL" for cls in TRADITIONAL},
    **{cls: "OTHER" for cls in OTHER},
}

# --- Translation from today's existing lowercase `subcategory` vocabulary ---
# (kept as the finer-grained field beneath garment_class; this table lets
# already-ingested/pre-existing subcategory strings resolve to a garment_class
# without discarding any existing data or prompts.)

SUBCATEGORY_TO_CLASS: Dict[str, str] = {
    # Tops
    "oxford_shirt": "SHIRT", "button_down_shirt": "SHIRT", "dress_shirt": "SHIRT",
    "flannel_shirt": "SHIRT", "tshirt": "T_SHIRT", "polo_shirt": "POLO",
    "henley": "TOP_OTHER", "tank_top": "TANK_TOP", "crop_top": "CROP_TOP",
    "blouse": "BLOUSE", "sweater": "SWEATER", "cardigan": "CARDIGAN",
    "hoodie": "HOODIE", "sweatshirt": "SWEATSHIRT", "vest": "VEST",
    # Bottoms
    "jeans": "JEANS", "trousers": "TROUSERS", "chinos": "CHINOS",
    "dress_pants": "TROUSERS", "cargo_pants": "CARGO_PANTS", "shorts": "SHORTS",
    "sweatpants": "JOGGERS", "leggings": "LEGGINGS", "skirt": "SKIRT", "palazzo_pants": "TROUSERS",
    "mini_skirt": "SKIRT", "midi_skirt": "SKIRT", "maxi_skirt": "SKIRT",
    # One Piece
    "dress": "DRESS", "sundress": "DRESS", "maxi_dress": "DRESS",
    "jumpsuit": "JUMPSUIT", "romper": "ROMPER", "overalls": "OVERALLS",
    # Outerwear
    "blazer": "BLAZER", "suit_jacket": "SUIT_JACKET", "coat": "COAT",
    "trench_coat": "TRENCH_COAT", "overcoat": "COAT", "parka": "OUTERWEAR_OTHER",
    "leather_jacket": "JACKET", "denim_jacket": "DENIM_JACKET",
    "bomber_jacket": "BOMBER", "puffer_jacket": "OUTERWEAR_OTHER",
    "windbreaker": "OUTERWEAR_OTHER",
    # Footwear
    "sneakers": "SNEAKERS", "boots": "BOOTS", "loafers": "LOAFERS",
    "oxfords": "FORMAL_SHOES", "derby": "FORMAL_SHOES", "sandals": "SANDALS",
    "heels": "HEELS", "flats": "FOOTWEAR_OTHER",
    # Accessories
    "belt": "BELT", "scarf": "SCARF", "hat": "HAT", "cap": "CAP", "tie": "TIE",
    "gloves": "ACCESSORY_OTHER", "bag": "BAG", "sunglasses": "SUNGLASSES",
    # New spec classes with no prior subcategory equivalent (identity mapping,
    # lowercased garment_class value doubles as the subcategory string)
    "saree": "SAREE", "dhoti": "DHOTI", "kurta": "KURTA", "lehenga": "LEHENGA",
    "sherwani": "SHERWANI", "salwar": "SALWAR", "dupatta": "DUPATTA",
    "joggers": "JOGGERS", "tube_top": "TUBE_TOP", "watch": "WATCH",
    "jewellery": "JEWELLERY", "innerwear": "INNERWEAR", "activewear": "ACTIVEWEAR",
}


def bundle_garment_class(garment_class: str) -> Tuple[Optional[str], str, bool]:
    """
    Looks up the canonical category bundle for a given garment_class.
    Returns: (canonical_category, taxonomy_version, requires_review)
    """
    normalized = (garment_class or "").upper().strip().replace(" ", "_").replace("-", "_")
    category = CLASS_TO_CATEGORY.get(normalized)
    if category is None:
        return None, GARMENT_CLASS_TAXONOMY_VERSION, True
    return category, GARMENT_CLASS_TAXONOMY_VERSION, False


def infer_garment_class_from_subcategory(subcategory: str) -> str:
    """
    Derives a garment_class from an existing (legacy) lowercase subcategory string.
    Never silently drops data (SPEC.md Section 37): unmapped subcategories fall back
    to a generic "<CATEGORY>_OTHER"-style class via the finer-grained taxonomy module,
    or plain "OTHER" if even that can't resolve.
    """
    normalized = (subcategory or "").lower().strip().replace(" ", "_").replace("-", "_")
    mapped = SUBCATEGORY_TO_CLASS.get(normalized)
    if mapped:
        return mapped

    # Fall back through the legacy subcategory->category lookup to pick a sane *_OTHER bucket.
    from app.rules.taxonomy import bundle_category  # local import avoids a circular module load

    legacy_category, _version, _requires_review = bundle_category(normalized)
    fallback_by_category = {
        "TOP": "TOP_OTHER",
        "BOTTOM": "BOTTOM_OTHER",
        "ONE_PIECE": "ONE_PIECE_OTHER",
        "OUTERWEAR": "OUTERWEAR_OTHER",
        "FOOTWEAR": "FOOTWEAR_OTHER",
        "ACCESSORY": "ACCESSORY_OTHER",
    }
    return fallback_by_category.get(legacy_category, "OTHER")
