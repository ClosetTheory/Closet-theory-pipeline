"""Structural Compatibility Decision Tree (Stage 8).

Evaluates structural compatibility based on outfit slots, coverage, fit, and silhouette harmony.
Persists decision, score, reason, and rule version.
"""

from typing import Any, Dict, Tuple
from app.config import settings

STRUCTURAL_RULE_VERSION = "structural_v1"


def evaluate_structural_compatibility(
    garment_a_attrs: Dict[str, Any],
    garment_b_attrs: Dict[str, Any],
) -> Tuple[str, float, str, str]:
    """
    Evaluates structural compatibility between two garments.
    Returns: (decision: 'COMPATIBLE' | 'INCOMPATIBLE' | 'REVIEW_REQUIRED', score, reason, rule_version)
    """
    cat_a = garment_a_attrs.get("category", "").upper()
    cat_b = garment_b_attrs.get("category", "").upper()
    categories = sorted([cat_a, cat_b])

    role_a = garment_a_attrs.get("layering_role", "").lower()
    role_b = garment_b_attrs.get("layering_role", "").lower()

    fit_a = garment_a_attrs.get("fit", "").lower()
    fit_b = garment_b_attrs.get("fit", "").lower()

    # Rule 1: Duplicate single-occupancy slot conflict (Two bottoms, two shoes)
    if cat_a == "BOTTOM" and cat_b == "BOTTOM":
        return (
            "INCOMPATIBLE",
            0.05,
            "Slot conflict: Two bottom garments occupy the same anatomical region.",
            STRUCTURAL_RULE_VERSION,
        )

    if cat_a == "FOOTWEAR" and cat_b == "FOOTWEAR":
        return (
            "INCOMPATIBLE",
            0.05,
            "Slot conflict: Multiple footwear pairs cannot be worn simultaneously.",
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 2: ONE_PIECE with BOTTOM conflict
    if "ONE_PIECE" in (cat_a, cat_b) and "BOTTOM" in (cat_a, cat_b):
        return (
            "INCOMPATIBLE",
            0.10,
            "Slot clash: A one-piece garment (dress/jumpsuit) overlaps completely with separate bottoms.",
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 3: ONE_PIECE with non-outerwear TOP conflict
    if "ONE_PIECE" in (cat_a, cat_b) and "TOP" in (cat_a, cat_b):
        outer_present = "outer" in (role_a, role_b)
        if not outer_present:
            return (
                "INCOMPATIBLE",
                0.20,
                "Torso coverage conflict: Non-outerwear top worn with full one-piece dress/jumpsuit.",
                STRUCTURAL_RULE_VERSION,
            )
        return (
            "COMPATIBLE",
            0.90,
            "Valid structural complement: Outerwear top worn over one-piece garment.",
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 4: Canonical complementary slots (TOP + BOTTOM)
    if categories == ["BOTTOM", "TOP"]:
        # Check silhouette & fit harmony
        score = 0.95
        reasons = ["Canonical complementary outfit slots: Upper torso (TOP) + Lower torso (BOTTOM)."]

        # Contrast balance bonus/check
        if fit_a == "tight" and fit_b == "tight":
            score = 0.75
            reasons.append("Both items have tight fit; structurally viable but aesthetically restrictive.")
        elif (fit_a == "oversized" and fit_b in ("slim", "regular")) or (fit_b == "oversized" and fit_a in ("slim", "regular")):
            score = 0.98
            reasons.append("Balanced silhouette: Oversized proportion anchored with structured fit.")

        return (
            "COMPATIBLE",
            score,
            " ".join(reasons),
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 5: FOOTWEAR with TOP, BOTTOM, or ONE_PIECE
    if "FOOTWEAR" in (cat_a, cat_b):
        return (
            "COMPATIBLE",
            0.95,
            "Distinct anatomical slots: Footwear structurally complements apparel piece.",
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 6: ACCESSORY with any piece
    if "ACCESSORY" in (cat_a, cat_b):
        return (
            "COMPATIBLE",
            0.95,
            "Accessory occupies peripheral slot without anatomical conflict.",
            STRUCTURAL_RULE_VERSION,
        )

    # Rule 7: TOP + TOP (outerwear layering on top slot)
    if cat_a in ("TOP", "OUTERWEAR") and cat_b in ("TOP", "OUTERWEAR"):
        if role_a != role_b and (role_a == "outer" or role_b == "outer"):
            return (
                "COMPATIBLE",
                0.90,
                "Layered torso structure: Distinct layering tiers allow coexistence in torso slot.",
                STRUCTURAL_RULE_VERSION,
            )
        return (
            "REVIEW_REQUIRED",
            0.45,
            "Torso slot contention: Multiple tops requiring detailed silhouette clearance.",
            STRUCTURAL_RULE_VERSION,
        )

    # Default fallback
    return (
        "REVIEW_REQUIRED",
        0.50,
        f"Structural relationship between {cat_a} and {cat_b} is atypical and flagged for review.",
        STRUCTURAL_RULE_VERSION,
    )
