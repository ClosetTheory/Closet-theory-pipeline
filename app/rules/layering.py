"""Layering Compatibility Decision Tree (Stage 7).

Evaluates whether two garments can occupy compatible layers.
Persists decision, score, reason, and rule version.
"""

from typing import Any, Dict, Tuple
from app.config import settings

LAYERING_RULE_VERSION = "layering_v1"


def evaluate_layering_compatibility(
    garment_a_attrs: Dict[str, Any],
    garment_b_attrs: Dict[str, Any],
) -> Tuple[str, float, str, str]:
    """
    Evaluates layering compatibility between two garment attribute dictionaries.
    Returns: (decision: 'COMPATIBLE' | 'INCOMPATIBLE' | 'REVIEW_REQUIRED', score, reason, rule_version)
    """
    role_a = garment_a_attrs.get("layering_role", "").lower()
    role_b = garment_b_attrs.get("layering_role", "").lower()

    # Normalize roles
    roles = sorted([role_a, role_b])

    cat_a = garment_a_attrs.get("category", "").upper()
    cat_b = garment_b_attrs.get("category", "").upper()

    # Rule 0: Bottom garments occupy lower body and do not conflict with upper torso layering
    if "BOTTOM" in (cat_a, cat_b):
        return (
            "COMPATIBLE",
            1.0,
            "Bottom garment occupies lower body and does not conflict with torso layering stack.",
            LAYERING_RULE_VERSION,
        )

    # Rule 1: Non-torso items (accessory, footwear) don't conflict in torso layering
    if "accessory" in roles or "footwear" in roles:
        return (
            "COMPATIBLE",
            1.0,
            "Accessory or footwear item does not conflict with torso layering stack.",
            LAYERING_RULE_VERSION,
        )

    # Rule 2: Outer + Outer clash
    if roles == ["outer", "outer"]:
        return (
            "INCOMPATIBLE",
            0.15,
            "Multiple heavy outer layers (outer + outer) create bulk and mobility restriction.",
            LAYERING_RULE_VERSION,
        )

    # Rule 3: Mid + Mid clash
    if roles == ["mid", "mid"]:
        return (
            "REVIEW_REQUIRED",
            0.40,
            "Stacking two mid-layers (e.g. sweater + hoodie) can cause bunching unless specifically styled.",
            LAYERING_RULE_VERSION,
        )

    # Rule 4: Base + Mid
    if roles == ["base", "mid"]:
        return (
            "COMPATIBLE",
            0.95,
            "Base layer under mid layer is a canonical, thermally sound layering combination.",
            LAYERING_RULE_VERSION,
        )

    # Rule 5: Mid + Outer
    if roles == ["mid", "outer"]:
        return (
            "COMPATIBLE",
            0.95,
            "Mid layer under outer layer provides optimal insulation and silhouette hierarchy.",
            LAYERING_RULE_VERSION,
        )

    # Rule 6: Base + Outer
    if roles == ["base", "outer"]:
        return (
            "COMPATIBLE",
            0.90,
            "Direct base layer under outer coat/jacket is a standard clean outfit pairing.",
            LAYERING_RULE_VERSION,
        )

    # Rule 7: Base + Base
    if roles == ["base", "base"]:
        cat_a = garment_a_attrs.get("subcategory", "")
        cat_b = garment_b_attrs.get("subcategory", "")
        # Undergarment/tank + shirt can work
        if "tank_top" in (cat_a, cat_b) or "crop_top" in (cat_a, cat_b):
            return (
                "COMPATIBLE",
                0.80,
                "Inner base layer (e.g. tank) under open/buttoned base layer is acceptable.",
                LAYERING_RULE_VERSION,
            )
        return (
            "INCOMPATIBLE",
            0.30,
            "Two full base layers worn simultaneously create seam friction and redundancy.",
            LAYERING_RULE_VERSION,
        )

    # Rule 8: Standalone + Outer
    if "standalone" in roles and "outer" in roles:
        return (
            "COMPATIBLE",
            0.85,
            "Standalone garment (dress/jumpsuit) paired with outerwear is compatible.",
            LAYERING_RULE_VERSION,
        )

    # Rule 9: Standalone + Mid/Base
    if "standalone" in roles:
        return (
            "REVIEW_REQUIRED",
            0.50,
            "Pairing standalone garment with base/mid layer requires silhouette validation.",
            LAYERING_RULE_VERSION,
        )

    # Default fallback
    return (
        "REVIEW_REQUIRED",
        0.50,
        f"Layering pairing ({role_a} + {role_b}) is non-standard and flagged for review.",
        LAYERING_RULE_VERSION,
    )
