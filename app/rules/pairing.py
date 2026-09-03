"""Pairing Compatibility (SPEC.md Section 12-13).

Pairing is distinct from Layering ("can these occupy these layer positions?"),
Structural ("can these physically/logically coexist?"), and Visual ("do these
look coherent?"). Pairing answers a narrower question: "can these garments
meaningfully be worn together as part of one outfit at all?" — a deterministic,
category/class-level check, independent of layer stacking or visual harmony.

Treated as a hard-constraint tier alongside Layering/Structural (SPEC.md Section 17's
hierarchy diagram omits Pairing, but Section 12-13 defines it as a 4th distinct
concept; this module's own docstring flags that placement as an implementation
judgment call, not an explicit spec statement).
"""

from typing import Any, Dict, Tuple

PAIRING_RULE_VERSION = "pairing_v1"

# Traditional garments (SAREE, KURTA, SHERWANI, ...) are not automatically
# incompatible with western pieces (fusion outfits are legitimate), but the
# combination is unusual enough to warrant review rather than an automatic pass.
WESTERN_BOTTOM_CATEGORIES = {"BOTTOM", "ONE_PIECE"}


def evaluate_pairing_compatibility(
    garment_a_attrs: Dict[str, Any],
    garment_b_attrs: Dict[str, Any],
) -> Tuple[str, float, str, str]:
    """
    Evaluates whether two garments can meaningfully be worn together at all.
    Returns: (decision: 'COMPATIBLE' | 'INCOMPATIBLE' | 'REVIEW_REQUIRED', score, reason, rule_version)
    """
    cat_a = garment_a_attrs.get("category", "").upper()
    cat_b = garment_b_attrs.get("category", "").upper()
    categories = sorted([cat_a, cat_b])

    # Rule 1: Two standalone one-piece garments can never form one outfit together.
    if categories == ["ONE_PIECE", "ONE_PIECE"]:
        return (
            "INCOMPATIBLE",
            0.05,
            "Two standalone one-piece garments cannot be meaningfully combined into a single outfit.",
            PAIRING_RULE_VERSION,
        )

    # Rule 2: Traditional + western bottom/one-piece is an unusual fusion pairing -> review.
    if "TRADITIONAL" in (cat_a, cat_b):
        other = cat_b if cat_a == "TRADITIONAL" else cat_a
        if other in WESTERN_BOTTOM_CATEGORIES:
            return (
                "REVIEW_REQUIRED",
                0.45,
                "Traditional garment paired with a western bottom/one-piece is an unconventional fusion combination.",
                PAIRING_RULE_VERSION,
            )
        return (
            "COMPATIBLE",
            0.85,
            "Traditional garment paired with accessory/outerwear/footwear is a conventional combination.",
            PAIRING_RULE_VERSION,
        )

    # Rule 3: Everything else is a generically valid pairing candidate; layering/structural
    # rules own the finer-grained "can these occupy the same outfit slot" decision.
    return (
        "COMPATIBLE",
        0.9,
        "No generic pairing conflict identified between these garment categories.",
        PAIRING_RULE_VERSION,
    )
