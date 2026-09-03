"""Deterministic rules package exports."""

from app.rules.taxonomy import bundle_category, CATEGORY_MAP, TAXONOMY_VERSION
from app.rules.layering import evaluate_layering_compatibility, LAYERING_RULE_VERSION
from app.rules.structural import evaluate_structural_compatibility, STRUCTURAL_RULE_VERSION
from app.rules.visual import evaluate_visual_rules, VISUAL_RULE_VERSION

__all__ = [
    "bundle_category",
    "CATEGORY_MAP",
    "TAXONOMY_VERSION",
    "evaluate_layering_compatibility",
    "LAYERING_RULE_VERSION",
    "evaluate_structural_compatibility",
    "STRUCTURAL_RULE_VERSION",
    "evaluate_visual_rules",
    "VISUAL_RULE_VERSION",
]
