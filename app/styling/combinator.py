"""Styling Stage 7 (part 1): Outfit Combination Assembly.

Builds candidate outfit combinations (one garment per required role) from the
role-grouped retrieval candidates, honoring locked-in anchor garments, and caps
the combinatorial explosion before the (relatively expensive) compatibility
evaluation runs on each one.
"""

from itertools import product
from typing import Dict, List, Tuple
from app.models.garment import Garment
from app.styling.retrieval import RoleCandidates, resolve_role

MAX_COMBOS_TO_EVALUATE = 60


def _role_options(role_candidates: RoleCandidates, role: str) -> List[Tuple[Garment, float]]:
    return role_candidates.get(role, [])


def build_outfit_combinations(
    role_candidates: RoleCandidates,
    anchors: List[Garment],
) -> List[Tuple[List[Garment], float]]:
    """
    Returns a capped list of (garments, retrieval_score_sum) candidate outfit combinations.
    Body coverage: TOP+BOTTOM or ONE_PIECE (whichever the wardrobe/anchors support).
    FOOTWEAR: included when available. OUTERWEAR: an additional layered variant per base combo.
    ACCESSORY: only included via anchors (not auto-added in V1).
    """
    anchors_by_role: Dict[str, List[Garment]] = {}
    for anchor in anchors:
        role = resolve_role(anchor)
        if role:
            anchors_by_role.setdefault(role, []).append(anchor)

    def options_for(role: str) -> List[Tuple[Garment, float]]:
        if role in anchors_by_role:
            return [(g, 1.0) for g in anchors_by_role[role]]
        return _role_options(role_candidates, role)

    body_options: List[List[Tuple[Garment, float]]] = []
    has_top, has_bottom = bool(options_for("TOP")), bool(options_for("BOTTOM"))
    has_one_piece = bool(options_for("ONE_PIECE"))

    if has_top and has_bottom:
        body_options.append([options_for("TOP"), options_for("BOTTOM")])
    if has_one_piece:
        body_options.append([options_for("ONE_PIECE")])

    if not body_options:
        return []

    footwear_options = options_for("FOOTWEAR")
    outerwear_options = options_for("OUTERWEAR")

    raw_combos: List[Tuple[List[Tuple[Garment, float]], float]] = []
    for slot_groups in body_options:
        slot_groups_with_shoes = list(slot_groups)
        if footwear_options:
            slot_groups_with_shoes = slot_groups_with_shoes + [footwear_options]

        for combo in product(*slot_groups_with_shoes):
            score_sum = sum(s for _g, s in combo)
            garments = [g for g, _s in combo]
            raw_combos.append((garments, score_sum))

            if outerwear_options:
                # Vary the outerwear layer across the top few options (not just the single
                # best match every time) so different base combos end up wearing different
                # jackets/coats instead of the wardrobe's one "best" outer layer everywhere.
                top_outers = sorted(outerwear_options, key=lambda pair: pair[1], reverse=True)[:3]
                for outer_garment, outer_score in top_outers:
                    raw_combos.append((garments + [outer_garment], score_sum + outer_score))

    # Dedup by garment-id set (anchors + limited options can otherwise repeat)
    seen = set()
    deduped: List[Tuple[List[Garment], float]] = []
    for garments, score in raw_combos:
        key = frozenset(g.id for g in garments)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((garments, score))

    deduped.sort(key=lambda pair: pair[1], reverse=True)
    return deduped[:MAX_COMBOS_TO_EVALUATE]
