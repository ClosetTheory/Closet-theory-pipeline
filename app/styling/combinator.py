"""Styling Stage 7 (part 1): Outfit Combination Assembly.

Builds candidate outfit combinations (one garment per required role) from the
role-grouped retrieval candidates, honoring locked-in anchor garments, and caps
the combinatorial explosion before the (relatively expensive) compatibility
evaluation runs on each one.
"""

from itertools import product
from typing import Dict, FrozenSet, List, Optional, Tuple
from app.models.garment import Garment
from app.rules.scoring import _garment_set_similarity
from app.schemas.styling import StylingIntent
from app.styling.retrieval import RoleCandidates, resolve_role

MAX_COMBOS_TO_EVALUATE = 60

# Occasions/formality levels where an added outerwear layer reads as "outfit plus a coat"
# rather than as the actual look — a dressy party/formal/evening request should never have a
# jacket layered on top of it just because the wardrobe has a well-matching one available.
NO_OUTERWEAR_CONTEXTS = {"party", "formal", "evening"}


def _occasion_prefers_no_outerwear(intent: Optional[StylingIntent]) -> bool:
    if intent is None:
        return False
    signals = {(intent.occasion or "").lower(), (intent.formality or "").lower()}
    return bool(signals & NO_OUTERWEAR_CONTEXTS)

# (garments, retrieval_score_sum, base_combo_key) — base_combo_key is set only for a
# "base combo + optional outerwear layer" variant, pointing at the frozenset of garment
# ids of the base combo it layers on top of, so the ranking stage can drop the variant
# unless the layer actually improves the outfit's final score (see ranking.py).
ComboEntry = Tuple[List[Garment], float, Optional[FrozenSet[str]]]

# Roles that form the structural skeleton of an outfit. Anything else an anchor resolves
# to (ACCESSORY, BAG, ...) is an "extra" that, in anchors-only mode, rides along on every
# combination rather than occupying a slot of its own.
BODY_ROLES = ("TOP", "BOTTOM", "ONE_PIECE")
SLOT_ROLES = BODY_ROLES + ("FOOTWEAR", "OUTERWEAR")


def anchors_cover_body(anchors: List[Garment]) -> bool:
    """True when the anchored garments alone can dress a body: a TOP plus a BOTTOM, or a
    ONE_PIECE. Used to fail an anchors-only request up front, with a message the stylist can
    act on, instead of letting it run nine stages and come back with nothing."""
    roles = {resolve_role(a) for a in anchors}
    return ("TOP" in roles and "BOTTOM" in roles) or "ONE_PIECE" in roles


def _role_options(role_candidates: RoleCandidates, role: str) -> List[Tuple[Garment, float]]:
    return role_candidates.get(role, [])


def build_outfit_combinations(
    role_candidates: RoleCandidates,
    anchors: List[Garment],
    intent: Optional[StylingIntent] = None,
    anchors_only: bool = False,
) -> List[ComboEntry]:
    """
    Returns a capped list of (garments, retrieval_score_sum, base_combo_key) candidate outfit
    combinations. Body coverage: TOP+BOTTOM or ONE_PIECE (whichever the wardrobe/anchors
    support). FOOTWEAR: included when available. OUTERWEAR: an additional layered variant per
    base combo, only ever *suggested* by the ranking stage if it improves the outfit's score —
    skipped entirely for party/formal/evening requests, where a layered coat reads as "outfit
    plus a coat" rather than the actual look. ACCESSORY: only included via anchors (not
    auto-added in V1).

    anchors_only=True changes the contract: the anchored garments are the *entire* candidate
    pool. No slot is ever filled from role_candidates, every role the anchors cover becomes a
    required slot (an anchored jacket is worn, not "suggested"), and anchored accessories/bags
    ride along on every combination. A stylist who pins five garments wants outfits made of
    those five and nothing else — confirmed live: with the previous "lock anchors, fill the
    rest from the wardrobe" behaviour, pinning a shirt and trousers still came back with
    unrelated shoes and a coat the stylist never picked.
    """
    anchors_by_role: Dict[str, List[Garment]] = {}
    for anchor in anchors:
        role = resolve_role(anchor)
        if role:
            anchors_by_role.setdefault(role, []).append(anchor)

    def options_for(role: str) -> List[Tuple[Garment, float]]:
        if role in anchors_by_role:
            return [(g, 1.0) for g in anchors_by_role[role]]
        if anchors_only:
            return []
        return _role_options(role_candidates, role)

    if anchors_only:
        return _anchors_only_combinations(anchors_by_role)

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
    outerwear_options = [] if _occasion_prefers_no_outerwear(intent) else options_for("OUTERWEAR")

    raw_combos: List[ComboEntry] = []
    for slot_groups in body_options:
        slot_groups_with_shoes = list(slot_groups)
        if footwear_options:
            slot_groups_with_shoes = slot_groups_with_shoes + [footwear_options]

        for combo in product(*slot_groups_with_shoes):
            # Average, not sum: a ONE_PIECE+FOOTWEAR combo has fewer terms than a
            # TOP+BOTTOM+FOOTWEAR one, so summing systematically ranks it lower regardless of
            # how well each individual item scores — confirmed live: with a real 6-item
            # ONE_PIECE retrieval pool available, zero ONE_PIECE combos survived into the
            # diversity-capped 60 because they could never out-sum a 3-term combo. Averaging
            # makes combos with different slot counts directly comparable.
            scores = [s for _g, s in combo]
            avg_score = sum(scores) / len(scores)
            garments = [g for g, _s in combo]
            base_key = frozenset(g.id for g in garments)
            raw_combos.append((garments, avg_score, None))

            if outerwear_options:
                # Vary the outerwear layer across the top few options (not just the single
                # best match every time) so different base combos end up wearing different
                # jackets/coats instead of the wardrobe's one "best" outer layer everywhere.
                # Tagged with base_key so ranking only suggests the layer if it scores better
                # than the base combo — otherwise it's dropped (see ranking.py).
                top_outers = sorted(outerwear_options, key=lambda pair: pair[1], reverse=True)[:3]
                for outer_garment, outer_score in top_outers:
                    raw_combos.append((garments + [outer_garment], (sum(scores) + outer_score) / (len(scores) + 1), base_key))

    # Dedup by garment-id set (anchors + limited options can otherwise repeat). Base combos are
    # deduped first so a base combo is never dropped in favor of a layered variant sharing the
    # same score-sort position — ranking needs the base present to judge the layer against it.
    seen = set()
    deduped: List[ComboEntry] = []
    for garments, score, base_key in sorted(raw_combos, key=lambda entry: entry[2] is not None):
        key = frozenset(g.id for g in garments)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((garments, score, base_key))

    return _diversity_capped(deduped, MAX_COMBOS_TO_EVALUATE)


def _anchors_only_combinations(anchors_by_role: Dict[str, List[Garment]]) -> List[ComboEntry]:
    """Every combination that can be dressed from the anchors alone: one garment per anchored
    slot role (TOP+BOTTOM and/or ONE_PIECE, plus FOOTWEAR and OUTERWEAR when anchored), with
    every anchored extra (accessories, bags, ...) attached to each. Retrieval score is a flat
    1.0 — these garments were hand-picked, there is nothing to rank them against — so the
    ranking stage decides purely on compatibility, request match and visual harmony."""
    body_options: List[List[List[Tuple[Garment, float]]]] = []
    if anchors_by_role.get("TOP") and anchors_by_role.get("BOTTOM"):
        body_options.append([[(g, 1.0) for g in anchors_by_role["TOP"]], [(g, 1.0) for g in anchors_by_role["BOTTOM"]]])
    if anchors_by_role.get("ONE_PIECE"):
        body_options.append([[(g, 1.0) for g in anchors_by_role["ONE_PIECE"]]])
    if not body_options:
        return []

    required_layers = [
        [(g, 1.0) for g in anchors_by_role[role]]
        for role in ("FOOTWEAR", "OUTERWEAR")
        if anchors_by_role.get(role)
    ]
    extras = [g for role, garments in anchors_by_role.items() if role not in SLOT_ROLES for g in garments]

    seen = set()
    combos: List[ComboEntry] = []
    for slot_groups in body_options:
        for combo in product(*(slot_groups + required_layers)):
            garments = [g for g, _s in combo] + extras
            key = frozenset(g.id for g in garments)
            if key in seen:
                continue
            seen.add(key)
            combos.append((garments, 1.0, None))
    return _diversity_capped(combos, MAX_COMBOS_TO_EVALUATE)


def _diversity_capped(combos: List[ComboEntry], cap: int) -> List[ComboEntry]:
    """
    A flat sort-by-score cap collapses onto whichever single item happens to score highest in
    each role (e.g. one "best" bottom + one "best" pair of shoes dominating almost every
    surviving combo, with only the top/outerwear slot actually varying) — confirmed live: of
    60 combos kept by raw score alone, effectively all of them shared the same bottom AND the
    same footwear. The later ranking-stage diversity penalty (apply_diversity_penalty) can only
    reshuffle among whatever survives this cap, so if real variety never gets this far, no
    amount of re-ranking can recover it. Greedily cap the same way ranking already diversifies
    its final shortlist — same Jaccard-similarity approach, applied one stage earlier — so a
    genuinely varied set of combinations survives into compatibility evaluation and ranking.
    """
    if len(combos) <= cap:
        return sorted(combos, key=lambda entry: entry[1], reverse=True)

    remaining = sorted(combos, key=lambda entry: entry[1], reverse=True)
    selected: List[ComboEntry] = []
    selected_id_sets: List[set] = []

    while remaining and len(selected) < cap:
        if not selected:
            chosen = remaining.pop(0)
        else:
            best_idx, best_adjusted = 0, -1.0
            for idx, (garments, score, _base_key) in enumerate(remaining):
                ids = [g.id for g in garments]
                max_similarity = max(
                    _garment_set_similarity(ids, list(s)) for s in selected_id_sets
                )
                adjusted = score - max_similarity
                if adjusted > best_adjusted:
                    best_idx, best_adjusted = idx, adjusted
            chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_id_sets.append({g.id for g in chosen[0]})

    return selected
