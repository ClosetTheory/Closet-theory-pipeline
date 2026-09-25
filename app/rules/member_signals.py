"""Deterministic rules that turn a member's *stated* profile into things the styling pipeline can
score and filter on. No I/O, no model calls — everything here is a pure function over dicts, so
it is unit-testable and reusable by whatever loads the profile (app.styling.member_signals).

Why this exists: production's Stage 2 context was `user_preferences = {}` on all 5,736 requests
even though 60 of 72 interaction profiles carried a colour analysis and 13 profiles carried a
full onboarding questionnaire. The signal was in the database and never reached the engine. These
rules are the translation layer, and they are shaped around production's own field names
(`consumer_interaction_profiles.color_analysis`, `consumer_profiles.preferences`,
`consumer_interaction_profiles.weekly_plan`) so the same code serves a real member and an
evaluation character alike.

Three ideas:

1. **Stated affinities.** The learned StyleProfile (app.rules.style_profile) starts empty and
   only moves on votes. A member who told us at onboarding that they love terracotta and never
   wear neon should not have to downvote three neon outfits first. So their stated colour, fit
   and pattern preferences are expressed in the *same* affinity shape ({field: {value: {score,
   count}}}) at a deliberately low count, and merged *under* the learned values — one real vote
   on a value replaces the stated prior for that value.
2. **Palette affinities.** A colour analysis is a list of hex swatches plus a temperature. Every
   named fashion colour the visual rules already know (app.rules.visual.COLOR_HUES) has a hue,
   so a swatch within ~30° of a named hue is a soft "this suits you" on that colour name, and the
   temperature says which neutrals flatter (cream/camel for warm, charcoal/optic white for cool).
   Lowest count of all, so both stated and learned signal override it.
3. **Hard constraints.** A constraint is a string like `no_sleeveless`. The ones an ingested
   garment's attributes can actually express are checked deterministically here; the rest
   (`needs_pockets`, `turban_colour_coordination`) are handed to the LLM stages as text. A
   constraint the code cannot check is still a constraint — it just is not silently dropped.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.rules.visual import COLOR_HUES, NEUTRAL_COLORS, _hue_distance

Affinities = Dict[str, Dict[str, Dict[str, float]]]

# --- stated preferences -> affinities ---------------------------------------------------------

# A stated preference is worth less than a stated *lesson* ("no florals" in a review comment,
# LESSON_COUNT_CREDIT = 2) and far less than accumulated votes, but more than a palette guess.
STATED_LOVE_SCORE, STATED_LOVE_COUNT = 0.6, 2
STATED_AVOID_SCORE, STATED_AVOID_COUNT = -0.8, 3
PALETTE_SCORE, PALETTE_COUNT = 0.4, 1
PALETTE_NEUTRAL_SCORE = 0.3
PALETTE_HUE_TOLERANCE = 30.0  # degrees; tighter than the visual rules' analogous band (40°)

# Words members use in colour_preferences.avoid that are really patterns or finishes, not colours.
_AVOID_WORD_TO_ATTRIBUTE: Dict[str, Tuple[str, str]] = {
    "florals": ("pattern", "floral"), "floral": ("pattern", "floral"),
    "prints": ("pattern", "graphic"), "print": ("pattern", "graphic"), "graphic": ("pattern", "graphic"),
    "animal print": ("pattern", "animal_print"), "polka dots": ("pattern", "polka_dot"),
    "stripes": ("pattern", "striped"), "plaid": ("pattern", "plaid"), "checks": ("pattern", "checkered"),
    "logos": ("pattern", "graphic"),
    "sequins": ("material", "sequin"), "sheer": ("material", "sheer"), "bodycon": ("fit", "tight"),
}

# Neutrals by colour temperature, for palette-derived affinities. A warm palette flatters the
# warm neutrals and vice versa; `olive` sits with warm because every olive-undertone character on
# the roster is an Autumn.
_WARM_NEUTRALS = ("cream", "ivory", "beige", "tan", "khaki", "brown", "olive", "camel")
_COOL_NEUTRALS = ("black", "white", "charcoal", "gray", "grey", "navy")


def _lower_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(v).strip().lower() for v in values if str(v).strip()]


def _put(aff: Affinities, field: str, value: str, score: float, count: int) -> None:
    aff.setdefault(field, {})[value] = {"score": float(score), "count": int(count)}


def stated_attribute_affinities(preferences: Dict[str, Any]) -> Affinities:
    """Production's `consumer_profiles.preferences` -> affinity dict. Reads `colour_preferences
    {love, avoid}` and `fits_loved`; recognises a few avoid-words that are patterns or finishes."""
    aff: Affinities = {}
    if not isinstance(preferences, dict):
        return aff
    colours = preferences.get("colour_preferences") or {}
    for colour in _lower_list(colours.get("love") if isinstance(colours, dict) else None):
        _put(aff, "colour", colour, STATED_LOVE_SCORE, STATED_LOVE_COUNT)
    for word in _lower_list(colours.get("avoid") if isinstance(colours, dict) else None):
        mapped = _AVOID_WORD_TO_ATTRIBUTE.get(word)
        if mapped:
            _put(aff, mapped[0], mapped[1], STATED_AVOID_SCORE, STATED_AVOID_COUNT)
        else:
            _put(aff, "colour", word, STATED_AVOID_SCORE, STATED_AVOID_COUNT)
    for fit in _lower_list(preferences.get("fits_loved")):
        _put(aff, "fit", fit, STATED_LOVE_SCORE, STATED_LOVE_COUNT)
    return aff


# --- colour analysis -> palette affinities ------------------------------------------------------

def _hex_to_hue(hex_colour: str) -> Optional[Tuple[float, float, float]]:
    """#rrggbb -> (hue_degrees, saturation, value) or None if unparseable."""
    s = str(hex_colour or "").strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta == 0:
        hue = 0.0
    elif mx == r:
        hue = (60 * ((g - b) / delta)) % 360
    elif mx == g:
        hue = 60 * ((b - r) / delta) + 120
    else:
        hue = 60 * ((r - g) / delta) + 240
    sat = 0.0 if mx == 0 else delta / mx
    return hue, sat, mx


def palette_affinities(color_analysis: Dict[str, Any]) -> Affinities:
    """Colour analysis -> soft colour affinities on the named colours the visual rules know.

    Only chromatic swatches (saturation > 0.15) vote for named hues — a near-grey swatch in a
    palette carries no hue information. Temperature picks the flattering neutrals."""
    aff: Affinities = {}
    if not isinstance(color_analysis, dict):
        return aff
    hues: List[float] = []
    for swatch in color_analysis.get("palette") or []:
        parsed = _hex_to_hue(swatch)
        if parsed and parsed[1] > 0.15:
            hues.append(parsed[0])
    for name, hue in COLOR_HUES.items():
        if any(_hue_distance(hue, h) <= PALETTE_HUE_TOLERANCE for h in hues):
            _put(aff, "colour", name, PALETTE_SCORE, PALETTE_COUNT)
    temperature = str(color_analysis.get("temperature") or "").lower()
    if temperature == "warm":
        for n in _WARM_NEUTRALS:
            _put(aff, "colour", n, PALETTE_NEUTRAL_SCORE, PALETTE_COUNT)
    elif temperature == "cool":
        for n in _COOL_NEUTRALS:
            _put(aff, "colour", n, PALETTE_NEUTRAL_SCORE, PALETTE_COUNT)
    return aff


def merge_affinities(*layers: Affinities) -> Affinities:
    """Later layers win per (field, value). Call as merge_affinities(palette, stated, learned) so
    a single real vote replaces the prior for that value while untouched values keep it."""
    merged: Affinities = {}
    for layer in layers:
        for field, values in (layer or {}).items():
            for value, stats in values.items():
                merged.setdefault(field, {})[value] = dict(stats)
    return merged


def palette_summary(color_analysis: Dict[str, Any]) -> Optional[str]:
    """One sentence for prompts and the context paragraph, from production's colour analysis
    shape. Prefers the authored summary; otherwise composes one from the structured fields."""
    if not isinstance(color_analysis, dict) or not color_analysis:
        return None
    summary = str(color_analysis.get("summary") or "").strip()
    if summary:
        return summary
    season = color_analysis.get("season_sub") or color_analysis.get("season")
    parts = [p for p in (
        f"{color_analysis['temperature']}-toned" if color_analysis.get("temperature") else None,
        f"{color_analysis['depth']}-depth" if color_analysis.get("depth") else None,
        str(season) if season else None,
    ) if p]
    if not parts:
        return None
    undertone = color_analysis.get("undertone")
    return f"A {' '.join(parts)}" + (f" with a {undertone} undertone" if undertone else "") + "."


# --- weekly plan -------------------------------------------------------------------------------

_DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def today_plan(weekly_plan: Dict[str, Any], today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Production's `weekly_plan` shape ({days: [{day, tags, note}], source}) -> today's entry,
    normalised to {day, tags, note}. None when there is no entry for today."""
    if not isinstance(weekly_plan, dict):
        return None
    key = _DAY_KEYS[(today or date.today()).weekday()]
    for entry in weekly_plan.get("days") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("day") or "").lower()[:3] == key:
            return {
                "day": key,
                "tags": _lower_list(entry.get("tags")),
                "note": (str(entry.get("note") or "").strip() or None),
            }
    return None


# --- weather -> warmth target ------------------------------------------------------------------

# Feels-like temperature (°C) -> the garment `warmth` (0 lightest .. 1 warmest) an outfit should
# average. Piecewise-linear between these anchors; the request-word table in app.styling.ranking
# (hot 0.1 .. freezing 0.95) is the coarse version of the same curve.
_WARMTH_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (36.0, 0.08), (32.0, 0.12), (28.0, 0.20), (24.0, 0.32), (20.0, 0.45),
    (15.0, 0.60), (10.0, 0.75), (5.0, 0.88), (0.0, 0.95),
)


def warmth_target_from_weather(weather: Dict[str, Any]) -> Optional[float]:
    """Maps a WeatherSnapshot dict to a target average warmth. Uses feels-like when present
    (humidity is the whole story in coastal India), and nudges up a little for rain."""
    if not isinstance(weather, dict):
        return None
    temp = weather.get("feels_like_c")
    if temp is None:
        temp = weather.get("temp_c")
    if temp is None:
        return None
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return None
    anchors = _WARMTH_ANCHORS
    if t >= anchors[0][0]:
        target = anchors[0][1]
    elif t <= anchors[-1][0]:
        target = anchors[-1][1]
    else:
        target = anchors[-1][1]
        for (t_hi, w_hi), (t_lo, w_lo) in zip(anchors, anchors[1:]):
            if t_lo <= t <= t_hi:
                frac = (t - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
                target = w_lo + frac * (w_hi - w_lo)
                break
    if weather.get("is_rainy"):
        target = min(1.0, target + 0.05)
    return round(target, 3)


# --- hard constraints --------------------------------------------------------------------------

# Human wording for prompts and trace cards. Anything not listed is shown with underscores
# replaced, so an unknown constraint still reads sensibly.
CONSTRAINT_LABELS: Dict[str, str] = {
    "no_sleeveless": "no sleeveless garments",
    "modest_coverage_full_sleeve": "full-sleeve modest coverage on every top",
    "covered_shoulders_temple": "shoulders always covered",
    "high_neckline_only": "high necklines only — nothing low, strapless or off-shoulder",
    "no_bare_midriff": "no bare midriff or crop tops",
    "no_above_knee": "nothing above the knee",
    "no_heels": "no heels",
    "no_leather": "no leather",
    "no_wool": "no wool",
    "quick_dry_only": "only fabrics that dry overnight — no wool, velvet, leather, suede or corduroy",
    "no_headwear": "no hats or caps (the turban is the only headwear)",
    "turban_colour_coordination": "the turban is the largest block of colour in every outfit and must coordinate",
    "no_gendered_ethnic": "no gender-coded ethnic garments (no saree, lehenga, sherwani, anarkali)",
    "needs_pockets": "needs pockets",
    "no_jeans": "never wears jeans",
    "no_trousers": "never wears trousers — bottoms are salwar, palazzo or saree only",
    "no_shorts": "no shorts outside the house",
    "no_sheer": "nothing sheer or semi-sheer",
    "no_bodycon": "nothing tight or bodycon",
}

_LOW_NECKLINES = {"v", "scoop", "sweetheart", "halter", "off_shoulder", "strapless", "keyhole", "cowl"}
_SLOW_DRY_MATERIALS = ("wool", "velvet", "leather", "suede", "corduroy")
_GENDERED_ETHNIC_CLASSES = {"SAREE", "LEHENGA", "SHERWANI", "ANARKALI", "BLOUSE", "SALWAR", "DUPATTA"}
_TROUSER_WORDS = ("jeans", "trouser", "pant", "chino", "cargo", "legging", "jogger")
_TOP_LIKE_CATEGORIES = {"top", "one_piece", "onepiece", "dress", "outerwear", "tops"}


def constraint_label(constraint: str) -> str:
    return CONSTRAINT_LABELS.get(constraint, constraint.replace("_", " "))


def _text(attrs: Dict[str, Any], *keys: str) -> str:
    return " ".join(str(attrs.get(k) or "") for k in keys).lower()


def _is_top_like(attrs: Dict[str, Any]) -> bool:
    category = str(attrs.get("category") or "").lower()
    if category in _TOP_LIKE_CATEGORIES:
        return True
    role = str(attrs.get("layering_role") or "").lower()
    return role in ("base", "mid", "outer") and category not in ("bottom", "bottoms", "footwear", "accessory")


def garment_constraint_violations(attrs: Dict[str, Any], hard_constraints: Iterable[str]) -> List[str]:
    """Which of the member's hard constraints this garment's attributes plainly violate. Only
    constraints the attribute schema can express are checked; the rest never match here."""
    attrs = attrs or {}
    violations: List[str] = []
    sleeve = str(attrs.get("sleeve_length") or "").lower()
    neckline = str(attrs.get("neckline") or "").lower()
    material = _text(attrs, "material", "primary_material", "fabric_class")
    subcat = _text(attrs, "subcategory", "garment_class")
    fit = str(attrs.get("fit") or "").lower()
    transparency = str(attrs.get("transparency") or "").lower()
    garment_class = str(attrs.get("garment_class") or "").upper()
    top_like = _is_top_like(attrs)

    for c in hard_constraints:
        hit = False
        if c in ("no_sleeveless", "covered_shoulders_temple"):
            hit = sleeve == "sleeveless" or neckline in ("strapless", "halter", "off_shoulder")
        elif c == "modest_coverage_full_sleeve":
            hit = top_like and (sleeve in ("sleeveless", "short") or neckline in ("strapless", "halter", "off_shoulder"))
        elif c == "high_neckline_only":
            hit = neckline in _LOW_NECKLINES
        elif c == "no_bare_midriff":
            hit = "crop" in subcat or "bralette" in subcat or "tube" in subcat
        elif c == "no_above_knee":
            hit = "shorts" in subcat or "mini" in subcat or "hot_pant" in subcat
        elif c == "no_heels":
            hit = "heel" in subcat or "stiletto" in subcat or "pump" in subcat
        elif c == "no_leather":
            hit = "leather" in material and not any(w in material for w in ("faux", "vegan", "pu "))
        elif c == "no_wool":
            hit = "wool" in material or "cashmere" in material
        elif c == "quick_dry_only":
            hit = any(w in material for w in _SLOW_DRY_MATERIALS)
        elif c == "no_headwear":
            hit = garment_class in ("HAT", "CAP", "BEANIE") or "hat" in subcat.split() or "cap" in subcat.split()
        elif c == "no_gendered_ethnic":
            hit = garment_class in _GENDERED_ETHNIC_CLASSES
        elif c == "no_jeans":
            hit = garment_class == "JEANS" or "jean" in subcat
        elif c == "no_trousers":
            hit = any(w in subcat for w in _TROUSER_WORDS) and "palazzo" not in subcat and "salwar" not in subcat
        elif c == "no_shorts":
            hit = "shorts" in subcat
        elif c == "no_sheer":
            hit = transparency in ("sheer", "semi_sheer") or "sheer" in material
        elif c == "no_bodycon":
            hit = fit == "tight" or "bodycon" in subcat
        if hit:
            violations.append(c)
    return violations


def split_checkable_constraints(hard_constraints: Iterable[str]) -> Tuple[List[str], List[str]]:
    """(checked deterministically here, handed to the LLM stages only)."""
    checkable = {
        "no_sleeveless", "covered_shoulders_temple", "modest_coverage_full_sleeve", "high_neckline_only",
        "no_bare_midriff", "no_above_knee", "no_heels", "no_leather", "no_wool", "quick_dry_only",
        "no_headwear", "no_gendered_ethnic", "no_jeans", "no_trousers", "no_shorts", "no_sheer", "no_bodycon",
    }
    constraints = [str(c) for c in hard_constraints if c]
    return [c for c in constraints if c in checkable], [c for c in constraints if c not in checkable]
