"""Maps raw MODA_NER(V) track output (crop / catalog / fullbody) into a partial
GarmentAttributes-shaped dict.

Each track's label vocabulary is its own (Fashionpedia for crop, Shopping100k for
catalog, DeepFashion-MM for fullbody) and only partially overlaps with ClosetTheory's
canonical taxonomy (app/schemas/attributes.py). Mapping is deliberately best-effort:
unmapped/ambiguous values are left out rather than guessed, and `subcategory` is left
for app/schemas/attributes.py's own taxonomy validation (step 5) to catch — an
unmappable subcategory correctly routes the garment to human review rather than
silently guessing (SPEC.md Section 37).

Fields with no reasonable equivalent in the track's raw output (e.g. fullbody has no
category/subcategory at all) are simply omitted; the caller (moda_ner.py) fills gaps
with a VLM top-up call.
"""

from typing import Any, Dict


def _first(value: Any) -> Any:
    """MODA_NER multi-label fields return a list; single-label fields return a scalar."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


# --- Crop track (Fashionpedia vocabulary) ---

_CROP_PATTERN_MAP = {
    "plain": "solid",
    "stripe": "striped",
    "check": "checkered",
    "floral": "floral",
    "cartoon": "graphic",
    "letters numbers": "graphic",
    "dot": "polka_dot",
    "geometric": "geometric",
    "chevron": "geometric",
    "herringbone": "geometric",
    "argyle": "geometric",
    "fair isle": "geometric",
    "abstract": "abstract",
    "toile de jouy": "abstract",
    "paisley": "abstract",
    "peacock": "abstract",
    "plant": "abstract",
    "cheetah": "animal_print",
    "leopard": "animal_print",
    "zebra": "animal_print",
    "giraffe": "animal_print",
    "snakeskin": "animal_print",
    "camouflage": "other",
    "houndstooth": "geometric",
}

_CROP_SLEEVE_LENGTH_MAP = {
    "sleeveless": "sleeveless",
    "short": "short",
    "mini": "short",
    "elbow-length": "three_quarter",
    "three quarter": "three_quarter",
    "wrist-length": "long",
    "above-the-hip": "not_applicable",
}

_CROP_SILHOUETTE_MAP = {
    "a-line": "a_line",
    "asymmetrical": "asymmetrical",
    "baggy": "boxy",
    "oversized": "boxy",
    "loose": "boxy",
    "tent": "boxy",
    "balloon": "draped",
    "circle": "draped",
    "curved": "draped",
    "peplum": "draped",
    "bell": "flared",
    "bell bottom": "flared",
    "flare": "flared",
    "trumpet": "flared",
    "wide leg": "flared",
    "bootcut": "flared",
    "fit and flare": "hourglass",
    "mermaid": "hourglass",
    "high low": "asymmetrical",
    "peg": "tapered",
    "pencil": "fitted",
    "tight": "fitted",
    "regular": "straight",
    "straight": "straight",
    "symmetrical": "straight",
}

# Direct/clear synonyms only — anything else passes through as-is and either matches
# KNOWN_SUBCATEGORIES or correctly fails taxonomy validation (routed to review).
_CROP_SUBCATEGORY_SYNONYMS = {
    "puffer": "puffer_jacket",
    "windbreaker": "windbreaker",
    "trench": "trench_coat",
    "bomber": "bomber_jacket",
    "tank": "tank_top",
    "cargo": "cargo_pants",
    "short": "shorts",
    "sweatpants": "sweatpants",
}


def map_crop_track(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fields: master_category, category, sub_category, silhouette, hemline,
    sleeve_length, sleeve_shape, neckline, collar_presence, collar_style, waist_type,
    material (multi), surface_treatment (multi), pattern, closure_type."""
    out: Dict[str, Any] = {}

    if raw.get("category"):
        out["category"] = str(raw["category"]).replace(" ", "_")
    if raw.get("sub_category"):
        sub = str(raw["sub_category"]).lower().strip().replace(" ", "_").replace("-", "_")
        out["subcategory"] = _CROP_SUBCATEGORY_SYNONYMS.get(sub, sub)

    pattern = _CROP_PATTERN_MAP.get(str(raw.get("pattern", "")).lower())
    if pattern:
        out["pattern"] = pattern

    sleeve = _CROP_SLEEVE_LENGTH_MAP.get(str(raw.get("sleeve_length", "")).lower())
    if sleeve:
        out["sleeve_length"] = sleeve

    silhouette = _CROP_SILHOUETTE_MAP.get(str(raw.get("silhouette", "")).lower())
    if silhouette:
        out["silhouette"] = silhouette

    if raw.get("collar_style"):
        out["collar_detail"] = str(raw["collar_style"])

    return out


# --- Catalog track (Shopping100k vocabulary) ---

_CATALOG_SUBCATEGORY_SYNONYMS = {
    "jean": "jeans",
    "t-shirt": "tshirt",
    "jumper": "sweater",
    "trouser": "trousers",
}

_CATALOG_FIT_MAP = {
    "jeggings": "tight",
    "large": "oversized",
    "loose": "loose",
    "oversize": "oversized",
    "regular": "regular",
    "skinny": "slim",
    "slim": "slim",
    "small": "slim",
    "tailered": "tailored",
    "tapered": "slim",
}

_CATALOG_PATTERN_MAP = {
    "animal": "animal_print",
    "burnout": "textured",
    "camouflage": "other",
    "checked": "checkered",
    "colour gradient": "abstract",
    "colourful": "other",
    "floral": "floral",
    "herringbone": "geometric",
    "marl": "textured",
    "paisley": "abstract",
    "photo": "graphic",
    "pinstriped": "striped",
    "plain": "solid",
    "polka dot": "polka_dot",
    "print": "graphic",
    "striped": "striped",
}

_CATALOG_SLEEVE_LENGTH_MAP = {
    "3/4": "three_quarter",
    "spaghetti": "sleeveless",
    "sleeveless": "sleeveless",
    "elbow": "three_quarter",
    "extra long": "extra_long",
    "extra short": "short",
    "long": "long",
    "short": "short",
    "strapless": "sleeveless",
}


def map_catalog_track(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fields: category, collar, color, fabric, fastening, fit, neckline, pattern,
    pocket, sleeve_length."""
    out: Dict[str, Any] = {}

    if raw.get("category"):
        cat = str(raw["category"]).lower().strip().replace(" ", "_").replace("-", "_")
        out["subcategory"] = _CATALOG_SUBCATEGORY_SYNONYMS.get(cat, cat)
        out["category"] = str(raw["category"])
    if raw.get("color"):
        out["colour"] = [str(raw["color"])]
    if raw.get("fabric"):
        out["material"] = str(raw["fabric"])
    if raw.get("pocket"):
        out["pocket_detail"] = str(raw["pocket"])

    fit = _CATALOG_FIT_MAP.get(str(raw.get("fit", "")).lower())
    if fit:
        out["fit"] = fit

    pattern = _CATALOG_PATTERN_MAP.get(str(raw.get("pattern", "")).lower())
    if pattern:
        out["pattern"] = pattern

    sleeve = _CATALOG_SLEEVE_LENGTH_MAP.get(str(raw.get("sleeve_length", "")).lower())
    if sleeve:
        out["sleeve_length"] = sleeve

    collar = raw.get("collar")
    neckline = raw.get("neckline")
    if collar or neckline:
        out["collar_detail"] = ", ".join(str(v) for v in (collar, neckline) if v)

    return out


# --- Fullbody track (DeepFashion-MM vocabulary) ---
#
# This track has no category/subcategory at all — it describes a whole outfit
# (upper/lower/outer garments together), not one garment. We only pull the
# "upper_*" fields as supplementary hints (the most common single-garment case for
# a full-body shot); category/subcategory/material/fit/silhouette/colour are left
# for the VLM top-up. "NA" is a real class in this track's vocabulary (explicit
# not-applicable), not missing data — treated the same as absent.

_FULLBODY_PATTERN_MAP = {
    "pure color": "solid",
    "solid": "solid",
    "stripe": "striped",
    "lattice": "checkered",
    "floral": "floral",
    "graphic": "graphic",
    "print": "graphic",
    "plaid": "plaid",
}

_FULLBODY_SLEEVE_LENGTH_MAP = {
    "sleeveless": "sleeveless",
    "short-sleeve": "short",
    "medium-sleeve": "three_quarter",
    "long-sleeve": "long",
    "not long-sleeve": "short",
}


def map_fullbody_track(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fields: glasses, hat, lower_clothing_length, lower_fabric, lower_pattern,
    neckline, neckwear, outer_clothing_cardigan, outer_fabric, outer_pattern, ring,
    sleeve_length, socks, upper_clothing_covering_navel, upper_fabric, upper_pattern,
    waist_accessories, wrist_wearing."""
    out: Dict[str, Any] = {}

    upper_fabric = raw.get("upper_fabric")
    if upper_fabric and str(upper_fabric).upper() != "NA":
        out["material"] = str(upper_fabric)

    pattern = _FULLBODY_PATTERN_MAP.get(str(raw.get("upper_pattern", "")).lower())
    if pattern:
        out["pattern"] = pattern

    sleeve = _FULLBODY_SLEEVE_LENGTH_MAP.get(str(raw.get("sleeve_length", "")).lower())
    if sleeve:
        out["sleeve_length"] = sleeve

    return out


TRACK_MAPPERS = {
    "crop": map_crop_track,
    "catalog": map_catalog_track,
    "fullbody": map_fullbody_track,
}
