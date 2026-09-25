"""Unit tests for app.rules.member_signals — the translation from a member's stated profile
(production's colour analysis / preferences / weekly plan shapes) into things the styling
pipeline can score and filter on — and for the ranking pieces that consume them."""

from datetime import date

from app.rules.feedback import normalise_reason_tags
from app.rules.member_signals import (
    PALETTE_COUNT,
    STATED_AVOID_COUNT,
    STATED_LOVE_COUNT,
    garment_constraint_violations,
    merge_affinities,
    palette_affinities,
    palette_summary,
    split_checkable_constraints,
    stated_attribute_affinities,
    today_plan,
    warmth_target_from_weather,
)
from app.rules.style_profile import attribute_affinity_score
from app.schemas.styling import StylingContext, StylingIntent, describe_member_profile
from app.styling.ranking import _weather_fit

# Aarushi's authored profile, verbatim from the roster — a warm Soft Autumn.
AUTUMN = {
    "season": "Autumn", "season_sub": "Soft Autumn", "undertone": "olive", "depth": "medium",
    "temperature": "warm", "contrast": "medium",
    "palette": ["#b25a3c", "#c19a6b", "#6b6b3a", "#9c4722", "#2f4f3a", "#efe3c8", "#7a4a2b"],
    "summary": "A warm-toned medium-depth Autumn best flattered by earthy terracottas and deep forest greens.",
}
PREFS = {
    "fits_loved": ["relaxed", "straight"],
    "colour_preferences": {"love": ["terracotta", "olive"], "avoid": ["neon", "florals"]},
    "aesthetic_leanings": ["minimal"],
}


class _G:
    def __init__(self, attrs):
        self.attributes_json = attrs


# --- stated preferences ------------------------------------------------------------------------

def test_stated_preferences_become_low_count_affinities():
    aff = stated_attribute_affinities(PREFS)
    assert aff["colour"]["terracotta"] == {"score": 0.6, "count": STATED_LOVE_COUNT}
    assert aff["colour"]["neon"]["score"] < 0 and aff["colour"]["neon"]["count"] == STATED_AVOID_COUNT
    assert aff["fit"]["relaxed"]["score"] > 0
    # "florals" is a pattern word, not a colour — it lands on the pattern field the profile tracks
    assert aff["pattern"]["floral"]["score"] < 0
    assert "florals" not in aff["colour"]


def test_empty_or_malformed_preferences_are_harmless():
    assert stated_attribute_affinities({}) == {}
    assert stated_attribute_affinities({"colour_preferences": "terracotta"}) == {}
    assert stated_attribute_affinities(None) == {}


# --- palette ------------------------------------------------------------------------------------

def test_warm_autumn_palette_votes_for_earth_hues_and_warm_neutrals_only():
    aff = palette_affinities(AUTUMN)["colour"]
    assert aff["terracotta"]["count"] == PALETTE_COUNT and aff["terracotta"]["score"] > 0
    assert "rust" in aff and "forest green" in aff
    assert "cream" in aff and "camel" in aff          # warm neutrals
    assert "charcoal" not in aff and "navy" not in aff  # cool neutrals stay out
    assert "fuchsia" not in aff and "sky blue" not in aff  # nowhere near the swatches


def test_palette_affinity_moves_the_ranking_component_in_the_right_direction():
    aff = palette_affinities(AUTUMN)
    earth = attribute_affinity_score([{"colour": ["terracotta"]}, {"colour": ["cream"]}], aff)
    off = attribute_affinity_score([{"colour": ["fuchsia"]}, {"colour": ["sky blue"]}], aff)
    assert earth > 0.5  # flattered
    assert off == 0.5   # unknown to the palette: neutral, not punished


def test_palette_summary_prefers_authored_text_then_composes():
    assert palette_summary(AUTUMN) == AUTUMN["summary"]
    composed = palette_summary({"season": "Winter", "temperature": "cool", "depth": "deep", "undertone": "cool"})
    assert composed == "A cool-toned deep-depth Winter with a cool undertone."
    assert palette_summary({}) is None


# --- merge order --------------------------------------------------------------------------------

def test_learned_votes_override_stated_which_override_palette_per_value():
    palette = {"colour": {"terracotta": {"score": 0.4, "count": 1}, "rust": {"score": 0.4, "count": 1}}}
    stated = {"colour": {"terracotta": {"score": 0.6, "count": 2}, "neon": {"score": -0.8, "count": 3}}}
    learned = {"colour": {"terracotta": {"score": -0.5, "count": 9}}}
    merged = merge_affinities(palette, stated, learned)
    assert merged["colour"]["terracotta"] == {"score": -0.5, "count": 9}  # the member's votes win
    assert merged["colour"]["rust"]["score"] == 0.4                       # palette survives where untouched
    assert merged["colour"]["neon"]["score"] == -0.8
    # inputs are not mutated
    assert palette["colour"]["terracotta"]["score"] == 0.4


# --- weekly plan --------------------------------------------------------------------------------

PLAN = {"source": "panel_authored", "days": [
    {"day": "mon", "tags": ["work"], "note": "Office"},
    {"day": "sat", "tags": ["wedding", "festive"], "note": "A sangeet"},
]}


def test_today_plan_picks_the_matching_weekday_and_normalises():
    assert today_plan(PLAN, date(2026, 9, 26)) == {"day": "sat", "tags": ["wedding", "festive"], "note": "A sangeet"}  # a Saturday
    assert today_plan(PLAN, date(2026, 9, 27)) is None  # Sunday: no entry
    assert today_plan({}, date(2026, 9, 28)) is None
    assert today_plan({"days": [{"day": "Monday", "tags": ["WORK"]}]}, date(2026, 9, 28))["tags"] == ["work"]


# --- weather ------------------------------------------------------------------------------------

def test_warmth_target_falls_as_it_gets_hotter_and_rain_nudges_up():
    hot = warmth_target_from_weather({"temp_c": 34, "feels_like_c": 39, "is_rainy": False})
    mild = warmth_target_from_weather({"temp_c": 22, "is_rainy": False})
    cold = warmth_target_from_weather({"temp_c": 4, "is_rainy": False})
    assert hot < mild < cold
    assert hot <= 0.12 and cold >= 0.88
    assert warmth_target_from_weather({"temp_c": 22, "is_rainy": True}) > mild
    assert warmth_target_from_weather({}) is None
    assert warmth_target_from_weather({"temp_c": "n/a"}) is None


def test_weather_fit_uses_the_real_snapshot_over_the_request_word():
    light = [_G({"warmth": 0.1}), _G({"warmth": 0.15})]
    intent = StylingIntent(weather="cold")  # the normaliser guessed wrong
    env = {"warmth_target": 0.1}            # Stage 2 knows it feels like 39°C
    assert _weather_fit(light, intent, env) > 0.9
    assert _weather_fit(light, intent, None) < 0.5   # without the snapshot the word wins
    assert _weather_fit(light, StylingIntent(), {}) == 0.7  # no signal at all: neutral


# --- hard constraints ---------------------------------------------------------------------------

def test_constraints_the_attributes_can_express_are_caught():
    sleeveless_top = {"category": "TOP", "sleeve_length": "sleeveless", "neckline": "crew", "material": "cotton"}
    heels = {"category": "FOOTWEAR", "subcategory": "heels", "material": "leather"}
    wool_coat = {"category": "OUTERWEAR", "material": "wool blend", "sleeve_length": "long"}
    crop = {"category": "TOP", "subcategory": "crop_top", "sleeve_length": "short"}
    jeans = {"category": "BOTTOM", "subcategory": "jeans", "garment_class": "JEANS", "material": "denim"}
    saree = {"category": "ONE_PIECE", "garment_class": "SAREE", "gender": "women"}
    sheer = {"category": "TOP", "transparency": "semi_sheer", "sleeve_length": "long"}

    assert garment_constraint_violations(sleeveless_top, ["no_sleeveless"]) == ["no_sleeveless"]
    assert garment_constraint_violations(sleeveless_top, ["modest_coverage_full_sleeve", "high_neckline_only"]) == ["modest_coverage_full_sleeve"]
    assert garment_constraint_violations(heels, ["no_heels", "no_leather"]) == ["no_heels", "no_leather"]
    assert garment_constraint_violations(wool_coat, ["no_wool", "quick_dry_only"]) == ["no_wool", "quick_dry_only"]
    assert garment_constraint_violations(crop, ["no_bare_midriff"]) == ["no_bare_midriff"]
    assert garment_constraint_violations(jeans, ["no_jeans", "no_trousers"]) == ["no_jeans", "no_trousers"]
    assert garment_constraint_violations(saree, ["no_gendered_ethnic"]) == ["no_gendered_ethnic"]
    assert garment_constraint_violations(sheer, ["no_sheer"]) == ["no_sheer"]


def test_constraints_do_not_fire_on_innocent_garments_or_unknown_codes():
    linen_shirt = {"category": "TOP", "sleeve_length": "long", "neckline": "collared", "material": "linen"}
    assert garment_constraint_violations(linen_shirt, ["no_sleeveless", "no_heels", "quick_dry_only", "needs_pockets"]) == []
    faux = {"category": "FOOTWEAR", "subcategory": "loafers", "material": "faux leather"}
    assert garment_constraint_violations(faux, ["no_leather"]) == []
    assert garment_constraint_violations({}, ["no_sleeveless"]) == []


def test_split_checkable_keeps_prompt_only_constraints_rather_than_dropping_them():
    checkable, prompt_only = split_checkable_constraints(["no_heels", "needs_pockets", "turban_colour_coordination", ""])
    assert checkable == ["no_heels"]
    assert prompt_only == ["needs_pockets", "turban_colour_coordination"]


# --- prompt lines -------------------------------------------------------------------------------

def test_describe_member_profile_is_empty_for_an_ordinary_member_and_full_for_a_character():
    assert describe_member_profile(StylingContext(intent=StylingIntent())) == ""
    ctx = StylingContext(
        intent=StylingIntent(),
        user_preferences={"palette": AUTUMN, "colour_love": ["terracotta"], "colour_avoid": ["neon"], "fits_loved": ["relaxed"]},
        environment={"weather": {"temp_c": 31, "feels_like_c": 36, "condition": "Humid", "humidity_pct": 80, "is_rainy": True},
                     "today": {"day": "mon", "tags": ["work"], "note": "Office"}},
        hard_constraints=["no_sleeveless", "needs_pockets"],
    )
    text = describe_member_profile(ctx)
    assert "Soft Autumn" in text and "warm-toned" in text
    assert "loves terracotta; avoids neon" in text
    assert "HARD constraints" in text and "no sleeveless garments" in text and "needs pockets" in text
    assert "feels like 36°C" in text and "rain likely" in text
    assert "plan for today: work — Office" in text


# --- production reason codes ----------------------------------------------------------------------

def test_production_reason_codes_map_onto_the_review_chips():
    assert normalise_reason_tags(["poor_silhouette", "too_casual", "colour_clash", "nonsense", "too_formal"]) == [
        "proportions", "wrong_occasion", "colour_clash",
    ]
    assert normalise_reason_tags(["not_my_style", "wouldnt_wear"]) == ["wouldnt_wear"]
