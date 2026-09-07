"""Garment attributes schema and 7-step validation pipeline."""

from enum import Enum
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class PatternEnum(str, Enum):
    SOLID = "solid"
    STRIPED = "striped"
    PLAID = "plaid"
    CHECKERED = "checkered"
    FLORAL = "floral"
    GRAPHIC = "graphic"
    POLKA_DOT = "polka_dot"
    GEOMETRIC = "geometric"
    ABSTRACT = "abstract"
    ANIMAL_PRINT = "animal_print"
    TEXTURED = "textured"
    OTHER = "other"


class FitEnum(str, Enum):
    SLIM = "slim"
    REGULAR = "regular"
    OVERSIZED = "oversized"
    RELAXED = "relaxed"
    TAILORED = "tailored"
    LOOSE = "loose"
    TIGHT = "tight"
    NOT_APPLICABLE = "not_applicable"


class SilhouetteEnum(str, Enum):
    STRAIGHT = "straight"
    A_LINE = "a_line"
    FITTED = "fitted"
    BOXY = "boxy"
    HOURGLASS = "hourglass"
    TAPERED = "tapered"
    FLARED = "flared"
    ASYMMETRICAL = "asymmetrical"
    DRAPED = "draped"
    NOT_APPLICABLE = "not_applicable"


class SleeveLengthEnum(str, Enum):
    SLEEVELESS = "sleeveless"
    SHORT = "short"
    THREE_QUARTER = "three_quarter"
    LONG = "long"
    EXTRA_LONG = "extra_long"
    NOT_APPLICABLE = "not_applicable"


class OccasionEnum(str, Enum):
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BUSINESS_CASUAL = "business_casual"
    FORMAL = "formal"
    WORK = "work"
    LOUNGE = "lounge"
    ACTIVEWEAR = "activewear"
    EVENING = "evening"
    PARTY = "party"
    CEREMONIAL = "ceremonial"


class SeasonEnum(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class LayeringRoleEnum(str, Enum):
    BASE = "base"
    MID = "mid"
    OUTER = "outer"
    STANDALONE = "standalone"
    ACCESSORY = "accessory"
    FOOTWEAR = "footwear"


class GenderEnum(str, Enum):
    WOMEN = "women"
    MEN = "men"
    UNISEX = "unisex"


# --- Enriched attribute enums (ClosetTheoryPilotBeta v6 schema, user-supplied 2026-09-06) ---
# Trimmed subset of the full 144-attribute schema: the ~28 fields judged highest-value for
# this catalogue (ethnic embellishment/construction/fabric detail) vs. cost, tested live
# against the full schema and the current 16-field prompt before adoption (8-garment
# comparison: full 144-field schema cost 8.6x more / took 5.6x longer per garment for only
# 5-8 distinct confidence values out of 144 fields — not worth it wholesale). All new fields
# are optional and default to None when the model has nothing to report, rather than forcing
# an invented answer.
class ColourTemperatureEnum(str, Enum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


class ColourSaturationEnum(str, Enum):
    MUTED = "muted"
    MID = "mid"
    VIVID = "vivid"
    FLUORESCENT = "fluorescent"


class PatternMotifEnum(str, Enum):
    GEOMETRIC = "geometric"
    FLORAL = "floral"
    PAISLEY = "paisley"
    ABSTRACT = "abstract"
    ANIMAL = "animal"
    BOTANICAL = "botanical"
    DIGITAL = "digital"
    PORTRAIT = "portrait"
    TYPOGRAPHIC = "typographic"
    CULTURAL_TRADITIONAL = "cultural_traditional"
    NONE = "none"


class PatternDensityEnum(str, Enum):
    SPARSE = "sparse"
    BALANCED = "balanced"
    DENSE = "dense"
    ALL_OVER = "all_over"
    NONE = "none"


class EmbellishmentEnum(str, Enum):
    SEQUIN = "sequin"
    BEAD = "bead"
    MIRROR = "mirror"
    ZARDOZI = "zardozi"
    KANTHA = "kantha"
    MUKAISH = "mukaish"
    CHIKANKARI = "chikankari"
    CUTWORK = "cutwork"
    APPLIQUE = "applique"
    NONE = "none"


class EmbellishmentDensityEnum(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class EmbellishmentPlacementEnum(str, Enum):
    ALLOVER = "allover"
    PANEL = "panel"
    HEM = "hem"
    YOKE = "yoke"
    SHOULDER = "shoulder"
    COLLAR = "collar"
    CUFF = "cuff"
    BORDER = "border"
    NONE = "none"


class FabricClassEnum(str, Enum):
    WOVEN = "woven"
    KNIT = "knit"
    NON_WOVEN = "non_woven"
    LEATHER = "leather"
    SYNTHETIC_FILM = "synthetic_film"


class WeaveTypeEnum(str, Enum):
    PLAIN = "plain"
    TWILL = "twill"
    SATIN = "satin"
    POPLIN = "poplin"
    OXFORD = "oxford"
    PIQUE = "pique"
    DOBBY = "dobby"
    JACQUARD = "jacquard"
    CHIFFON = "chiffon"
    GEORGETTE = "georgette"
    CREPE = "crepe"
    ORGANZA = "organza"
    CHAMBRAY = "chambray"
    CANVAS = "canvas"
    CORDUROY = "corduroy"
    VELVET = "velvet"
    SHERPA = "sherpa"
    NONE = "none"


class TransparencyEnum(str, Enum):
    OPAQUE = "opaque"
    SEMI_SHEER = "semi_sheer"
    SHEER = "sheer"


class SheenEnum(str, Enum):
    MATTE = "matte"
    SATIN = "satin"
    SHINY = "shiny"
    METALLIC = "metallic"
    IRIDESCENT = "iridescent"


class DrapeEnum(str, Enum):
    STIFF = "stiff"
    STRUCTURED = "structured"
    SOFTLY_FALLING = "softly_falling"
    FLUID = "fluid"
    LIQUID = "liquid"


class StretchGradeEnum(str, Enum):
    RIGID = "rigid"
    SLIGHT = "slight"
    MODERATE = "moderate"
    HIGH = "high"
    FOUR_WAY = "four_way"


class NecklineEnum(str, Enum):
    CREW = "crew"
    V = "v"
    SCOOP = "scoop"
    BOAT = "boat"
    SQUARE = "square"
    SWEETHEART = "sweetheart"
    HALTER = "halter"
    OFF_SHOULDER = "off_shoulder"
    COWL = "cowl"
    HIGH = "high"
    MOCK = "mock"
    TURTLE = "turtle"
    COLLARED = "collared"
    KEYHOLE = "keyhole"
    STRAPLESS = "strapless"


class CollarTypeEnum(str, Enum):
    POINT = "point"
    SPREAD = "spread"
    BUTTON_DOWN = "button_down"
    BAND = "band"
    MANDARIN = "mandarin"
    CAMP = "camp"
    PETER_PAN = "peter_pan"
    SHAWL = "shawl"
    NOTCH = "notch"
    PEAK = "peak"
    FUNNEL = "funnel"
    NO_COLLAR = "no_collar"


class SleeveTypeEnum(str, Enum):
    SET_IN = "set_in"
    RAGLAN = "raglan"
    DOLMAN = "dolman"
    KIMONO = "kimono"
    PUFF = "puff"
    BISHOP = "bishop"
    BELL = "bell"
    LEG_OF_MUTTON = "leg_of_mutton"
    BRACELET = "bracelet"
    FLUTTER = "flutter"


class ClosureTypeEnum(str, Enum):
    BUTTON = "button"
    ZIP = "zip"
    PULLOVER = "pullover"
    WRAP = "wrap"
    LACE_UP = "lace_up"
    HOOK_AND_EYE = "hook_and_eye"
    DRAWSTRING = "drawstring"
    ELASTIC = "elastic"
    BELTED = "belted"
    SNAP = "snap"
    NONE = "none"


class HemStyleEnum(str, Enum):
    CLEAN = "clean"
    ROLLED = "rolled"
    RAW = "raw"
    CUFFED = "cuffed"
    SCALLOPED = "scalloped"
    CURVED = "curved"
    ASYMMETRIC = "asymmetric"


class RiseEnum(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class LegShapeEnum(str, Enum):
    SKINNY = "skinny"
    SLIM = "slim"
    STRAIGHT = "straight"
    TAPERED = "tapered"
    WIDE = "wide"
    FLARE = "flare"
    BOOTCUT = "bootcut"
    BELL = "bell"
    PALAZZO = "palazzo"
    CARGO = "cargo"
    PAPERBAG = "paperbag"
    JOGGER = "jogger"


class WaistbandStyleEnum(str, Enum):
    FITTED = "fitted"
    ELASTIC = "elastic"
    DRAWSTRING = "drawstring"
    PAPERBAG = "paperbag"
    FOLD_OVER = "fold_over"
    NONE = "none"


class ShoeSilhouetteEnum(str, Enum):
    SNEAKER = "sneaker"
    LOAFER = "loafer"
    OXFORD = "oxford"
    DERBY = "derby"
    MULE = "mule"
    HEEL = "heel"
    BOOT = "boot"
    SANDAL = "sandal"
    JUTI = "juti"
    MOJARI = "mojari"
    KOLHAPURI = "kolhapuri"
    CHAPPAL = "chappal"
    ESPADRILLE = "espadrille"
    SLIPPER = "slipper"


class AccessoryKindEnum(str, Enum):
    BAG = "bag"
    BELT = "belt"
    SCARF = "scarf"
    HAT = "hat"
    SUNGLASSES = "sunglasses"
    JEWELLERY = "jewellery"
    WATCH = "watch"
    HAIR = "hair"
    TIE = "tie"
    POCKET_SQUARE = "pocket_square"
    DUPATTA = "dupatta"
    STOLE = "stole"
    BROOCH = "brooch"


class FormalityEnum(str, Enum):
    LOUNGEWEAR = "loungewear"
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BUSINESS = "business"
    FORMAL = "formal"
    CEREMONIAL = "ceremonial"


class MoodIntensityEnum(str, Enum):
    LOW_KEY = "low_key"
    BALANCED = "balanced"
    STATEMENT = "statement"


class WashStateVisibleEnum(str, Enum):
    RAW = "raw"
    MID_WASH = "mid_wash"
    HEAVILY_WASHED = "heavily_washed"
    FADED = "faded"
    NONE = "none"


# Fields where "not_applicable" / an empty answer must become None rather than a hard schema
# error — a vision model asked ~28 extra questions per garment will very reasonably answer
# "not_applicable" (in prose) for a shoe-only field on a shirt, even though the field is
# already Optional and the prompt asks for null in that case.
_OPTIONAL_ENUM_NONE_TOKENS = {"not_applicable", "n/a", "na", "null", "none_", "", "unknown"}

# Field name -> its Enum class, used to soft-fail an invalid value to None instead of
# raising — a model asked ~28 extra questions per garment will occasionally put a value from
# a NEIGHBORING field into the wrong one (e.g. "off_shoulder" — a valid neckline — answered
# under sleeve_type instead), and since every one of these fields is optional, a single
# misplaced value should silently drop out rather than abort the whole extraction and fall
# back to generic mock data (confirmed live: this exact off_shoulder/sleeve_type mixup did
# exactly that before this fix).
_OPTIONAL_ENUM_FIELD_TYPES: Dict[str, type] = {
    "colour_temperature": ColourTemperatureEnum,
    "colour_saturation": ColourSaturationEnum,
    "pattern_motif": PatternMotifEnum,
    "pattern_density": PatternDensityEnum,
    "embellishment": EmbellishmentEnum,
    "embellishment_density": EmbellishmentDensityEnum,
    "embellishment_placement": EmbellishmentPlacementEnum,
    "fabric_class": FabricClassEnum,
    "weave_type": WeaveTypeEnum,
    "transparency": TransparencyEnum,
    "sheen": SheenEnum,
    "drape": DrapeEnum,
    "stretch_grade": StretchGradeEnum,
    "neckline": NecklineEnum,
    "collar_type": CollarTypeEnum,
    "sleeve_type": SleeveTypeEnum,
    "closure_type": ClosureTypeEnum,
    "hem_style": HemStyleEnum,
    "rise": RiseEnum,
    "leg_shape": LegShapeEnum,
    "waistband_style": WaistbandStyleEnum,
    "shoe_silhouette": ShoeSilhouetteEnum,
    "accessory_kind": AccessoryKindEnum,
    "formality": FormalityEnum,
    "mood_intensity": MoodIntensityEnum,
    "wash_state_visible": WashStateVisibleEnum,
}


# Canonical taxonomy subcategory dictionary for taxonomy validation
KNOWN_SUBCATEGORIES = {
    # Tops
    "oxford_shirt", "button_down_shirt", "dress_shirt", "flannel_shirt",
    "tshirt", "polo_shirt", "henley", "tank_top", "crop_top",
    "blouse", "sweater", "cardigan", "hoodie", "sweatshirt", "vest",
    # Bottoms
    "jeans", "trousers", "chinos", "dress_pants", "cargo_pants",
    "shorts", "sweatpants", "leggings", "skirt", "mini_skirt", "midi_skirt", "maxi_skirt",
    # One Piece
    "dress", "sundress", "maxi_dress", "jumpsuit", "romper", "overalls",
    # Outerwear
    "blazer", "suit_jacket", "coat", "trench_coat", "overcoat", "parka",
    "leather_jacket", "denim_jacket", "bomber_jacket", "puffer_jacket", "windbreaker",
    # Footwear
    "sneakers", "boots", "loafers", "oxfords", "derby", "sandals", "heels", "flats",
    # Accessories
    "belt", "scarf", "hat", "cap", "tie", "gloves", "bag", "sunglasses",
    # Traditional (South Asian) — SPEC.md TRADITIONAL class already maps these in
    # app/rules/garment_class.py's SUBCATEGORY_TO_CLASS; they were missing here, which forced
    # the model to downgrade a correctly-recognized "kurta" into the closest Western word
    # ("blouse") once the extraction prompt started constraining output to this exact set.
    "saree", "dhoti", "kurta", "lehenga", "sherwani", "salwar", "dupatta", "palazzo_pants",
    "anarkali", "kimono",
}

# Common alternate spellings a vision model reasonably produces for a compound-word
# subcategory (e.g. "t_shirt" is a completely natural way to write "tshirt") — normalized to
# the canonical taxonomy value before the strict taxonomy check, rather than routing a
# correctly-identified garment to REVIEW_REQUIRED purely over spelling.
SUBCATEGORY_ALIASES = {
    "t_shirt": "tshirt",
    "tee_shirt": "tshirt",
    "tee": "tshirt",
    "polo": "polo_shirt",
    "tank": "tank_top",
    "croptop": "crop_top",
    "denim_jacket_": "denim_jacket",
    "sunglass": "sunglasses",
    "shades": "sunglasses",
    "handbag": "bag",
    "purse": "bag",
    "tote": "bag",
    "tote_bag": "bag",
}

# Common cases where a model answers the "fit" question with a silhouette word instead —
# "fitted" describes how a garment follows the body's shape (a silhouette concept), not one of
# FitEnum's actual values, but models conflate the two constantly.
FIT_ALIASES = {
    "fitted": "tailored",
    "form_fitting": "slim",
    "body_fitting": "slim",
    "snug": "slim",
    "baggy": "loose",
}


class GarmentAttributes(BaseModel):
    """Canonical, strongly validated garment attributes."""

    category: str = Field(..., description="High-level category or raw category string")
    subcategory: str = Field(..., description="Normalized fine-grained garment type")
    garment_class: Optional[str] = Field(
        default=None,
        description="Canonical SPEC.md garment class (controlled vocabulary, e.g. T_SHIRT, JEANS, SAREE)",
    )
    colour: List[str] = Field(..., min_length=1, description="One or more primary/secondary colors")
    pattern: PatternEnum = Field(..., description="Fabric pattern")
    material: str = Field(..., min_length=1, description="Primary fabric material")
    fit: FitEnum = Field(..., description="Fit characteristics")
    silhouette: SilhouetteEnum = Field(..., description="Body silhouette")
    sleeve_length: SleeveLengthEnum = Field(
        default=SleeveLengthEnum.NOT_APPLICABLE,
        description="Length of garment sleeves",
    )
    occasion: List[OccasionEnum] = Field(..., min_length=1, description="Applicable dress codes")
    season: List[SeasonEnum] = Field(..., min_length=1, description="Suitable seasons")
    layering_role: LayeringRoleEnum = Field(..., description="Layering tier")
    gender: GenderEnum = Field(
        default=GenderEnum.UNISEX,
        description="Gendered styling association — women/men/unisex, by cut/styling not assumption",
    )
    warmth: float = Field(..., ge=0.0, le=1.0, description="Normalized thermal rating 0.0-1.0")
    versatility: float = Field(..., ge=0.0, le=1.0, description="Mix-and-match versatility rating 0.0-1.0")
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    visual_description: Optional[str] = Field(
        default=None,
        description="Comprehensive fine-grained visual description for 1:1 photorealistic digitisation",
    )
    pattern_detail: Optional[str] = Field(
        default=None,
        description="Specific pattern description including layout, colors, and orientation",
    )
    pocket_detail: Optional[str] = Field(
        default=None,
        description="Pocket placement, cut, and accent details",
    )
    button_detail: Optional[str] = Field(
        default=None,
        description="Button style, color, and spacing",
    )
    collar_detail: Optional[str] = Field(
        default=None,
        description="Collar style and neck details",
    )
    brand_label: Optional[str] = Field(
        default=None,
        description="Visible brand label or text",
    )

    # --- Enriched attributes (trimmed subset of the 144-field schema, added 2026-09-06) ---
    colour_secondary_name: Optional[str] = Field(default=None, description="Secondary/accent colour name")
    colour_temperature: Optional[ColourTemperatureEnum] = Field(default=None)
    colour_saturation: Optional[ColourSaturationEnum] = Field(default=None)
    pattern_motif: Optional[PatternMotifEnum] = Field(default=None)
    pattern_density: Optional[PatternDensityEnum] = Field(default=None)
    embellishment: Optional[EmbellishmentEnum] = Field(
        default=None, description="Embellishment technique, incl. Indian ethnic techniques (zardozi/kantha/mukaish/chikankari)"
    )
    embellishment_density: Optional[EmbellishmentDensityEnum] = Field(default=None)
    embellishment_placement: Optional[EmbellishmentPlacementEnum] = Field(default=None)
    fabric_class: Optional[FabricClassEnum] = Field(default=None)
    weave_type: Optional[WeaveTypeEnum] = Field(default=None)
    transparency: Optional[TransparencyEnum] = Field(default=None)
    sheen: Optional[SheenEnum] = Field(default=None)
    drape: Optional[DrapeEnum] = Field(default=None)
    stretch_grade: Optional[StretchGradeEnum] = Field(default=None)
    neckline: Optional[NecklineEnum] = Field(default=None)
    collar_type: Optional[CollarTypeEnum] = Field(default=None)
    sleeve_type: Optional[SleeveTypeEnum] = Field(default=None)
    closure_type: Optional[ClosureTypeEnum] = Field(default=None)
    hem_style: Optional[HemStyleEnum] = Field(default=None)
    rise: Optional[RiseEnum] = Field(default=None)
    leg_shape: Optional[LegShapeEnum] = Field(default=None)
    waistband_style: Optional[WaistbandStyleEnum] = Field(default=None)
    shoe_silhouette: Optional[ShoeSilhouetteEnum] = Field(default=None)
    accessory_kind: Optional[AccessoryKindEnum] = Field(default=None)
    formality: Optional[FormalityEnum] = Field(default=None)
    mood_intensity: Optional[MoodIntensityEnum] = Field(default=None)
    vibe_words: Optional[List[str]] = Field(default=None, description="Short free-form styling/vibe descriptor words")
    wash_state_visible: Optional[WashStateVisibleEnum] = Field(default=None)

    @field_validator(
        "colour_temperature", "colour_saturation", "pattern_motif", "pattern_density", "embellishment",
        "embellishment_density", "embellishment_placement", "fabric_class", "weave_type", "transparency",
        "sheen", "drape", "stretch_grade", "neckline", "collar_type", "sleeve_type", "closure_type",
        "hem_style", "rise", "leg_shape", "waistband_style", "shoe_silhouette", "accessory_kind",
        "formality", "mood_intensity", "wash_state_visible",
        mode="before",
    )
    @classmethod
    def normalize_optional_enum(cls, v, info: ValidationInfo):
        # A model asked ~28 extra questions per garment will very reasonably answer
        # "not_applicable" in prose for an irrelevant field even though these fields are
        # already Optional and the prompt asks for null — coerce that into a real None
        # instead of a hard enum-validation error.
        if v is None:
            return v
        if isinstance(v, str):
            clean_v = v.lower().strip().replace(" ", "_").replace("-", "_")
            if clean_v in _OPTIONAL_ENUM_NONE_TOKENS:
                return None
            enum_cls = _OPTIONAL_ENUM_FIELD_TYPES.get(info.field_name)
            if enum_cls is not None:
                valid_values = {member.value for member in enum_cls}
                if clean_v not in valid_values:
                    # A cross-field mixup (a valid value for a NEIGHBORING field, e.g. a
                    # neckline word answered under sleeve_type) — drop it rather than crash
                    # the whole extraction over one optional field.
                    return None
            return clean_v
        return v

    @field_validator("fit", mode="before")
    @classmethod
    def normalize_fit(cls, v):
        # A vision model asked for "fit" will sometimes answer with a SilhouetteEnum word
        # instead (most commonly "fitted", which isn't a valid FitEnum value — it describes
        # silhouette, not fit) — confirmed live this recurring mismatch was crashing extraction
        # outright (a hard schema error) rather than routing to a sensible fit value, which
        # then aborted a cross-model retry before the second model even got a chance to answer.
        if isinstance(v, str):
            clean_v = v.lower().strip().replace(" ", "_").replace("-", "_")
            return FIT_ALIASES.get(clean_v, clean_v)
        return v

    @field_validator("subcategory")
    @classmethod
    def validate_subcategory_taxonomy(cls, v: str) -> str:
        clean_v = v.lower().strip().replace(" ", "_").replace("-", "_")
        clean_v = SUBCATEGORY_ALIASES.get(clean_v, clean_v)
        if clean_v in KNOWN_SUBCATEGORIES:
            return clean_v

        # A vision model asked to pick from a fixed list will still sometimes prepend a
        # descriptive modifier it wants to convey (e.g. "graphic_tshirt" for a tshirt with a
        # graphic print, which is also separately captured in the `pattern` field). Rather than
        # reject a garment that's actually correctly identified purely because of an unrequested
        # modifier, match against the longest known subcategory that's a suffix of the given
        # value (longest-first so "polo_shirt" wins over a shorter unrelated match).
        for known in sorted(KNOWN_SUBCATEGORIES, key=len, reverse=True):
            if clean_v.endswith(f"_{known}") or clean_v == known:
                return known

        # No match even with the suffix fallback — return the cleaned value as-is; the
        # taxonomy check downstream (step 5) will route this to REVIEW_REQUIRED.
        return clean_v

    @field_validator("colour")
    @classmethod
    def validate_colours(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Garment must have at least one color")
        return [c.lower().strip() for c in v if c.strip()]

    @field_validator("garment_class")
    @classmethod
    def validate_garment_class(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v.upper().strip().replace(" ", "_").replace("-", "_")


class AttributeValidationError(Exception):
    """Raised when attribute validation fails at any of the 7 stages."""
    def __init__(self, stage: str, message: str, details: Any = None):
        super().__init__(f"Validation failed at stage '{stage}': {message}")
        self.stage = stage
        self.message = message
        self.details = details


def validate_extracted_attributes(raw_input: Any, min_confidence: float = 0.5) -> GarmentAttributes:
    """
    Executes the strict 7-step validation pipeline required by Section 10:
    1. Parse JSON
    2. Pydantic/schema validation
    3. Enum validation
    4. Range validation
    5. Taxonomy validation
    6. Required fields check
    7. Confidence checks
    """
    # Step 1: Parse JSON if string
    data = raw_input
    if isinstance(raw_input, str):
        try:
            data = json.loads(raw_input)
        except Exception as e:
            raise AttributeValidationError("1_parse_json", f"Invalid JSON string: {e}")

    if not isinstance(data, dict):
        raise AttributeValidationError("1_parse_json", f"Expected dictionary or JSON object, got {type(data)}")

    # Step 6: Required fields check (pre-check required keys before Pydantic coercion)
    required_keys = [
        "category", "subcategory", "colour", "pattern", "material",
        "fit", "silhouette", "occasion", "season", "layering_role",
        "warmth", "versatility"
    ]
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        raise AttributeValidationError("6_required_fields", f"Missing required fields: {missing}")

    # Step 4: Range validation pre-check
    for range_field in ("warmth", "versatility"):
        val = data.get(range_field)
        if val is not None:
            try:
                num_val = float(val)
                if num_val < 0.0 or num_val > 1.0:
                    raise AttributeValidationError(
                        "4_range_validation",
                        f"Field '{range_field}' value {num_val} is outside allowed range [0.0, 1.0]"
                    )
            except ValueError:
                raise AttributeValidationError("4_range_validation", f"Field '{range_field}' is not numeric")

    # Step 2 & 3: Pydantic schema validation & Enum validation
    try:
        attributes = GarmentAttributes.model_validate(data)
    except Exception as e:
        # Check if error was enum related
        err_msg = str(e)
        if "Input should be" in err_msg or "Enum" in err_msg:
            raise AttributeValidationError("3_enum_validation", f"Enum validation error: {err_msg}", details=e)
        raise AttributeValidationError("2_schema_validation", f"Pydantic schema validation error: {err_msg}", details=e)

    # Step 5: Taxonomy validation
    subcat = attributes.subcategory
    if subcat not in KNOWN_SUBCATEGORIES:
        raise AttributeValidationError(
            "5_taxonomy_validation",
            f"Subcategory '{subcat}' is not in known taxonomy list"
        )

    # garment_class (SPEC.md Section 8-10): derive if missing, remap unmapped
    # classes to a *_OTHER bucket rather than rejecting (SPEC.md Section 37 -
    # taxonomy mapping failures must never silently discard the garment).
    from app.rules.garment_class import GARMENT_CLASSES, infer_garment_class_from_subcategory

    if not attributes.garment_class:
        attributes.garment_class = infer_garment_class_from_subcategory(subcat)
    elif attributes.garment_class not in GARMENT_CLASSES:
        attributes.garment_class = infer_garment_class_from_subcategory(subcat)

    # Cross-field consistency: extraction providers score each field somewhat independently
    # (e.g. MODA_NER's structural track vs. the VLM top-up), which can produce a subcategory
    # and sleeve_length that contradict each other — a "tank_top" is sleeveless by definition,
    # so a co-extracted "three_quarter" sleeve_length is simply wrong, not a real variant. The
    # subcategory is the more reliable signal here (it's what a human actually asked for /
    # would visually recognize), so it wins when the two disagree.
    SLEEVELESS_SUBCATEGORIES = {"tank_top", "tube_top"}
    if subcat in SLEEVELESS_SUBCATEGORIES and attributes.sleeve_length != SleeveLengthEnum.SLEEVELESS:
        attributes.sleeve_length = SleeveLengthEnum.SLEEVELESS

    # Step 7: Confidence checks
    if attributes.confidence is not None and attributes.confidence < min_confidence:
        raise AttributeValidationError(
            "7_confidence_checks",
            f"Extraction confidence {attributes.confidence} is below minimum threshold {min_confidence}"
        )

    return attributes
