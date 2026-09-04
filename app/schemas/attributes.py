"""Garment attributes schema and 7-step validation pipeline."""

from enum import Enum
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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
    "belt", "scarf", "hat", "cap", "tie", "gloves", "bag", "sunglasses"
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
