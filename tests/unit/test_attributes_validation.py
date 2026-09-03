"""Unit tests for Stage 3: Garment Attributes 7-Step Validation Pipeline."""

import copy
import json
import pytest
from app.schemas.attributes import (
    AttributeValidationError,
    GarmentAttributes,
    validate_extracted_attributes,
)


def test_valid_attributes_pass(valid_attributes_dict):
    attrs = validate_extracted_attributes(valid_attributes_dict)
    assert isinstance(attrs, GarmentAttributes)
    assert attrs.subcategory == "oxford_shirt"
    assert attrs.warmth == 0.25
    assert attrs.versatility == 0.85


def test_json_string_parsing(valid_attributes_dict):
    json_str = json.dumps(valid_attributes_dict)
    attrs = validate_extracted_attributes(json_str)
    assert attrs.subcategory == "oxford_shirt"


def test_step_1_invalid_json():
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes("{malformed json string,")
    assert exc.value.stage == "1_parse_json"


def test_step_6_missing_required_fields(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    del data["material"]
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data)
    assert exc.value.stage == "6_required_fields"
    assert "material" in exc.value.message


def test_step_4_warmth_out_of_range(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    data["warmth"] = 1.5
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data)
    assert exc.value.stage == "4_range_validation"


def test_step_4_versatility_out_of_range(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    data["versatility"] = -0.1
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data)
    assert exc.value.stage == "4_range_validation"


def test_step_3_invalid_enum(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    data["pattern"] = "rainbow_sparkles"  # Not a valid PatternEnum
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data)
    assert exc.value.stage == "3_enum_validation"


def test_step_5_unknown_taxonomy_subcategory(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    data["subcategory"] = "space_suit_helmet"
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data)
    assert exc.value.stage == "5_taxonomy_validation"


def test_step_7_low_confidence(valid_attributes_dict):
    data = copy.deepcopy(valid_attributes_dict)
    data["confidence"] = 0.20
    with pytest.raises(AttributeValidationError) as exc:
        validate_extracted_attributes(data, min_confidence=0.50)
    assert exc.value.stage == "7_confidence_checks"
