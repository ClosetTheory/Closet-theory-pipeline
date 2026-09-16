r"""Reading JSON out of a language model's reply.

Each case here is a shape a model has actually produced. The three marked below are ones the
previous `re.search(r"\{.*\}", text, re.DOTALL)` approach silently failed on, turning a
recoverable reply into a lost call.
"""

import pytest
from app.providers.json_parsing import (
    ModelJSONError,
    find_balanced_json,
    parse_model_json,
    parse_model_json_or_none,
    strip_code_fences,
)


@pytest.mark.parametrize(
    "label,reply,expected",
    [
        ("plain object", '{"score": 4}', {"score": 4}),
        ("fenced with language", '```json\n{"score": 4}\n```', {"score": 4}),
        ("fenced without language", '```\n{"score": 4}\n```', {"score": 4}),
        ("prose before", 'Here is the result: {"score": 4}', {"score": 4}),
        # The greedy regex spanned from the first "{" to the last "}", swallowing the prose and
        # its braces along with it.
        ("prose after containing braces", '{"score": 4} because {item} was off', {"score": 4}),
        ("trailing comma", '{"a": 1, "b": 2,}', {"a": 1, "b": 2}),
        ("top-level array", '[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
        ("fence wrapped in chat", 'Sure!\n```json\n{"score": 5}\n```\nHope that helps.', {"score": 5}),
        ("brace inside a string", '{"reason": "the {} looked wrong"}', {"reason": "the {} looked wrong"}),
        ("escaped quote inside a string", '{"reason": "he said \\"no\\""}', {"reason": 'he said "no"'}),
        ("nested with a brace value", '{"a": {"b": [1, 2]}, "c": "}"}', {"a": {"b": [1, 2]}, "c": "}"}),
    ],
)
def test_parses_real_model_output_shapes(label, reply, expected):
    assert parse_model_json(reply) == expected, label


@pytest.mark.parametrize(
    "label,reply",
    [
        ("empty", ""),
        ("whitespace only", "   \n  "),
        ("refusal prose", "I'm sorry, I can't help with that."),
        # max_tokens cut the reply mid-string. Must fail rather than half-parse.
        ("truncated", '{"score": 4, "reason": "it wa'),
        ("mismatched brackets", '{"a": 1]'),
    ],
)
def test_unusable_replies_raise_rather_than_guess(label, reply):
    """Substituting a default here is how a broken reply becomes a confidently wrong garment
    attribute, so the contract is to raise and let the caller decide."""
    with pytest.raises(ModelJSONError):
        parse_model_json(reply, context=label)


def test_error_carries_a_preview_of_what_arrived():
    with pytest.raises(ModelJSONError) as excinfo:
        parse_model_json("I cannot do that", context="classify (gpt-4o)")
    message = str(excinfo.value)
    assert "classify (gpt-4o)" in message, "context identifies which call failed"
    assert "I cannot do that" in message, "preview shows what the model actually said"


def test_long_replies_are_truncated_in_the_error():
    with pytest.raises(ModelJSONError) as excinfo:
        parse_model_json("x" * 5000)
    assert "more chars" in str(excinfo.value)
    assert len(str(excinfo.value)) < 700


def test_or_none_variant_degrades_instead_of_raising():
    assert parse_model_json_or_none("not json") is None
    assert parse_model_json_or_none('{"a": 1}') == {"a": 1}


def test_strip_code_fences_only_unwraps_a_fully_fenced_reply():
    assert strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'
    # A stray fence mid-prose is not a wrapper and must be left alone for the scanner.
    assert strip_code_fences('text ```json\n{"a":1}\n``` more') == 'text ```json\n{"a":1}\n``` more'


def test_find_balanced_json_stops_at_the_first_complete_value():
    assert find_balanced_json('{"a": 1} {"b": 2}') == '{"a": 1}'
    assert find_balanced_json("no json here") is None
    assert find_balanced_json('{"a": 1') is None, "unbalanced input is not a value"
