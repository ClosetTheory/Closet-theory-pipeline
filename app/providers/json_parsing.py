"""Parsing JSON out of a language model's reply.

Every provider in this package asks a model for JSON and then has to get a Python object back
out of whatever actually arrived. That is not the same problem as `json.loads`, because models
reliably do three things the standard parser rejects:

1. **Markdown fences.** ```` ```json\\n{...}\\n``` ```` is the single most common shape, and
   nothing in this codebase stripped them. `response_format={"type": "json_object"}` helps but
   is not honoured by every model or every OpenRouter route, and the fallback models used on
   retry are exactly the ones least likely to honour it.
2. **Prose around the JSON.** "Here is the result: {...} — let me know if you need more."
3. **Trailing commas**, a habit picked up from JavaScript.

Several call sites tried to cope with `re.search(r"\\{.*\\}", text, re.DOTALL)`. That looks
right and is subtly wrong: `.*` is greedy, so it spans from the FIRST `{` to the LAST `}` in the
whole reply. Given `{"score": 4} because {item} was off`, it captures the entire span and fails
to parse — turning a recoverable reply into a total loss. It also cannot handle a reply whose
top level is a JSON array.

So this module scans for the first *balanced* JSON value instead, tracking string literals and
escapes so a brace inside a quoted string cannot end the scan early.
"""

import json
import re
from typing import Any, Optional

# ```json ... ```  /  ```JSON ... ```  /  ``` ... ```  — with or without a trailing newline.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
# A comma immediately before a closing brace/bracket, ignoring whitespace.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

_OPENERS = {"{": "}", "[": "]"}


class ModelJSONError(ValueError):
    """The model's reply could not be read as JSON.

    Carries a truncated preview of what actually arrived, because "Expecting value: line 1
    column 1" on its own tells you nothing about which model said what.
    """

    def __init__(self, message: str, raw: str, context: str = ""):
        self.raw = raw
        self.context = context
        preview = raw if len(raw) <= 400 else raw[:400] + f"… (+{len(raw) - 400} more chars)"
        where = f" [{context}]" if context else ""
        super().__init__(f"{message}{where}. Model returned: {preview!r}")


def strip_code_fences(text: str) -> str:
    """Removes one layer of markdown fencing, if the whole reply is fenced."""
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def find_balanced_json(text: str) -> Optional[str]:
    """Returns the first balanced JSON object or array in `text`, or None.

    Walks the string tracking depth, and skips anything inside a quoted string so that a brace
    in a value (`{"reason": "the {} looked wrong"}`) cannot close the scan early. This is the
    part the previous greedy regex got wrong.
    """
    start = None
    closer = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char in _OPENERS:
                start, closer, depth = index, _OPENERS[char], 1
            continue

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in _OPENERS:
            depth += 1
        elif char in ("}", "]"):
            depth -= 1
            if depth == 0:
                # Only accept it if the value closed with the bracket type it opened with;
                # a mismatch means the reply is malformed, not merely wrapped in prose.
                return text[start : index + 1] if char == closer else None

    return None  # unbalanced — usually a reply truncated by max_tokens


def parse_model_json(text: Optional[str], context: str = "") -> Any:
    """Reads a model reply as JSON, coping with fences, surrounding prose and trailing commas.

    Raises ModelJSONError rather than returning a default, so a caller decides what a failure
    means. Silently substituting `{}` here is how a broken model reply becomes a confidently
    wrong garment attribute — the failure mode this pipeline has been bitten by repeatedly.
    """
    if text is None or not text.strip():
        raise ModelJSONError("Model returned an empty reply", text or "", context)

    raw = text.strip()
    candidates = []

    stripped = strip_code_fences(raw).strip()
    candidates.append(stripped)
    if stripped != raw:
        candidates.append(raw)

    for candidate in list(candidates):
        balanced = find_balanced_json(candidate)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as e:
                last_error = e

    raise ModelJSONError(f"Could not parse JSON from model reply ({last_error})", raw, context)


def parse_model_json_or_none(text: Optional[str], context: str = "") -> Optional[Any]:
    """Same, for the call sites whose contract is already "degrade rather than raise"."""
    try:
        return parse_model_json(text, context)
    except ModelJSONError:
        return None
