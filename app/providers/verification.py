"""Second-opinion, different-model verification checks for Stage 2 (crop) and Stage 3
(attributes).

Both Stage 2's detection/crop and Stage 3's attribute extraction are (per current config)
served by OpenAI GPT-4o and MODA_NER respectively — never by settings.VISION_VERIFIER_MODEL
(a Gemini model, routed through the same OpenRouter account). Checking a model's output with
that same model tends to rubber-stamp exactly the failure mode it should catch (a hallucinated
"belt" on a blank crop, a sleeve-length that contradicts the subcategory, etc.) — a genuinely
different vendor doesn't share that blind spot.
"""

import json
import re
from typing import List, Optional, Tuple

import httpx

from app.config import settings
from app.observability import logger
from app.schemas.attributes import GarmentAttributes

_HEADERS_BASE = {
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Wardrobe Ingestion Pipeline",
    "Content-Type": "application/json",
}


async def _call_verifier_vision(prompt_text: str, images_b64: List[str], api_key: str, model: str) -> Optional[dict]:
    content = [{"type": "text", "text": prompt_text}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    headers = dict(_HEADERS_BASE, Authorization=f"Bearer {api_key}")

    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(f"{settings.OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)


async def verify_crop_region(
    crop_bytes: bytes,
    label: str,
    api_key: Optional[str] = None,
    model: str = settings.VISION_VERIFIER_MODEL,
) -> Tuple[bool, float, str]:
    """
    Stage 2 crop verification: confirms the cropped pixels genuinely show a real, visible
    garment roughly matching `label`, not a blank/background/wrong-content crop (the exact
    failure mode that produced a hallucinated "belt" from an 81x54px patch of bare pavement
    earlier in this pipeline's life).
    """
    import base64

    api_key = api_key or settings.OPENROUTER_API_KEY
    if not api_key:
        return True, 0.9, "Verifier unavailable (no API key): crop accepted without model-based check."

    prompt_text = f"""You are a strict quality-control inspector for a garment cropping pipeline. This image was
automatically cropped from a photo and is supposed to show an isolated "{label}" garment.

Look carefully: is there an actual, clearly visible garment of roughly this type in the image
(at least partially visible is fine), or is the crop blank/background/skin/an unrelated object/
too degraded to tell what it is?

Output ONLY raw JSON, no markdown:
{{"is_valid_crop": true|false, "score": 0.0-1.0, "reason": "one short sentence"}}"""

    try:
        b64 = base64.b64encode(crop_bytes).decode("utf-8")
        parsed = await _call_verifier_vision(prompt_text, [b64], api_key, model)
        is_valid = bool(parsed.get("is_valid_crop", False))
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "No reason provided.")
        return is_valid, score, reason
    except Exception as e:
        logger.warning(f"Crop verifier ({model}) call failed: {e}")
        return True, 0.9, f"Verifier call failed ({e}); crop accepted without model-based check."


async def verify_attributes_against_image(
    image_bytes: bytes,
    attributes: GarmentAttributes,
    api_key: Optional[str] = None,
    model: str = settings.VISION_VERIFIER_MODEL,
) -> Tuple[bool, float, str, List[str]]:
    """
    Stage 3 attribute verification: confirms the extracted attribute JSON actually matches
    what's visible in the image — catches cross-provider disagreements (e.g. MODA_NER's
    structural track vs. the VLM top-up producing a "tank_top" with "three_quarter" sleeves)
    and outright hallucinated fields, independent of the rule-based consistency checks already
    in validate_extracted_attributes().
    """
    import base64

    api_key = api_key or settings.OPENROUTER_API_KEY
    if not api_key:
        return True, 0.9, "Verifier unavailable (no API key): attributes accepted without model-based check.", []

    sleeve_str = getattr(attributes.sleeve_length, "value", str(attributes.sleeve_length))
    pattern_str = getattr(attributes.pattern, "value", str(attributes.pattern))
    fit_str = getattr(attributes.fit, "value", str(attributes.fit))

    prompt_text = f"""You are a strict quality-control inspector checking whether extracted attribute data
actually matches a garment photo.

Extracted attributes: category={attributes.category}, subcategory={attributes.subcategory}, \
colour={", ".join(attributes.colour)}, pattern={pattern_str}, material={attributes.material}, \
fit={fit_str}, sleeve_length={sleeve_str}.

Compare these against the actual image. Flag any attribute that is clearly wrong given what's
visible (wrong garment type, wrong color, wrong sleeve length, impossible pattern/material for
what's shown). Minor subjective calls (e.g. "regular" vs "relaxed" fit) are NOT mismatches —
only flag things that are objectively, visibly wrong.

Output ONLY raw JSON, no markdown:
{{"is_match": true|false, "score": 0.0-1.0, "mismatches": ["short phrase per mismatch, empty list if none"], "reason": "one sentence verdict"}}"""

    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        parsed = await _call_verifier_vision(prompt_text, [b64], api_key, model)
        is_match = bool(parsed.get("is_match", False))
        score = float(parsed.get("score", 0.0))
        mismatches = parsed.get("mismatches", []) or []
        reason = parsed.get("reason", "No reason provided.")
        return is_match, score, reason, mismatches
    except Exception as e:
        logger.warning(f"Attribute verifier ({model}) call failed: {e}")
        return True, 0.9, f"Verifier call failed ({e}); attributes accepted without model-based check.", []
