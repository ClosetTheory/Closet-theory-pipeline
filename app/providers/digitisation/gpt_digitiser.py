"""GPT Studio Image Digitisation Provider via OpenRouter."""

import base64
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx
from app.config import settings
from app.providers.json_parsing import parse_model_json
from app.observability import logger
from app.providers.base import (
    BaseDigitisationProvider,
    ImageGenerationRefusedError,
    VerifierUnavailableError,
)
from app.rules.garment_class import bundle_garment_class, infer_garment_class_from_subcategory
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import DigitisationResult


_PLACEHOLDER_TEXT = {"none", "null", "n/a", "na", "not_applicable", "not applicable", "unknown", ""}


def _meaningful(value: Optional[str]) -> Optional[str]:
    """Return the text only if it actually says something.

    An extractor answering the *string* "null" instead of JSON null yields a truthy value that
    reads as real content. Observed live: 31 garments carried brand_label = "null", so the
    prompt asked for an inner label reading "null" and the image model drew exactly that into
    the garment's neck. GarmentAttributes now strips these at ingestion; this keeps already
    stored rows from leaking the same text into a prompt without re-extracting them.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return None if cleaned.lower() in _PLACEHOLDER_TEXT else cleaned


def _extract_provider_message(body: str) -> str:
    """Pulls the human-readable message out of a provider error body, falling back to the raw
    text. Providers nest it inconsistently, and the useful sentence is always the deepest one."""
    try:
        data = json.loads(body)
    except Exception:
        return (body or "").strip()[:400]
    for path in (("error", "message"), ("error", "metadata", "raw"), ("message",), ("detail",)):
        node = data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str) and node.strip():
            return node.strip()[:400]
    return (body or "").strip()[:400]


def _classify_generation_failure(status: int, message: str) -> str:
    """Names the failure so the UI can say something useful rather than showing a status code.

    The distinction that matters is refusal versus outage: a safety rejection is a permanent
    property of this garment's print and will never succeed on retry, whereas a rate limit or a
    502 is worth coming back to.
    """
    text = (message or "").lower()
    if "safety" in text or "content policy" in text or "rejected by the safety" in text:
        return "safety_refusal"
    if status in (401, 403) or "api key" in text or "unauthorized" in text:
        return "auth"
    if status == 429 or "rate limit" in text or "quota" in text or "credit" in text:
        return "rate_limited"
    if status and status >= 500:
        return "provider_error"
    return "rejected"


class GPTStudioDigitisationProvider(BaseDigitisationProvider):
    """
    GPT-guided Canonical Studio Digitisation.
    1. Injects validated garment identity (type, color, pattern, material, cut) into prompt.
    2. Sends reference image directly to OpenRouter `openai/gpt-image-2` for image-to-image synthesis.
    3. Safe local fallback with hole-protected garment segmentation on off-white studio backdrop.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.DIGITISATION_MODEL_NAME,
        model_version: str = settings.DIGITISATION_MODEL_VERSION,
        prompt_version: str = settings.DIGITISATION_PROMPT_VERSION,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name
        self.model_version = model_version
        self.prompt_version = prompt_version
        self._last_generated_bytes: Optional[bytes] = None
        # Why each image model declined or failed, in the order tried. Surfaced through the
        # stage run so the reason survives a container restart and reaches the UI.
        self._last_generation_errors: List[Dict[str, Any]] = []
        # Set when the image came from something other than the first-choice model, so the run
        # can say which model drew the garment and what the preferred one said.
        self._last_fallback: Optional[Dict[str, Any]] = None
        self._last_prompt: str = ""
        self._last_negative_prompt: str = ""
        self._active_model: str = "GPT-Studio-Segmenter-v1"
        self.verifier_model_name: str = settings.DIGITISATION_VERIFIER_MODEL
        self._last_verification: Optional[dict] = None

    def build_prompt(
        self,
        attributes: GarmentAttributes,
        garment_label: Optional[str] = None,
        previous_rejections: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, str]:
        """
        Builds a 1:1-preservation e-commerce catalogue product-shot prompt from the real
        extracted attributes only. Never invents specifics for a field the extractor didn't
        populate (no fake brand labels, no fake pocket/button descriptions) — an invented
        detail tells the image model to draw something that isn't actually on the garment,
        which is precisely what produces an unrelated-looking result. Any structural feature
        not confirmed by the reference photo is instead left to "match the reference image
        exactly" rather than described from a guess.

        The identity section is also garment-category-aware: pockets/buttons/collars only
        get asked for on tops/outerwear/one-pieces that plausibly have them — describing a
        "chest pocket" and "collar" on a pair of shoes or trousers is exactly the kind of
        mismatch that makes the generated result look like a different, wrong garment.
        """
        colors_str = ", ".join(attributes.colour) if attributes.colour else "as shown in the reference photo"
        pattern_str = getattr(attributes.pattern, "value", str(attributes.pattern)) if attributes.pattern else None
        subcategory_str = (attributes.subcategory or attributes.category or "garment").replace("_", " ")
        material_str = getattr(attributes.material, "value", str(attributes.material)) if attributes.material else None
        sleeve_str = getattr(attributes.sleeve_length, "value", str(attributes.sleeve_length)) if attributes.sleeve_length else None
        fit_str = getattr(attributes.fit, "value", str(attributes.fit)) if attributes.fit else None
        silhouette_str = getattr(attributes.silhouette, "value", str(attributes.silhouette)) if attributes.silhouette else None

        garment_class = attributes.garment_class or infer_garment_class_from_subcategory(attributes.subcategory or "")
        category, _version, _requires_review = bundle_garment_class(garment_class) if garment_class else (None, "", True)
        has_collar_buttons_pockets = category in ("TOP", "OUTERWEAR", "ONE_PIECE")
        has_sleeves = category in ("TOP", "OUTERWEAR", "ONE_PIECE") and sleeve_str and sleeve_str != "not_applicable"

        identity_lines = []
        type_desc = f"{fit_str + ' fit, ' if fit_str else ''}{silhouette_str + ' silhouette ' if silhouette_str else ''}{subcategory_str}"
        if has_sleeves:
            type_desc += f" with {sleeve_str} sleeves"
        identity_lines.append(f"- Garment Type: {type_desc}")
        if material_str:
            identity_lines.append(f"- Fabric & Material: {material_str} fabric texture, matching the weave/texture shown in the reference photo")
        identity_lines.append(f"- Color Palette: {colors_str}")

        # Pattern/pocket/button/collar/brand: only describe what's actually known — otherwise
        # defer entirely to the reference photo rather than inventing a specific that may not exist.
        pattern_detail = getattr(attributes, "pattern_detail", None)
        if pattern_detail:
            identity_lines.append(f"- Pattern Structure: {pattern_detail}")
        elif pattern_str and pattern_str != "solid":
            identity_lines.append(f"- Pattern Structure: {pattern_str} pattern, matching the reference photo exactly")

        if has_collar_buttons_pockets:
            pocket_detail = _meaningful(getattr(attributes, "pocket_detail", None))
            if pocket_detail:
                identity_lines.append(f"- Pocket(s): {pocket_detail}")
            button_detail = _meaningful(getattr(attributes, "button_detail", None))
            if button_detail:
                identity_lines.append(f"- Buttons & Placket: {button_detail}")
            collar_detail = _meaningful(getattr(attributes, "collar_detail", None))
            if collar_detail:
                identity_lines.append(f"- Collar & Neckline: {collar_detail}")
            brand_label = _meaningful(getattr(attributes, "brand_label", None))
            if brand_label:
                identity_lines.append(f"- Inside the neck opening, an inner label reads '{brand_label}'")
            if has_sleeves:
                identity_lines.append(f"- Sleeves & Hem: Symmetrical {sleeve_str} sleeves positioned neatly alongside the torso, matching the cuff/hem style shown in the reference photo")

        identity_lines.append(
            "- Any other structural detail not listed above (trims, closures, hardware, seams, "
            "hem shape) must match the reference photo exactly — do not invent additional features."
        )

        # When the reference photo shows multiple garments (garment_label set), tell the model
        # which one to isolate — tested this session to produce far more faithful renders than
        # pixel-cropping first, since the model keeps the garment's true proportions/drape
        # instead of reconstructing them from a small/degraded crop.
        reference_instruction = (
            f"The reference photo shows a person wearing MULTIPLE garments/accessories. Isolate and reproduce "
            f"ONLY the {garment_label} — ignore every other garment, accessory, and the person entirely. "
            f"Use that garment in the reference photo as the ground truth for its real appearance — reproduce it "
            f"faithfully, do not substitute a generic or different item."
            if garment_label else
            "Use the provided reference photo as the ground truth for this exact garment's real appearance — "
            "reproduce it faithfully, do not substitute a generic or different item."
        )

        # Stated up front, because a print described in isolation ("a character with purple
        # flames") reads as a request to generate licensed artwork, and gets refused. Naming the
        # actual task resolves that: this is catalogue photography of an object the owner
        # already has, not the creation of new artwork.
        provenance = (
            "Context for this task: this is a wardrobe cataloguing service. The reference photo "
            "was taken by the owner of this physical garment, which they already possess, and is "
            "supplied with their permission for the sole purpose of building their own private "
            "digital wardrobe catalogue. The task is to re-photograph that existing object under "
            "standard studio conditions — equivalent to placing the same garment on a catalogue "
            "backdrop and shooting it again. Nothing is being invented, altered or restyled. Any "
            "printed graphic, logo or text on the garment is part of the physical object and "
            "should appear only as it already appears on it, at the same position and scale, as "
            "incidental detail of the product — never enlarged, isolated, extracted or reproduced "
            "as artwork in its own right."
        )

        positive_prompt = f"""Commercial e-commerce product photograph of a {subcategory_str}, centered on a solid dark charcoal studio background (#161922). \
The garment floats with natural three-dimensional volume and shape, exactly as if being worn, but with no visible body, support structure, or object holding it up. \
{reference_instruction}

### What this image is for:
{provenance}

### Exact Garment Identity (1:1 Preservation — Highest Priority):
{chr(10).join(identity_lines)}

### Presentation & Photography Style:
- Invisible-body 3D form: The garment has natural 3D volume with no body, form, or object visible inside it
- Symmetrical straight-on front-facing view, eye-level camera angle, perfectly centered composition
- Pristine e-commerce catalogue quality: perfectly ironed/cleaned, true-to-life colors
- Background: Solid dark charcoal studio backdrop (#161922) with seamless contrast
- Studio Lighting: Soft diffused commercial studio key lighting with subtle rim light outlining the garment silhouette. 8k resolution, ultra-sharp focus on fabric texture, no dramatic shadows"""

        visual_desc = getattr(attributes, "visual_description", None)
        if visual_desc:
            positive_prompt += f"\n\n### Detailed Visual Specifications (from the reference photo):\n{visual_desc}"

        # Tell the model exactly what the verifier rejected last time. Without this a retry is a
        # blind re-roll: the same invented detail reappears and eventually slips past a noisy
        # verifier, which is how a plain collar came back as a ruffled high collar and passed.
        correction_lines = []
        for rejection in previous_rejections or []:
            for mismatch in rejection.get("mismatches") or []:
                correction_lines.append(f"- {mismatch}")
            reason = (rejection.get("reason") or "").strip()
            if reason:
                correction_lines.append(f"- {reason}")
        if correction_lines:
            seen, deduped = set(), []
            for line in correction_lines:
                if line not in seen:
                    seen.add(line)
                    deduped.append(line)
            positive_prompt += (
                "\n\n### Corrections — a previous attempt at THIS garment was REJECTED (highest priority):\n"
                + "\n".join(deduped)
                + "\nDo not repeat these mistakes. Where the reference photo and your instinct for a "
                "'typical' garment of this type disagree, follow the reference photo — it is the real "
                "garment. Do not add decorative features (ruffles, frills, puffed sleeves, contrast "
                "trims, extra plackets) that are not clearly visible in the reference photo."
            )

        negative_prompt = (
            "different garment, wrong garment, generic garment, invented details not in the reference photo, "
            "human, person, face, skin, hands, arms, body, visible head, visible neck, visible support structure, "
            "dress form, dummy, hanger, rack, closet, cluttered background, white wall, "
            "slats, wrinkles, creases, asymmetrical, tilted, floating fabric, distorted pattern, "
            "blurry, low resolution, artifacts, dark shadows, watermark, text overlays, cartoon, illustration, stylized"
        )

        return positive_prompt, negative_prompt

    async def digitise(
        self,
        crop_bytes: bytes,
        attributes: GarmentAttributes,
        attempt: int = 1,
        garment_label: Optional[str] = None,
        previous_rejections: Optional[List[Dict[str, Any]]] = None,
    ) -> DigitisationResult:
        prompt, negative_prompt = self.build_prompt(
            attributes, garment_label=garment_label, previous_rejections=previous_rejections
        )
        self._last_prompt = prompt
        self._last_negative_prompt = negative_prompt
        self._last_generation_errors = []
        self._last_fallback = None

        # 1. Generate real canonical studio image via OpenRouter Images API with reference image conditioning
        if not self.api_key:
            raise ImageGenerationRefusedError(
                "OPENROUTER_API_KEY is not set, so no image model could be called."
            )

        gen_bytes, model_used = await self._call_openrouter_image_gen(prompt, crop_bytes)
        if gen_bytes:
            self._last_generated_bytes = gen_bytes
            self._active_model = f"OpenRouter ({model_used})"
            if model_used != settings.OPENROUTER_IMAGE_MODEL:
                declined = [e for e in self._last_generation_errors if e["model"] != model_used]
                self._last_fallback = {
                    "model": model_used,
                    "preferred_model": settings.OPENROUTER_IMAGE_MODEL,
                    "declined": declined,
                    "reason": declined[0]["reason"] if declined else "",
                    "kind": declined[0]["kind"] if declined else "",
                }
                logger.info(
                    f"Canonical studio image generated by fallback model {model_used}; "
                    f"{settings.OPENROUTER_IMAGE_MODEL} declined."
                )
            else:
                logger.info(f"Canonical studio image successfully generated via OpenRouter ({model_used}).")
            return DigitisationResult(
                canonical_image_uri="",
                quality_score=0.98,
                model=self._active_model,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                attempts=attempt,
            )

        # 2. Nothing drew it. There is deliberately no local composite here — see
        # ImageGenerationRefusedError. The garment goes to human review carrying the reason each
        # model gave, which is a real answer; a grabCut cut-out reported as a 0.92 success is not.
        summary = "; ".join(
            f"{e['model']}: {e['reason']}" for e in self._last_generation_errors
        ) or "no image model returned an image"
        raise ImageGenerationRefusedError(summary)

    async def _call_openrouter_image_gen(self, prompt: str, crop_bytes: bytes) -> Tuple[Optional[bytes], str]:
        """Calls OpenRouter /api/v1/images API conditioned on reference crop."""
        url = "https://openrouter.ai/api/v1/images"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Ingestion Pipeline",
            "Content-Type": "application/json",
        }

        # settings.OPENROUTER_IMAGE_MODEL is the single source of truth for which image model
        # to try first (same setting the styling outfit-imaging provider uses); the rest of the
        # ladder is OPENROUTER_IMAGE_FALLBACK_MODELS, which must reach a second vendor — see the
        # note on that setting for why a same-vendor fallback is not a fallback at all.
        models_to_try = [settings.OPENROUTER_IMAGE_MODEL] + [
            m.strip() for m in settings.OPENROUTER_IMAGE_FALLBACK_MODELS.split(",") if m.strip()
        ]
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        # Base64 encode the reference crop image for image-to-image conditioning
        b64_image = base64.b64encode(crop_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_image}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            for model_id in models_to_try:
                try:
                    payload = {
                        "model": model_id,
                        "prompt": prompt,
                        "input_references": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_uri,
                                },
                            }
                        ],
                    }
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        if "data" in data and len(data["data"]) > 0:
                            item = data["data"][0]
                            if "b64_json" in item:
                                raw_bytes = base64.b64decode(item["b64_json"])
                                return raw_bytes, model_id
                            elif "url" in item:
                                img_resp = await client.get(item["url"])
                                if img_resp.status_code == 200:
                                    return img_resp.content, model_id
                    else:
                        detail = _extract_provider_message(resp.text)
                        self._last_generation_errors.append({
                            "model": model_id,
                            "status": resp.status_code,
                            "reason": detail,
                            "kind": _classify_generation_failure(resp.status_code, detail),
                        })
                        logger.warning(
                            f"OpenRouter image model {model_id} returned HTTP {resp.status_code}: {detail[:150]}"
                        )
                except Exception as ex:
                    self._last_generation_errors.append({
                        "model": model_id,
                        "status": None,
                        "reason": f"{type(ex).__name__}: {ex}",
                        "kind": "transport",
                    })
                    logger.warning(f"OpenRouter model {model_id} request error: {ex}")
                    continue

        return None, ""

    async def validate_digitisation(
        self,
        original_crop_bytes: bytes,
        generated_bytes: bytes,
        attributes: GarmentAttributes,
        garment_label: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """
        Real vision-based verification of the generated canonical image against the
        original reference crop and the extracted attributes.

        Deliberately calls a different model/vendor (settings.DIGITISATION_VERIFIER_MODEL,
        default a Gemini model) than whatever generated the image (openai/gpt-image-2 or
        openai/gpt-5.4-image-2) — a verifier built on the same model family shares the same
        blind spots as the generator, so it would tend to rubber-stamp exactly the failure
        modes (wrong garment, hallucinated details, dropped sleeves, etc.) it should catch.
        """
        subcategory_str = (attributes.subcategory or attributes.category or "garment").replace("_", " ")
        colors_str = ", ".join(attributes.colour) if attributes.colour else "unspecified"
        sleeve_str = getattr(attributes.sleeve_length, "value", str(attributes.sleeve_length)) if attributes.sleeve_length else "not_applicable"
        pattern_str = getattr(attributes.pattern, "value", str(attributes.pattern)) if attributes.pattern else "solid"

        if not self.api_key:
            raise VerifierUnavailableError(
                "Digitisation verifier has no API key, so the generated image cannot be checked "
                "against the original. Refusing to report it as verified."
            )

        focus_note = (
            f"Image 1 shows a person wearing MULTIPLE garments — judge Image 2 only against the "
            f"{garment_label} in Image 1, ignoring every other garment/accessory in that photo.\n\n"
            if garment_label else ""
        )

        prompt_text = focus_note + f"""You are a strict quality-control inspector comparing two images of the SAME garment.
Image 1 is the ORIGINAL reference photo (ground truth). Image 2 is a GENERATED standardized studio image meant to depict the exact same garment in isolation.

Extracted attributes for this garment (for reference, not necessarily exhaustive): type={subcategory_str}, color(s)={colors_str}, sleeve_length={sleeve_str}, pattern={pattern_str}.

Check whether Image 2 faithfully preserves Image 1's garment: same garment type/category, same color(s), same sleeve length (e.g. do not accept long sleeves if the reference is sleeveless, or vice versa), same silhouette/pattern, and no hallucinated details (logos, text, pockets, accessories) that are not visible in Image 1. Minor differences in pose, lighting, or background are fine and expected — only flag differences in the garment ITSELF.

Output ONLY raw JSON, no markdown:
{{"is_match": true|false, "score": 0.0-1.0, "mismatches": ["short phrase per mismatch, empty list if none"], "reason": "one sentence verdict"}}"""

        b64_original = base64.b64encode(original_crop_bytes).decode("utf-8")
        b64_generated = base64.b64encode(generated_bytes).decode("utf-8")

        payload = {
            "model": self.verifier_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_original}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_generated}"}},
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Ingestion Pipeline",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                parsed = parse_model_json(content, context="digitisation verifier")

                is_match = bool(parsed.get("is_match", False))
                score = float(parsed.get("score", 0.0))
                mismatches = parsed.get("mismatches", []) or []
                reason = parsed.get("reason", "No reason provided.")

                self._last_verification = {
                    "model": self.verifier_model_name,
                    "is_valid": is_match,
                    "score": score,
                    "reason": reason,
                    "mismatches": mismatches,
                }
                return is_match, score, reason
        except VerifierUnavailableError:
            raise
        except Exception as e:
            logger.warning(f"Digitisation verifier ({self.verifier_model_name}) call failed: {e}")
            self._last_verification = {
                "model": self.verifier_model_name,
                "is_valid": None,
                "score": None,
                "reason": f"Verifier call failed: {type(e).__name__}: {e}",
                "mismatches": [],
            }
            raise VerifierUnavailableError(
                f"Digitisation verifier ({self.verifier_model_name}) call failed: "
                f"{type(e).__name__}: {e}"
            ) from e
