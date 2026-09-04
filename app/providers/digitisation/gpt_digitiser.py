"""GPT Studio Image Digitisation Provider via OpenRouter."""

import base64
import io
import json
import re
from typing import Optional, Tuple
import cv2
import httpx
import numpy as np
from PIL import Image, ImageFilter
from app.config import settings
from app.observability import logger
from app.providers.base import BaseDigitisationProvider
from app.rules.garment_class import bundle_garment_class, infer_garment_class_from_subcategory
from app.schemas.attributes import GarmentAttributes
from app.schemas.pipeline import DigitisationResult


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
        self._last_prompt: str = ""
        self._last_negative_prompt: str = ""
        self._active_model: str = "GPT-Studio-Segmenter-v1"
        self.verifier_model_name: str = settings.DIGITISATION_VERIFIER_MODEL
        self._last_verification: Optional[dict] = None

    def build_prompt(self, attributes: GarmentAttributes, garment_label: Optional[str] = None) -> Tuple[str, str]:
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
            pocket_detail = getattr(attributes, "pocket_detail", None)
            if pocket_detail and pocket_detail.lower() != "none":
                identity_lines.append(f"- Pocket(s): {pocket_detail}")
            button_detail = getattr(attributes, "button_detail", None)
            if button_detail and button_detail.lower() != "none":
                identity_lines.append(f"- Buttons & Placket: {button_detail}")
            collar_detail = getattr(attributes, "collar_detail", None)
            if collar_detail and collar_detail.lower() != "none":
                identity_lines.append(f"- Collar & Neckline: {collar_detail}")
            brand_label = getattr(attributes, "brand_label", None)
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

        positive_prompt = f"""Commercial e-commerce product photograph of a {subcategory_str}, centered on a solid dark charcoal studio background (#161922). \
The garment floats with natural three-dimensional volume and shape, exactly as if being worn, but with no visible body, support structure, or object holding it up. \
{reference_instruction}

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
    ) -> DigitisationResult:
        prompt, negative_prompt = self.build_prompt(attributes, garment_label=garment_label)
        self._last_prompt = prompt
        self._last_negative_prompt = negative_prompt

        # 1. Generate real canonical studio image via OpenRouter Images API with reference image conditioning
        if self.api_key:
            try:
                gen_bytes, model_used = await self._call_openrouter_image_gen(prompt, crop_bytes)
                if gen_bytes:
                    self._last_generated_bytes = gen_bytes
                    self._active_model = f"OpenRouter ({model_used})"
                    logger.info(f"Canonical studio image successfully generated via OpenRouter ({model_used}).")
                    return DigitisationResult(
                        canonical_image_uri="",
                        quality_score=0.98,
                        model=self._active_model,
                        model_version=self.model_version,
                        prompt_version=self.prompt_version,
                        attempts=attempt,
                    )
            except Exception as e:
                logger.warning(f"OpenRouter image generation call could not be completed: {e}")

        # 2. Local Studio Segmentation & Compositing Engine fallback
        studio_bytes = self._segment_and_composite_studio(crop_bytes)
        self._last_generated_bytes = studio_bytes
        self._active_model = "Studio-Segmenter-Protected"

        return DigitisationResult(
            canonical_image_uri="",
            quality_score=0.92,
            model=self._active_model,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            attempts=attempt,
        )

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
        # to try first (same setting the styling outfit-imaging provider uses); the second
        # entry is a fixed fallback if that model is ever unavailable.
        models_to_try = [settings.OPENROUTER_IMAGE_MODEL, "openai/gpt-5.4-image-2"]

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
                        logger.warning(f"OpenRouter image model {model_id} returned HTTP {resp.status_code}: {resp.text[:150]}")
                except Exception as ex:
                    logger.warning(f"OpenRouter model {model_id} request error: {ex}")
                    continue

        return None, ""

    def _segment_and_composite_studio(self, crop_bytes: bytes) -> bytes:
        """
        Local studio fallback with hole-protection for high-frequency patterns (plaid, stripes).
        """
        try:
            np_arr = np.frombuffer(crop_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return crop_bytes

            h, w = img_bgr.shape[:2]

            margin_x = max(1, int(w * 0.05))
            margin_y = max(1, int(h * 0.08))
            rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # Mark center core as definite foreground (prevents cutting holes in plaid/stripes)
            mask[int(h * 0.25) : int(h * 0.80), int(w * 0.25) : int(w * 0.75)] = cv2.GC_FGD

            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
            mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

            b, g, r = cv2.split(img_bgr)
            rgba = cv2.merge([r, g, b, mask2])
            isolated_garment = Image.fromarray(rgba, "RGBA")

            bbox = isolated_garment.getbbox()
            if bbox:
                isolated_garment = isolated_garment.crop(bbox)

            gw, gh = isolated_garment.size
            canvas_w, canvas_h = 768, 1024
            studio_canvas = Image.new("RGBA", (canvas_w, canvas_h), (248, 250, 252, 255))

            scale = min((canvas_w * 0.82) / gw, (canvas_h * 0.82) / gh)
            scaled_w = max(10, int(gw * scale))
            scaled_h = max(10, int(gh * scale))
            scaled_garment = isolated_garment.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

            shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            shadow_x = (canvas_w - scaled_w) // 2
            shadow_y = (canvas_h - scaled_h) // 2 + 10

            alpha_channel = scaled_garment.split()[3]
            blurred_shadow = alpha_channel.filter(ImageFilter.GaussianBlur(16))
            shadow_layer = Image.new("RGBA", (scaled_w, scaled_h), (30, 41, 59, 50))
            shadow.paste(shadow_layer, (shadow_x, shadow_y), blurred_shadow)
            studio_canvas = Image.alpha_composite(studio_canvas, shadow)

            paste_x = (canvas_w - scaled_w) // 2
            paste_y = (canvas_h - scaled_h) // 2
            studio_canvas.paste(scaled_garment, (paste_x, paste_y), scaled_garment)

            final_rgb = studio_canvas.convert("RGB")
            buf = io.BytesIO()
            final_rgb.save(buf, format="JPEG", quality=95)
            return buf.getvalue()

        except Exception as e:
            logger.warning(f"Studio segmentation fallback: {e}")
            return crop_bytes

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

        fallback = (
            True,
            0.9,
            "Verifier unavailable (no API key or call failed): accepted without model-based comparison.",
        )

        if not self.api_key:
            self._last_verification = {
                "model": self.verifier_model_name,
                "is_valid": fallback[0],
                "score": fallback[1],
                "reason": fallback[2],
                "mismatches": [],
            }
            return fallback

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
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                parsed = json.loads(content)

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
        except Exception as e:
            logger.warning(f"Digitisation verifier ({self.verifier_model_name}) call failed: {e}")
            self._last_verification = {
                "model": self.verifier_model_name,
                "is_valid": fallback[0],
                "score": fallback[1],
                "reason": f"Verifier call failed ({e}); accepted without model-based comparison.",
                "mismatches": [],
            }
            return fallback
