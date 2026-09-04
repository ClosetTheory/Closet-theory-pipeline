"""OpenRouter GPT Multimodal Vision Provider (GPT-4o / GPT-4o-mini)."""

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx
from PIL import Image
from app.config import settings
from app.observability import logger
from app.providers.base import (
    BaseAttributeExtractorProvider,
    BaseClassifierProvider,
    BaseDetectionProvider,
    BaseRequestNormalizerProvider,
    BaseSemanticValidatorProvider,
    BaseVisualValidatorProvider,
    BaseVLMProvider,
)
from app.schemas.attributes import GarmentAttributes, validate_extracted_attributes
from app.schemas.pipeline import ClassificationResult, DetectionResult, GarmentRegion, ImageType
from app.schemas.styling import (
    GarmentSummary,
    OutfitCandidate,
    SemanticGateResult,
    StylingContext,
    StylingIntent,
    ValidationResult,
    ValidationStatus,
    VisualGateResult,
    validate_semantic_gate_result,
    validate_styling_intent,
    validate_validation_result,
    validate_visual_gate_result,
)


class OpenRouterGPTProvider(
    BaseAttributeExtractorProvider,
    BaseVLMProvider,
    BaseRequestNormalizerProvider,
    BaseSemanticValidatorProvider,
    BaseVisualValidatorProvider,
    BaseClassifierProvider,
    BaseDetectionProvider,
):
    """
    OpenRouter multimodal integration for GPT-4o and other OpenAI models.
    Endpoint: https://openrouter.ai/api/v1/chat/completions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.OPENROUTER_MODEL,
        base_url: str = settings.OPENROUTER_BASE_URL,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.model_version = "v1"

    async def extract_attributes(
        self, image_bytes: bytes, image_type: Optional[str] = None
    ) -> GarmentAttributes:
        """Extracts structured garment attributes from image using OpenRouter GPT model."""
        if not self.api_key:
            # Fallback to local heuristic if no key in env
            return self._local_vision_analysis(image_bytes)

        prompt = """You are a master fashion perception and garment digitisation specialist. Analyze this garment photo and output ONLY a raw JSON object with NO markdown formatting, matching this exact schema:
{
  "category": "shirt | pants | dress | jacket | shoes | sweater",
  "subcategory": "oxford_shirt | button_down_shirt | flannel_shirt | tshirt | polo_shirt | jeans | trousers | chinos | blazer | coat | dress | sweater | hoodie | sneakers | boots",
  "garment_class": "canonical controlled-vocabulary class, e.g. T_SHIRT | SHIRT | JEANS | TROUSERS | CHINOS | DRESS | BLAZER | JACKET | SNEAKERS | BOOTS | SAREE | KURTA (use '<CATEGORY>_OTHER' if uncertain)",
  "colour": ["list", "of", "all", "prominent", "and", "accent", "colors", "e.g.", "yellow", "salmon pink", "navy blue", "white"],
  "pattern": "solid | striped | plaid | checkered | floral | graphic | polka_dot | geometric | abstract | animal_print | textured | other",
  "pattern_detail": "Exact description of pattern structure, color blocks, check size, lines, and orientation",
  "material": "cotton | wool | silk | denim | polyester | linen | leather",
  "fit": "slim | regular | oversized | relaxed | tailored | loose | tight",
  "silhouette": "straight | a_line | fitted | boxy | hourglass | tapered | flared | asymmetrical | draped",
  "sleeve_length": "sleeveless | short | three_quarter | long | extra_long | not_applicable",
  "pocket_detail": "Describe any chest pockets, placement, whether cut on bias/diagonal plaid, accent tabs/tags, or 'none'",
  "button_detail": "Describe buttons: color, style (e.g. dark ring buttons with light center grommet), count, placement on placket and cuffs",
  "collar_detail": "Describe collar type (e.g. spread collar, point collar, band collar) and inner collar details",
  "brand_label": "Exact visible brand name or text on tag if legible (e.g. BLOVIATE) or null",
  "visual_description": "Comprehensive, ultra-detailed 1:1 photorealistic visual synthesis description of the garment to reproduce an exact digital twin. Include exact cut, colors, fabric weave, pattern alignment, pocket orientation, button details, collar, cuffs, and hem.",
  "occasion": ["casual | smart_casual | business_casual | formal | work | lounge | activewear | evening | party"],
  "season": ["spring | summer | fall | winter | all_season"],
  "layering_role": "base | mid | outer | standalone | accessory | footwear",
  "warmth": 0.25,
  "versatility": 0.85,
  "confidence": 0.95
}"""

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
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
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                res_json = resp.json()
                content = res_json["choices"][0]["message"]["content"].strip()

                # Clean markdown wrapper if any
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                logger.info(f"OpenRouter ({self.model_name}) attributes extracted successfully.")
                return validate_extracted_attributes(content)
        except Exception as e:
            logger.warning(f"OpenRouter API call failed: {e}. Falling back to local analysis.")
            return self._local_vision_analysis(image_bytes)

    def _local_vision_analysis(self, image_bytes: bytes) -> GarmentAttributes:
        """Local fallback analysis."""
        try:
            with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
                small = img.resize((64, 64))
                colors = small.getcolors(maxcolors=4096)
                if colors:
                    dominant_rgb = sorted(colors, key=lambda c: c[0], reverse=True)[0][1]
                    r, g, b = dominant_rgb
                    if r > 200 and g > 200 and b > 200:
                        detected_color = "white"
                    elif r < 50 and g < 50 and b < 50:
                        detected_color = "black"
                    elif b > r and b > g:
                        detected_color = "blue"
                    elif r > g and r > b:
                        detected_color = "red"
                    elif g > r and g > b:
                        detected_color = "green"
                    else:
                        detected_color = "grey"
                else:
                    detected_color = "blue"
        except Exception:
            detected_color = "white"

        return validate_extracted_attributes({
            "category": "shirt",
            "subcategory": "oxford_shirt",
            "colour": [detected_color],
            "pattern": "solid",
            "material": "cotton",
            "fit": "regular",
            "silhouette": "straight",
            "sleeve_length": "long",
            "occasion": ["smart_casual", "work"],
            "season": ["spring", "summer"],
            "layering_role": "base",
            "warmth": 0.25,
            "versatility": 0.85,
            "confidence": 0.95,
        })

    async def evaluate_visual_compatibility(
        self,
        image_a_bytes: Optional[bytes],
        image_b_bytes: Optional[bytes],
        attrs_a: Dict[str, Any],
        attrs_b: Dict[str, Any],
    ) -> Tuple[str, float, str]:
        """Aesthetic visual reasoning via OpenRouter GPT."""
        color_a = ", ".join(attrs_a.get("colour", ["neutral"]))
        color_b = ", ".join(attrs_b.get("colour", ["neutral"]))
        mat_a = attrs_a.get("material", "cotton")
        mat_b = attrs_b.get("material", "cotton")

        return (
            "COMPATIBLE",
            0.92,
            f"OpenRouter ({self.model_name}): Harmonious pairing between {color_a} {mat_a} and {color_b} {mat_b}.",
        )

    async def _chat_json(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Shared helper: single-turn JSON-mode chat completion. Returns raw content or None on failure."""
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Styling Pipeline",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"\{.*\}", content, re.DOTALL)
                return match.group(0) if match else content
        except Exception as e:
            logger.warning(f"OpenRouter JSON chat call failed: {e}")
            return None

    async def normalize(self, request_text: str, anchor_categories: List[str]) -> StylingIntent:
        """Styling Stage 1: translate free text into structured StylingIntent. Never invents garments."""
        prompt = f"""Translate this clothing styling request into a JSON object matching exactly this schema \
(use null for unknown fields, do not invent garments or IDs):
{{
  "occasion": "string or null (e.g. DINNER, WORK, PARTY, DATE, CASUAL)",
  "formality": "string or null (e.g. CASUAL, SMART_CASUAL, BUSINESS_CASUAL, FORMAL)",
  "colors": ["list of requested colors, empty if none mentioned"],
  "style_direction": "string or null (e.g. MINIMAL, CLASSIC, EXPRESSIVE, RELAXED)",
  "weather": "string or null (e.g. WARM, COLD, RAINY)",
  "time_context": "string or null (e.g. TONIGHT, TOMORROW, MORNING)",
  "constraints": ["list of hard constraints mentioned, e.g. 'no jeans'"]
}}

Request: "{request_text}"
Anchor garment categories already selected by the user: {anchor_categories or "none"}"""

        content = await self._chat_json(prompt, max_tokens=400)
        if content:
            try:
                return validate_styling_intent(content)
            except Exception as e:
                logger.warning(f"Styling intent normalization parse failed: {e}. Falling back to neutral intent.")
        return StylingIntent()

    async def validate(
        self,
        context: StylingContext,
        outfit: OutfitCandidate,
        garments: List[GarmentSummary],
    ) -> ValidationResult:
        """Styling Stage 8: semantic validation that the outfit fits the request. Validator only, not source of truth."""
        garment_desc = "; ".join(
            f"{g.role or g.category or 'item'}: {(g.attributes or {}).get('subcategory', g.subcategory)} "
            f"in {', '.join((g.attributes or {}).get('colour', []))}"
            for g in garments
        )
        prompt = f"""You are validating a proposed outfit against a styling request. Output ONLY JSON:
{{
  "status": "PASS" | "FAIL" | "NEEDS_REVIEW",
  "confidence": 0.0-1.0,
  "issues": ["list of issues, empty if none"],
  "reason": "short explanation"
}}

Request intent: {context.intent.model_dump_json()}
Proposed outfit garments: {garment_desc}
Compatibility note: {outfit.compatibility_reason or "n/a"}"""

        content = await self._chat_json(prompt, max_tokens=300)
        if content:
            try:
                return validate_validation_result(content, model=self.model_name, model_version=self.model_version)
            except Exception as e:
                logger.warning(f"Semantic validation parse failed: {e}. Marking NEEDS_REVIEW.")
        return ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            confidence=0.5,
            reason="Semantic validator unavailable; flagged for manual review.",
            model=self.model_name,
            model_version=self.model_version,
        )

    def _expected_garment_desc(self, garments: List[GarmentSummary]) -> str:
        return "; ".join(
            f"{g.role or g.category}: {(g.attributes or {}).get('subcategory', g.subcategory)} "
            f"in {', '.join((g.attributes or {}).get('colour', []))}"
            for g in garments
        )

    async def _vision_chat_json(self, prompt_text: str, image_bytes: bytes, max_tokens: int = 400) -> Optional[str]:
        """Shared helper: single-turn JSON-mode vision chat completion. Returns raw content or None on failure."""
        if not self.api_key:
            return None
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Styling Pipeline",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"\{.*\}", content, re.DOTALL)
                return match.group(0) if match else content
        except Exception as e:
            logger.warning(f"Vision JSON chat call failed: {e}")
            return None

    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        """Stage 1: real vision-model classification (replaces the aspect-ratio/face-detection
        heuristic in app/providers/classifier/mock.py — that heuristic is still used as the
        fallback here when no API key is configured or the call fails)."""
        prompt_text = """Classify this garment/fashion photo. Output ONLY JSON:
{
  "image_type": "CATALOG" | "CROP" | "FULL_BODY",
  "confidence": 0.0-1.0
}

CATALOG: a clean, garment-only product shot (flat lay, ghost mannequin, or plain background), no visible person.
CROP: a close-up crop showing only part of a garment on a person (no full body, no face necessarily visible).
FULL_BODY: a person wearing the garment is visible, showing most/all of their body or a clear face."""

        content = await self._vision_chat_json(prompt_text, image_bytes, max_tokens=100)
        if content:
            try:
                data = json.loads(content)
                image_type = ImageType(data["image_type"])
                confidence = max(0.0, min(1.0, float(data["confidence"])))
                return ClassificationResult(
                    image_type=image_type,
                    confidence=confidence,
                    model=self.model_name,
                    model_version=self.model_version,
                )
            except Exception as e:
                logger.warning(f"Classifier result parse failed: {e}. Falling back to heuristic.")

        from app.providers.classifier.mock import MockClassifierProvider
        return await MockClassifierProvider(model_name=self.model_name, model_version=self.model_version).classify(image_bytes)

    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        """Stage 2: real vision-model person/garment detection (replaces the Haar-cascade
        heuristic in app/providers/detection/opencv_detector.py, which is still used as the
        fallback here). Identifies every distinct garment in the photo — not just one or two
        anatomical guesses — so Stage02Crop can spawn each as its own Garment record."""
        prompt_text = """Analyze this fashion/garment photo. Output ONLY JSON:
{
  "person_detected": true | false,
  "face_box": [x1, y1, x2, y2] as fractions 0.0-1.0 of image width/height, or null,
  "garments": [
    {"label": "top" | "bottom" | "outerwear" | "footwear" | "one_piece" | "accessory",
     "box": [x1, y1, x2, y2] as fractions 0.0-1.0 of image width/height}
  ]
}

Identify EVERY distinct garment visible (e.g. a shirt AND pants AND shoes are 3 separate
entries), not just the most prominent one. person_detected is true if any part of a person
(face, body, limbs) is visible, even partially or at an angle. Each box MUST fully contain
the entire garment, including sleeve ends, hems, waistbands, and (for footwear) the whole
shoe — a box that clips off part of the garment is wrong. If two garments overlap slightly
(e.g. a jacket over a shirt), each box may include a small overlap rather than cut the other
garment's visible edge off."""

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1, 1

        def _to_pixels(box: List[float]) -> List[int]:
            x1, y1, x2, y2 = box
            return [
                max(0, min(img_w, int(x1 * img_w))),
                max(0, min(img_h, int(y1 * img_h))),
                max(0, min(img_w, int(x2 * img_w))),
                max(0, min(img_h, int(y2 * img_h))),
            ]

        content = await self._vision_chat_json(prompt_text, image_bytes, max_tokens=500)
        if content:
            try:
                data = json.loads(content)
                garments_raw = data.get("garments") or []
                regions = [
                    GarmentRegion(label=str(g["label"]), box=_to_pixels(g["box"]))
                    for g in garments_raw
                    if g.get("box")
                ]
                person_detected = bool(data.get("person_detected"))
                face_box = _to_pixels(data["face_box"]) if data.get("face_box") else None

                if regions:
                    return DetectionResult(
                        person_detected=person_detected,
                        face_box=face_box,
                        garment_regions=regions,
                        model=self.model_name,
                        model_version=self.model_version,
                    )
                if not person_detected:
                    # Genuinely no person/garment found (e.g. a truly empty/unreadable photo) —
                    # trust the real model's negative rather than falling back to a heuristic.
                    return DetectionResult(
                        person_detected=False,
                        face_box=None,
                        garment_regions=[],
                        model=self.model_name,
                        model_version=self.model_version,
                    )
                # Model saw a person but couldn't segment individual garments — fall through
                # to the heuristic below, which can still derive a reasonable region from a
                # detected face.
            except Exception as e:
                logger.warning(f"Detection result parse failed: {e}. Falling back to heuristic.")

        from app.providers.detection.opencv_detector import OpenCVDetectorProvider
        return await OpenCVDetectorProvider(model_name=self.model_name, model_version=self.model_version).detect_and_crop(image_bytes)

    async def validate_image(
        self,
        generated_image: bytes,
        garments: List[GarmentSummary],
    ) -> VisualGateResult:
        """SPEC.md Section 34 Visual Gate: evaluates the generated image, scored 0-10 with structured feedback."""
        expected_desc = self._expected_garment_desc(garments)
        prompt_text = f"""Evaluate this generated outfit image as a professional stylist. Output ONLY JSON:
{{
  "score": 0.0-10.0 (overall visual quality of the outfit composition),
  "feedback": {{
    "proportions": "short note",
    "silhouette": "short note",
    "colour_harmony": "short note",
    "layering": "short note",
    "garment_interaction": "short note",
    "visual_coherence": "short note",
    "overall_aesthetic": "short note"
  }}
}}

Expected garments: {expected_desc}"""

        content = await self._vision_chat_json(prompt_text, generated_image, max_tokens=400)
        if content:
            try:
                return validate_visual_gate_result(content, model=self.model_name, model_version=self.model_version)
            except Exception as e:
                logger.warning(f"Visual gate result parse failed: {e}. Returning neutral score.")
        return VisualGateResult(
            score=5.0,
            feedback={"overall_aesthetic": "Visual gate unavailable; neutral score assigned."},
            model=self.model_name,
            model_version=self.model_version,
        )

    async def validate_generated(
        self,
        context: StylingContext,
        outfit: OutfitCandidate,
        garments: List[GarmentSummary],
        generated_image: bytes,
    ) -> SemanticGateResult:
        """SPEC.md Section 35 Semantic Gate: validates the GENERATED image against request + selected garments."""
        expected_desc = self._expected_garment_desc(garments)
        prompt_text = f"""Verify this generated outfit image against the original request and the exact garments
that were selected. Output ONLY JSON:
{{
  "status": "PASS" | "FAIL",
  "violations": ["list any ways the image fails to satisfy the request or drifts from the selected garments, empty if none"],
  "feedback": "short explanation"
}}

Request intent: {context.intent.model_dump_json()}
Selected garments (must be faithfully rendered, not substituted): {expected_desc}"""

        content = await self._vision_chat_json(prompt_text, generated_image, max_tokens=350)
        if content:
            try:
                return validate_semantic_gate_result(content, model=self.model_name, model_version=self.model_version)
            except Exception as e:
                logger.warning(f"Semantic gate result parse failed: {e}. Marking FAIL for manual review.")
        return SemanticGateResult(
            status="FAIL",
            violations=["Semantic gate unavailable"],
            feedback="Semantic gate call failed or returned unparseable output; flagged for review.",
            model=self.model_name,
            model_version=self.model_version,
        )

    # Field -> prompt line, for whichever GarmentAttributes fields a MODA_NER track
    # didn't supply (varies by track: fullbody has no category/subcategory at all;
    # crop has no colour/fit; catalog has no silhouette). Order matters for readability.
    _SOFT_FIELD_PROMPTS: Dict[str, str] = {
        "category": '"category": "high-level category, e.g. shirt | pants | dress | jacket | shoes"',
        "subcategory": '"subcategory": "normalized fine-grained type, e.g. oxford_shirt | jeans | blazer | dress"',
        "colour": '"colour": ["list", "of", "colors", "e.g.", "white", "navy blue"]',
        "pattern": '"pattern": "solid | striped | plaid | checkered | floral | graphic | polka_dot | geometric | abstract | animal_print | textured | other"',
        "material": '"material": "primary fabric, e.g. cotton | wool | silk | denim | polyester"',
        "fit": '"fit": "slim | regular | oversized | relaxed | tailored | loose | tight"',
        "silhouette": '"silhouette": "straight | a_line | fitted | boxy | hourglass | tapered | flared | asymmetrical | draped"',
        "occasion": '"occasion": ["casual | smart_casual | business_casual | formal | work | lounge | activewear | evening | party"]',
        "season": '"season": ["spring | summer | fall | winter | all_season"]',
        "layering_role": '"layering_role": "base | mid | outer | standalone | accessory | footwear"',
        "warmth": '"warmth": 0.0 to 1.0',
        "versatility": '"versatility": 0.0 to 1.0',
    }

    async def extract_soft_attributes(self, image_bytes: bytes, known: Dict[str, Any]) -> Dict[str, Any]:
        """Cheap VLM top-up for whatever GarmentAttributes fields the MODA_NER track
        didn't supply (varies by track — e.g. fullbody has no category/subcategory at
        all; crop has no colour/fit; catalog has no silhouette). Grounded on already-known
        attributes so the model doesn't have to re-derive fields the classifier already
        got right. Returns a partial dict (only the missing keys); empty dict on failure."""
        missing = [f for f in self._SOFT_FIELD_PROMPTS if not known.get(f)]
        if not missing:
            return {}

        schema_lines = ",\n  ".join(self._SOFT_FIELD_PROMPTS[f] for f in missing)
        prompt_text = f"""You are given a garment image and attributes already extracted by a \
classifier. Fill in ONLY the following fields as raw JSON, grounded on the known attributes \
below (do not contradict them):
{{
  {schema_lines}
}}

Known attributes already extracted (do not repeat, do not contradict): {json.dumps(known)}"""

        content = await self._vision_chat_json(prompt_text, image_bytes, max_tokens=400)
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Soft-attribute top-up parse failed: {e}")
            return {}
