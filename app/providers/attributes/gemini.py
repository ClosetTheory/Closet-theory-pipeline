"""Gemini Attribute Extractor Provider (routed through OpenRouter).

Used as Stage 3's cross-model retry attempt: when GPT-4o's extraction fails independent
verification, re-asking GPT-4o again tends to reproduce the same mistake (same model, same
blind spot, same image). Routing the retry through a genuinely different vendor gives a real
second opinion instead of an echo. Goes through OpenRouter (like every other real vision call
in this codebase) rather than the direct Google Generative Language API, since only
OPENROUTER_API_KEY is actually configured in this environment.
"""

import base64
import json
import re
from typing import Optional
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.schemas.attributes import AttributeValidationError, KNOWN_SUBCATEGORIES, GarmentAttributes, validate_extracted_attributes


class GeminiAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """Gemini (via OpenRouter) provider extracting structured attributes."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.VISION_VERIFIER_MODEL,
        model_version: str = "v1",
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = MockAttributeExtractorProvider(model_name=model_name, model_version=model_version)
        self._last_prompt: str = ""  # for frontend visibility of which prompt actually produced the result

    async def extract_attributes(
        self, image_bytes: bytes, image_type: Optional[str] = None, garment_label: Optional[str] = None
    ) -> GarmentAttributes:
        if not self.api_key:
            return await self._fallback.extract_attributes(image_bytes)

        focus_preamble = (
            f"This photo shows a person wearing multiple garments. Focus ONLY on the {garment_label} "
            f"(ignore all other garments/accessories) when answering below.\n\n"
            if garment_label else ""
        )
        subcategory_list = " | ".join(sorted(KNOWN_SUBCATEGORIES))

        prompt = focus_preamble + f"""Analyze this garment image and return ONLY a raw JSON object (no markdown) with these exact keys:
- category: string
- subcategory: EXACTLY one of these values, choosing the closest match — never invent a new one: {subcategory_list}
- garment_class: canonical class, e.g. T_SHIRT | SHIRT | JEANS | TROUSERS | DRESS | BLAZER | SNEAKERS | SAREE | KURTA (use "<CATEGORY>_OTHER" if uncertain)
- colour: list of strings (e.g. ["white", "blue"])
- pattern: solid | striped | plaid | checkered | floral | graphic | polka_dot | geometric | abstract | animal_print | textured | other
- material: string (e.g. cotton, wool, silk, denim, polyester)
- fit: slim | regular | oversized | relaxed | tailored | loose | tight | not_applicable
- silhouette: straight | a_line | fitted | boxy | hourglass | tapered | flared | asymmetrical | draped | not_applicable
- sleeve_length: sleeveless | short | three_quarter | long | extra_long | not_applicable
- occasion: list of casual | smart_casual | business_casual | formal | work | lounge | activewear | evening | party
- season: list of spring | summer | fall | winter | all_season
- layering_role: base | mid | outer | standalone | accessory | footwear
- gender: women | men | unisex — classify by the garment's actual cut/styling, not by assuming from context
- warmth: float 0.0 to 1.0
- versatility: float 0.0 to 1.0
- confidence: float 0.0 to 1.0"""
        self._last_prompt = prompt

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
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
                return validate_extracted_attributes(content)
        except AttributeValidationError:
            # Genuinely unmappable output (e.g. subcategory outside the known taxonomy) must
            # route to human review, not be silently swapped for a fake mock result.
            raise
        except Exception as e:
            logger.warning(f"Gemini ({self.model_name}) attribute extraction failed: {e}. Falling back to mock.")
            return await self._fallback.extract_attributes(image_bytes)
