"""Gemini Flash Attribute Extractor Provider."""

import base64
import json
from typing import Optional
import httpx
from app.config import settings
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.schemas.attributes import GarmentAttributes, validate_extracted_attributes


class GeminiAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """Gemini Flash provider extracting structured attributes with JSON schema enforcement."""

    def __init__(
        self,
        api_key: Optional[str] = settings.GEMINI_API_KEY,
        model_name: str = "gemini-flash",
        model_version: str = "v1",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = MockAttributeExtractorProvider(model_name=model_name, model_version=model_version)

    async def extract_attributes(
        self, image_bytes: bytes, image_type: Optional[str] = None, garment_label: Optional[str] = None
    ) -> GarmentAttributes:
        if not self.api_key:
            return await self._fallback.extract_attributes(image_bytes)

        # Build multimodal prompt
        prompt = """Analyze this garment image and return a strict JSON object with these exact keys:
- category: string
- subcategory: string (e.g. oxford_shirt, jeans, blazer, dress)
- garment_class: canonical class, e.g. T_SHIRT | SHIRT | JEANS | TROUSERS | DRESS | BLAZER | SNEAKERS | SAREE | KURTA (see full controlled vocabulary; use "<CATEGORY>_OTHER" if uncertain)
- colour: list of strings (e.g. ["white", "blue"])
- pattern: solid | striped | plaid | checkered | floral | graphic | polka_dot | geometric | abstract | animal_print | textured | other
- material: string (e.g. cotton, wool, silk, denim, polyester)
- fit: slim | regular | oversized | relaxed | tailored | loose | tight
- silhouette: straight | a_line | fitted | boxy | hourglass | tapered | flared | asymmetrical | draped
- sleeve_length: sleeveless | short | three_quarter | long | extra_long | not_applicable
- occasion: list of casual | smart_casual | business_casual | formal | work | lounge | activewear | evening | party
- season: list of spring | summer | fall | winter | all_season
- layering_role: base | mid | outer | standalone | accessory | footwear
- warmth: float 0.0 to 1.0
- versatility: float 0.0 to 1.0
- confidence: float 0.0 to 1.0"""

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image,
                        }
                    },
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            res_data = resp.json()
            raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            # Validate through strict 7-step pipeline
            return validate_extracted_attributes(raw_text)
