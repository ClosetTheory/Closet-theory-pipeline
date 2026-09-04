"""Claude Sonnet Attribute Extractor Provider."""

import base64
import json
from typing import Optional
import httpx
from app.config import settings
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.schemas.attributes import GarmentAttributes, validate_extracted_attributes


class ClaudeAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """Claude Sonnet provider extracting structured attributes."""

    def __init__(
        self,
        api_key: Optional[str] = settings.ANTHROPIC_API_KEY,
        model_name: str = "claude-3-5-sonnet-20241022",
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

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        prompt = "Extract garment attributes as raw JSON matching the required schema."
        payload = {
            "model": self.model_name,
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            res_data = resp.json()
            raw_text = res_data["content"][0]["text"]
            return validate_extracted_attributes(raw_text)
