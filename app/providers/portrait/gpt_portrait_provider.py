"""Portrait generation via OpenRouter's image endpoint.

Same call shape as GPTOutfitImageProvider, minus `input_references` — there is nothing to
condition on, because a character has no photographs. Everything the image needs comes from the
prompt (see prompt.py).
"""

import base64
from typing import Optional
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BasePortraitProvider


class GPTPortraitProvider(BasePortraitProvider):
    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        model_version: str = "v1",
    ):
        self.api_key = api_key
        # PORTRAIT_MODEL exists so portraits can be pinned to a different model from outfits
        # without touching styling; blank means "whatever outfits use".
        self.model_name = model_name or settings.PORTRAIT_MODEL or settings.OPENROUTER_IMAGE_MODEL
        self.model_version = model_version

    async def generate(self, prompt: str) -> Optional[bytes]:
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Styling Pipeline",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/images",
                    headers=headers,
                    json={"model": self.model_name, "prompt": prompt},
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"Portrait model {self.model_name} returned HTTP {resp.status_code}: "
                        f"{resp.text[:150]}"
                    )
                    return None
                data = resp.json()
                items = data.get("data") or []
                if not items:
                    logger.warning(f"Portrait model {self.model_name} returned no image data")
                    return None
                item = items[0]
                if "b64_json" in item:
                    return base64.b64decode(item["b64_json"])
                if "url" in item:
                    img_resp = await client.get(item["url"])
                    if img_resp.status_code == 200:
                        return img_resp.content
                return None
        except Exception as e:
            logger.warning(f"Portrait generation failed: {e}")
            return None
