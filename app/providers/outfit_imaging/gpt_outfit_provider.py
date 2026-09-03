"""GPT-image-2 Composite Outfit Image Provider via OpenRouter.

Reuses the same /api/v1/images endpoint and input_references image-to-image
conditioning pattern as GPTStudioDigitisationProvider (single-garment digitisation),
but sends every selected garment's canonical image as a reference so the model
composes them into one outfit presentation shot.
"""

import base64
from typing import List, Optional
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BaseOutfitImageProvider
from app.schemas.styling import GarmentSummary


class GPTOutfitImageProvider(BaseOutfitImageProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.OPENROUTER_IMAGE_MODEL,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name
        self.model_version = "v1"

    def _build_prompt(self, garments: List[GarmentSummary]) -> str:
        pieces = []
        for g in garments:
            attrs = g.attributes or {}
            colors = ", ".join(attrs.get("colour", []))
            subcat = (attrs.get("subcategory") or g.subcategory or g.category or "item").replace("_", " ")
            pieces.append(f"{colors} {subcat}".strip())

        items_desc = "; ".join(pieces)
        return f"""Commercial e-commerce outfit photograph combining these exact garments, displayed together as a \
single coordinated outfit on a matte black full-body mannequin, on a solid dark charcoal studio backdrop: {items_desc}.

Requirements:
- Use each reference image's exact garment (color, pattern, silhouette) unchanged — do not substitute or invent a different garment.
- Display the full outfit layered correctly (top-to-bottom outfit order) on a single solid matte black mannequin form (faceless, headless or featureless head, smooth matte black finish) — not a flat-lay, not a ghost/invisible mannequin, not a real person.
- Studio lighting with subtle rim light outlining the mannequin and garments, no people, no faces, no text overlays, no watermark, 8k sharp product photography quality."""

    async def generate(self, garments: List[GarmentSummary], canonical_images: List[bytes]) -> Optional[bytes]:
        if not self.api_key or not canonical_images:
            return None

        url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/images"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Styling Pipeline",
            "Content-Type": "application/json",
        }
        prompt = self._build_prompt(garments)
        input_references = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(b).decode('utf-8')}"}}
            for b in canonical_images
        ]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"model": self.model_name, "prompt": prompt, "input_references": input_references},
                )
                if resp.status_code != 200:
                    logger.warning(f"Outfit image model {self.model_name} returned HTTP {resp.status_code}: {resp.text[:150]}")
                    return None
                data = resp.json()
                if "data" in data and data["data"]:
                    item = data["data"][0]
                    if "b64_json" in item:
                        return base64.b64decode(item["b64_json"])
                    if "url" in item:
                        img_resp = await client.get(item["url"])
                        if img_resp.status_code == 200:
                            return img_resp.content
        except Exception as e:
            logger.warning(f"Outfit image generation call failed: {e}")

        return None
