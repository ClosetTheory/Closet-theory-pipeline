"""Mock Outfit Image Provider: composites garment canonical images side-by-side via PIL."""

import io
from typing import List, Optional
from PIL import Image
from app.providers.base import BaseOutfitImageProvider
from app.schemas.styling import GarmentSummary


class MockOutfitImageProvider(BaseOutfitImageProvider):
    def __init__(self, model_name: str = "mock-outfit-imaging", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def generate(self, garments: List[GarmentSummary], canonical_images: List[bytes]) -> Optional[bytes]:
        if not canonical_images:
            return None
        try:
            tiles = [Image.open(io.BytesIO(b)).convert("RGB") for b in canonical_images]
            tile_h = 512
            resized = []
            for t in tiles:
                ratio = tile_h / t.height
                resized.append(t.resize((max(1, int(t.width * ratio)), tile_h)))
            total_w = sum(t.width for t in resized) + 16 * (len(resized) - 1)
            canvas = Image.new("RGB", (total_w, tile_h), color=(245, 245, 245))
            x = 0
            for t in resized:
                canvas.paste(t, (x, 0))
                x += t.width + 16
            buf = io.BytesIO()
            canvas.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception:
            return None
