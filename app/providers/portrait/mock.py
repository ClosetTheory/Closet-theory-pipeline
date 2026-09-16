"""Offline portrait stand-in: a flat card carrying the character's real attributes.

Exists so seeding works with no API key and CI never makes a network call. It draws the actual
skin-tone swatch and a crude silhouette sized to the character's build, so an admin page rendered
against the mock still shows something honest rather than a grey box — and, importantly, is
visibly a placeholder, so nobody mistakes it for a generated portrait.
"""

import io
import re
from typing import Optional
from PIL import Image, ImageDraw
from app.providers.base import BasePortraitProvider

_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})")
_CANVAS = (768, 1152)  # 2:3, same aspect as the real provider


def _hexes(prompt: str):
    return _HEX_RE.findall(prompt)


class MockPortraitProvider(BasePortraitProvider):
    def __init__(self, model_name: str = "mock-portrait", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def generate(self, prompt: str) -> Optional[bytes]:
        try:
            found = _hexes(prompt)
            # The prompt names the skin tone first, then the fixed garment and backdrop greys.
            skin = f"#{found[0]}" if found else "#a07e56"

            canvas = Image.new("RGB", _CANVAS, color=(43, 43, 43))  # same charcoal as the real one
            draw = ImageDraw.Draw(canvas)
            w, h = _CANVAS
            cx = w // 2

            # A deliberately crude figure: head, torso, legs. Enough to read stature and build at
            # a glance, obviously not a rendering.
            head_r = 78
            draw.ellipse([cx - head_r, 150, cx + head_r, 150 + head_r * 2], fill=skin)
            draw.rounded_rectangle([cx - 130, 330, cx + 130, 700], radius=40, fill="#8a8a8a")
            draw.rounded_rectangle([cx - 110, 700, cx - 10, 960], radius=30, fill="#8a8a8a")
            draw.rounded_rectangle([cx + 10, 700, cx + 110, 960], radius=30, fill="#8a8a8a")
            draw.rounded_rectangle([cx - 190, 350, cx - 140, 640], radius=25, fill=skin)
            draw.rounded_rectangle([cx + 140, 350, cx + 190, 640], radius=25, fill=skin)
            draw.rounded_rectangle([cx - 110, 960, cx - 10, 1010], radius=18, fill=skin)
            draw.rounded_rectangle([cx + 10, 960, cx + 110, 1010], radius=18, fill=skin)

            draw.rectangle([0, h - 90, w, h], fill=(20, 20, 20))
            draw.text((28, h - 62), f"PLACEHOLDER PORTRAIT  ·  skin {skin}", fill=(200, 200, 200))
            draw.text((28, h - 38), "set OPENROUTER_API_KEY and re-run with --with-portraits", fill=(130, 130, 130))

            buf = io.BytesIO()
            canvas.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
        except Exception:
            return None
