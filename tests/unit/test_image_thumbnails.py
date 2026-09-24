"""Thumbnail serving must never block the API's event loop and must be built once per asset.

Background: production's catalogue page requests every garment (`?limit=3000`) with a
`?thumb=320` image each. The original endpoint decoded a 1-1.6MB studio image and ran a
LANCZOS resize synchronously inside the async handler, on every request, for every visitor.
On the 1-2 vCPU droplet that starved every other request — the constant-string `/health`
route was observed hanging 20s+ while a catalogue loaded. These tests pin the two
properties that fix it: a pure, thread-safe build function, and a content-addressed cache key
so the second request for the same asset never rebuilds.
"""

import io

from PIL import Image

from app.api.v1.images import (
    THUMB_MAX_EDGE,
    THUMB_MIN_EDGE,
    build_thumbnail,
    thumbnail_object_key,
)


def _png(width: int, height: int, mode: str = "RGBA") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (width, height), (200, 40, 40, 255) if mode == "RGBA" else (200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_thumbnail_fits_within_edge_and_keeps_aspect():
    out = build_thumbnail(_png(1400, 1000), 320)
    with Image.open(io.BytesIO(out)) as img:
        assert img.format == "JPEG"
        assert max(img.size) == 320
        # 1400x1000 -> 320x229 (PIL rounds 228.6 up; aspect preserved, not squashed to a square)
        assert img.size == (320, 229)


def test_build_thumbnail_flattens_alpha_to_rgb_jpeg():
    # Studio digitisations are RGBA PNGs on transparent backgrounds; JPEG has no alpha,
    # so the build must convert rather than raise.
    out = build_thumbnail(_png(600, 600, "RGBA"), 100)
    with Image.open(io.BytesIO(out)) as img:
        assert img.mode == "RGB"
        assert img.size == (100, 100)


def test_build_thumbnail_never_upscales():
    out = build_thumbnail(_png(120, 80, "RGB"), 320)
    with Image.open(io.BytesIO(out)) as img:
        assert img.size == (120, 80)


def test_thumbnail_key_is_content_addressed_per_edge():
    sha = "ab" * 32
    assert thumbnail_object_key(sha, 320) == f"thumbs/{sha}_320.jpg"
    # Same bytes at a different size is a different cached object; different bytes never collide.
    assert thumbnail_object_key(sha, 320) != thumbnail_object_key(sha, 640)
    assert thumbnail_object_key(sha, 320) != thumbnail_object_key("cd" * 32, 320)


def test_edge_bounds_are_sane():
    assert 0 < THUMB_MIN_EDGE < 320 <= THUMB_MAX_EDGE
