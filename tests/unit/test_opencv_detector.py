"""Unit tests for OpenCV face detector and garment cropper."""

import io
import pytest
from PIL import Image, ImageDraw
from app.providers.detection.opencv_detector import OpenCVDetectorProvider


@pytest.mark.asyncio
async def test_opencv_detector_runs_and_crops():
    # Create test image with simulated face circle and body rectangle
    img = Image.new("RGB", (600, 800), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Face
    draw.ellipse([250, 100, 350, 200], fill=(220, 180, 160))
    # Eyes
    draw.point((280, 140), fill=(0, 0, 0))
    draw.point((320, 140), fill=(0, 0, 0))
    # Torso shirt
    draw.rectangle([150, 220, 450, 650], fill=(30, 80, 180))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    provider = OpenCVDetectorProvider()
    res = await provider.detect_and_crop(raw_bytes)

    assert res.garment_regions is not None
    assert len(res.garment_regions) >= 1
    box = res.garment_regions[0].box
    assert len(box) == 4
    assert box[2] > box[0]
    assert box[3] > box[1]
