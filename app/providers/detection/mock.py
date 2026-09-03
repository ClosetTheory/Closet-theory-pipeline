"""Mock / Heuristic Provider for RetinaFace + SAM Detection & Crop."""

import io
from PIL import Image
from app.config import settings
from app.providers.base import BaseDetectionProvider
from app.schemas.pipeline import DetectionResult, GarmentRegion


class MockDetectionProvider(BaseDetectionProvider):
    """
    Heuristic detection provider that extracts garment regions based on image proportions.
    Provides pixel-level bounding boxes and person/face detection.
    """

    def __init__(
        self,
        model_name: str = settings.DETECTION_MODEL_NAME,
        model_version: str = settings.DETECTION_MODEL_VERSION,
        person_detected: bool = True,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.person_detected = person_detected

    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
        except Exception:
            w, h = 800, 1200

        if not self.person_detected:
            # Catalog or flat-lay: garment occupies center region
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            regions = [
                GarmentRegion(
                    label="upper_body",
                    box=[pad_x, pad_y, w - pad_x, h - pad_y],
                    mask_ref=None,
                )
            ]
            return DetectionResult(
                person_detected=False,
                face_box=None,
                garment_regions=regions,
                model=self.model_name,
                model_version=self.model_version,
            )

        # Person detected: face at top (10% to 25% height)
        face_x1 = int(w * 0.35)
        face_y1 = int(h * 0.05)
        face_x2 = int(w * 0.65)
        face_y2 = int(h * 0.25)
        face_box = [face_x1, face_y1, face_x2, face_y2]

        # Upper body region: 25% to 65% height
        upper_box = [int(w * 0.15), int(h * 0.22), int(w * 0.85), int(h * 0.65)]

        # Lower body region: 60% to 95% height
        lower_box = [int(w * 0.20), int(h * 0.60), int(w * 0.80), int(h * 0.95)]

        regions = [
            GarmentRegion(
                label="upper_body",
                box=upper_box,
                mask_ref=None,
            ),
            GarmentRegion(
                label="lower_body",
                box=lower_box,
                mask_ref=None,
            ),
        ]

        return DetectionResult(
            person_detected=True,
            face_box=face_box,
            garment_regions=regions,
            model=self.model_name,
            model_version=self.model_version,
        )
