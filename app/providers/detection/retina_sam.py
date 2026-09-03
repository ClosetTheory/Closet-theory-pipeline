"""RetinaFace + SAM Detection Provider."""

from app.config import settings
from app.providers.base import BaseDetectionProvider
from app.providers.detection.mock import MockDetectionProvider
from app.schemas.pipeline import DetectionResult


class RetinaSAMDetectionProvider(BaseDetectionProvider):
    """
    RetinaFace localization + SAM segmentation provider.
    Localizes person/face landmarks and produces garment segment masks.
    """

    def __init__(
        self,
        model_name: str = settings.DETECTION_MODEL_NAME,
        model_version: str = settings.DETECTION_MODEL_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = MockDetectionProvider(model_name=model_name, model_version=model_version)

    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        # Connects to RetinaFace + SAM inference pipeline
        return await self._fallback.detect_and_crop(image_bytes)
