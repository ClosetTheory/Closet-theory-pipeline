"""Mock / Heuristic Classifier Provider for MobileNetV3."""

import io
from PIL import Image
from app.config import settings
from app.providers.base import BaseClassifierProvider
from app.schemas.pipeline import ClassificationResult, ImageType


class MockClassifierProvider(BaseClassifierProvider):
    """
    Deterministic heuristic classifier:
    - If aspect ratio (height/width) > 1.6 -> FULL_BODY
    - If aspect ratio between 1.1 and 1.6 -> CROP
    - If square or wide (<= 1.1) -> CATALOG
    """

    def __init__(
        self,
        model_name: str = settings.CLASSIFIER_MODEL_NAME,
        model_version: str = settings.CLASSIFIER_MODEL_VERSION,
        forced_type: ImageType | None = None,
        confidence: float = 0.95,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.forced_type = forced_type
        self.confidence = confidence

    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        if self.forced_type:
            return ClassificationResult(
                image_type=self.forced_type,
                confidence=self.confidence,
                model=self.model_name,
                model_version=self.model_version,
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                ratio = h / max(w, 1)

                if ratio >= 1.7:
                    img_type = ImageType.FULL_BODY
                    conf = 0.96
                elif ratio >= 1.2:
                    img_type = ImageType.CROP
                    conf = 0.92
                else:
                    img_type = ImageType.CATALOG
                    conf = 0.98
        except Exception:
            img_type = ImageType.CATALOG
            conf = self.confidence

        return ClassificationResult(
            image_type=img_type,
            confidence=conf,
            model=self.model_name,
            model_version=self.model_version,
        )
