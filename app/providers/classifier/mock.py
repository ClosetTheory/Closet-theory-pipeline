"""Mock / Heuristic Classifier Provider for MobileNetV3."""

import io
from PIL import Image
from app.config import settings
from app.providers.base import BaseClassifierProvider
from app.providers.detection.opencv_detector import OpenCVDetectorProvider
from app.schemas.pipeline import ClassificationResult, ImageType


class MockClassifierProvider(BaseClassifierProvider):
    """
    Heuristic classifier — no trained model is wired up here (see mobilenet.py, which
    delegates to this same class). Aspect ratio alone is a poor signal: most real photos
    (portrait phone shots, standard 4:3/3:2 crops) land in a "tall-ish" band regardless of
    whether they actually show a person, so a pure ratio threshold misclassifies the large
    majority of uploads as CROP. Instead, this uses a local, deterministic, zero-cost face
    check (OpenCVDetectorProvider — not the globally-configured detection provider, which may
    be a real paid API call; a class named "Mock" must stay safe to call with zero network
    dependency regardless of DETECTION_PROVIDER) as the primary signal for FULL_BODY — a photo
    with a detected face is a full-body/on-model shot regardless of its exact aspect ratio.
    Aspect ratio is only used as a fallback to distinguish CATALOG vs CROP among the
    remaining (person-less) images.
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
            detection = await OpenCVDetectorProvider().detect_and_crop(image_bytes)
        except Exception:
            detection = None

        if detection and (detection.person_detected or detection.face_box):
            return ClassificationResult(
                image_type=ImageType.FULL_BODY,
                confidence=0.96,
                model=self.model_name,
                model_version=self.model_version,
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                ratio = h / max(w, 1)
                # No person/face detected (or detection itself failed) — fall back to aspect
                # ratio so a genuinely tall full-body photo where detection missed the face
                # (turned away, obscured, low-res) still isn't lost entirely to CROP/CATALOG.
                if ratio >= 1.7:
                    img_type = ImageType.FULL_BODY
                    conf = 0.85
                elif ratio >= 1.3:
                    img_type = ImageType.CROP
                    conf = 0.9
                else:
                    img_type = ImageType.CATALOG
                    conf = 0.95
        except Exception:
            img_type = ImageType.CATALOG
            conf = self.confidence

        return ClassificationResult(
            image_type=img_type,
            confidence=conf,
            model=self.model_name,
            model_version=self.model_version,
        )
