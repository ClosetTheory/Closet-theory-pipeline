"""Detection provider factory."""

from app.config import settings
from app.providers.base import BaseDetectionProvider
from app.providers.detection.mock import MockDetectionProvider
from app.providers.detection.opencv_detector import OpenCVDetectorProvider
from app.providers.detection.retina_sam import RetinaSAMDetectionProvider


def get_detection_provider() -> BaseDetectionProvider:
    provider_name = settings.DETECTION_PROVIDER.lower()
    if provider_name == "opencv":
        return OpenCVDetectorProvider(
            model_name=settings.DETECTION_MODEL_NAME,
            model_version=settings.DETECTION_MODEL_VERSION,
        )
    elif provider_name == "retina_sam":
        return RetinaSAMDetectionProvider(
            model_name=settings.DETECTION_MODEL_NAME,
            model_version=settings.DETECTION_MODEL_VERSION,
        )
    return MockDetectionProvider(
        model_name=settings.DETECTION_MODEL_NAME,
        model_version=settings.DETECTION_MODEL_VERSION,
    )
