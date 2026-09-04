"""RetinaFace + MobileSAM Detection Provider.

Localizes faces (RetinaFace) and refines anatomical seed boxes into real garment
segmentation-derived bounding boxes (MobileSAM), served from a Runpod Serverless
endpoint (see runpod/retina_sam_detect.py). Falls back to the OpenCV heuristic
detector if the endpoint isn't configured or the call fails.
"""

import base64
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BaseDetectionProvider
from app.providers.detection.opencv_detector import OpenCVDetectorProvider
from app.schemas.pipeline import DetectionResult, GarmentRegion


class RetinaSAMDetectionProvider(BaseDetectionProvider):
    """
    RetinaFace localization + MobileSAM segmentation provider.
    Localizes person/face landmarks and produces garment segment masks.
    """

    def __init__(
        self,
        model_name: str = settings.DETECTION_MODEL_NAME,
        model_version: str = settings.DETECTION_MODEL_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = OpenCVDetectorProvider(model_name=model_name, model_version=model_version)

    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        if not settings.RUNPOD_API_KEY or not settings.RUNPOD_DETECTION_ENDPOINT_ID:
            return await self._fallback.detect_and_crop(image_bytes)

        try:
            headers = {
                "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            }
            url = f"{settings.RUNPOD_BASE_URL}/{settings.RUNPOD_DETECTION_ENDPOINT_ID}/runsync"
            payload = {"input": {"image_b64": base64.b64encode(image_bytes).decode("utf-8")}}
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()

            if result.get("status") != "COMPLETED":
                raise ValueError(f"Runpod job did not complete: {result.get('status')}")

            output = result["output"]
            garment_regions = [
                GarmentRegion(label=r["label"], box=r["box"]) for r in output.get("garment_regions", [])
            ]

            return DetectionResult(
                person_detected=output.get("person_detected", False),
                face_box=output.get("face_box"),
                garment_regions=garment_regions,
                model=self.model_name,
                model_version=self.model_version,
            )
        except Exception as e:
            logger.warning(
                f"Runpod RetinaFace+SAM detection call failed: {e}. Falling back to OpenCV detector."
            )
            return await self._fallback.detect_and_crop(image_bytes)
