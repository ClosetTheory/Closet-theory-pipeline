"""OpenCV Real Computer Vision Face Detector and Garment Cropper."""

import io
import cv2
import numpy as np
from PIL import Image
from app.config import settings
from app.providers.base import BaseDetectionProvider
from app.schemas.pipeline import DetectionResult, GarmentRegion


class OpenCVDetectorProvider(BaseDetectionProvider):
    """
    Executes real computer vision on CPU:
    1. Detects human faces using OpenCV frontal-face Haar cascades.
    2. Derives anatomical garment torso/leg coordinates from facial landmarks.
    3. If no face is detected (flat-lay/catalog), extracts garment contours.
    """

    def __init__(
        self,
        model_name: str = settings.DETECTION_MODEL_NAME,
        model_version: str = settings.DETECTION_MODEL_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    async def detect_and_crop(self, image_bytes: bytes) -> DetectionResult:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            # Fallback if unreadable
            return DetectionResult(
                person_detected=False,
                face_box=None,
                garment_regions=[GarmentRegion(label="upper_body", box=[0, 0, 100, 100])],
                model=self.model_name,
                model_version=self.model_version,
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(w * 0.08), int(h * 0.08)),
        )

        if len(faces) > 0:
            # Sort by area descending to find the primary face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            face_box = [int(fx), int(fy), int(fx + fw), int(fy + fh)]

            # Anatomical upper-body torso projection below the face
            upper_y1 = min(h - 1, int(fy + fh * 0.95))
            upper_y2 = min(h, int(fy + fh * 4.8))
            upper_x1 = max(0, int(fx - fw * 1.4))
            upper_x2 = min(w, int(fx + fw * 2.4))

            garment_regions = [
                GarmentRegion(
                    label="upper_body",
                    box=[upper_x1, upper_y1, upper_x2, upper_y2],
                    mask_ref=None,
                )
            ]

            # If full-body image, add lower-body region
            if upper_y2 < h - int(h * 0.15):
                lower_box = [
                    max(0, int(fx - fw * 1.0)),
                    upper_y2,
                    min(w, int(fx + fw * 2.0)),
                    h,
                ]
                garment_regions.append(GarmentRegion(label="lower_body", box=lower_box))

            return DetectionResult(
                person_detected=True,
                face_box=face_box,
                garment_regions=garment_regions,
                model=self.model_name,
                model_version=self.model_version,
            )

        # No person/face detected: Flat-lay or catalog photo
        # Contour detection to find garment bounding box
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)
            # Add small 5% padding
            pad_x = int(cw * 0.05)
            pad_y = int(ch * 0.05)
            box = [
                max(0, cx - pad_x),
                max(0, cy - pad_y),
                min(w, cx + cw + pad_x),
                min(h, cy + ch + pad_y),
            ]
        else:
            box = [int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)]

        return DetectionResult(
            person_detected=False,
            face_box=None,
            garment_regions=[GarmentRegion(label="upper_body", box=box)],
            model=self.model_name,
            model_version=self.model_version,
        )
