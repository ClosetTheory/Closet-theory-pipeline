"""MODA_NER Attribute Extractor Provider.

Calls the Runpod-hosted MODA_NER(V) endpoint (see runpod/moda_attributes.py) for the
track matching the image's classification (crop/catalog/fullbody), maps its raw output
into ClosetTheory's canonical schema (app/providers/attributes/moda_ner_mapping.py),
and tops up whatever fields that track structurally cannot produce (occasion, season,
warmth, versatility, and — depending on track — category/subcategory/colour/fit/
silhouette) with a cheap grounded OpenRouter VLM call. This hybrid split matches
SPEC.md Section 38: MODA_NER owns attribute extraction, VLM owns validation/judgment.
"""

import base64
from typing import Optional
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BaseAttributeExtractorProvider
from app.providers.attributes.mock import MockAttributeExtractorProvider
from app.providers.attributes.moda_ner_mapping import TRACK_MAPPERS
from app.providers.vlm.openrouter import OpenRouterGPTProvider
from app.schemas.attributes import GarmentAttributes, validate_extracted_attributes

_IMAGE_TYPE_TO_TRACK = {
    "CATALOG": "catalog",
    "CROP": "crop",
    "FULL_BODY": "fullbody",
}


class ModaNerAttributeExtractorProvider(BaseAttributeExtractorProvider):
    """MODA fashion named entity and attribute recognition provider."""

    def __init__(
        self,
        model_name: str = settings.ATTRIBUTE_MODEL_NAME,
        model_version: str = settings.ATTRIBUTE_MODEL_VERSION,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self._fallback = MockAttributeExtractorProvider(model_name=model_name, model_version=model_version)
        self._topup = OpenRouterGPTProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )

    async def extract_attributes(
        self, image_bytes: bytes, image_type: Optional[str] = None, garment_label: Optional[str] = None
    ) -> GarmentAttributes:
        track = _IMAGE_TYPE_TO_TRACK.get((image_type or "").upper(), "crop")

        if not settings.RUNPOD_API_KEY or not settings.RUNPOD_ATTRIBUTE_ENDPOINT_ID:
            return await self._fallback.extract_attributes(image_bytes)

        try:
            partial = await self._extract_structural(image_bytes, track)
        except Exception as e:
            logger.warning(f"MODA_NER Runpod call failed: {e}. Falling back to mock.")
            return await self._fallback.extract_attributes(image_bytes)

        try:
            topup = await self._topup.extract_soft_attributes(image_bytes, known=partial)
        except Exception as e:
            logger.warning(f"MODA_NER VLM top-up failed: {e}. Proceeding with structural fields only.")
            topup = {}

        merged = {**partial, **topup}
        merged.setdefault("confidence", 0.75)

        # Deliberately NOT caught here: an AttributeValidationError means the merged
        # result is genuinely unmappable (e.g. subcategory outside the known taxonomy)
        # and must route to human review (Stage03Attributes), not a silent mock swap.
        return validate_extracted_attributes(merged)

    async def _extract_structural(self, image_bytes: bytes, track: str) -> dict:
        headers = {
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{settings.RUNPOD_BASE_URL}/{settings.RUNPOD_ATTRIBUTE_ENDPOINT_ID}/runsync"
        payload = {
            "input": {
                "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                "track": track,
            }
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()

        if result.get("status") != "COMPLETED":
            raise ValueError(f"Runpod job did not complete: {result.get('status')}")

        raw = result["output"]["attributes"]
        return TRACK_MAPPERS[track](raw)
