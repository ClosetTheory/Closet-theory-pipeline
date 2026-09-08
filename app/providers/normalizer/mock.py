"""Mock Request Normalizer: simple keyword heuristics, no LLM call."""

import re
from typing import List
from app.config import settings
from app.providers.base import BaseRequestNormalizerProvider
from app.schemas.styling import StylingIntent

_OCCASION_KEYWORDS = {
    "dinner": "DINNER", "date": "DATE", "work": "WORK", "office": "WORK",
    "party": "PARTY", "wedding": "WEDDING", "gym": "ACTIVEWEAR", "casual": "CASUAL",
    "funeral": "FUNERAL", "memorial": "FUNERAL", "interview": "INTERVIEW",
    "graduation": "GRADUATION", "baby shower": "BABY_SHOWER",
}
_FORMALITY_KEYWORDS = {
    "formal": "FORMAL", "smart casual": "SMART_CASUAL", "business casual": "BUSINESS_CASUAL",
    "casual": "CASUAL", "dressy": "SMART_CASUAL",
}
# Occasions with an obvious conventional formality, applied only when no explicit formality
# keyword was already matched above — mirrors the inference the real LLM normalizer is now
# prompted to do, so the offline mock path behaves consistently.
_OCCASION_DEFAULT_FORMALITY = {
    "FUNERAL": "FORMAL", "WEDDING": "FORMAL", "INTERVIEW": "FORMAL", "GRADUATION": "FORMAL",
}
_COLOR_KEYWORDS = [
    "black", "white", "navy", "blue", "grey", "gray", "beige", "brown", "olive",
    "green", "red", "pink", "yellow", "dark", "light", "neutral",
]


class MockRequestNormalizerProvider(BaseRequestNormalizerProvider):
    """Deterministic keyword-based normalizer used when no LLM provider is configured."""

    def __init__(self, model_name: str = "mock-normalizer", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def normalize(self, request_text: str, anchor_categories: List[str]) -> StylingIntent:
        text = (request_text or "").lower()

        occasion = next((v for k, v in _OCCASION_KEYWORDS.items() if k in text), None)
        formality = next((v for k, v in _FORMALITY_KEYWORDS.items() if k in text), None)
        if not formality and occasion:
            formality = _OCCASION_DEFAULT_FORMALITY.get(occasion)
        colors = [c.upper() for c in _COLOR_KEYWORDS if re.search(rf"\b{c}\b", text)]
        time_context = "TONIGHT" if "tonight" in text else ("TOMORROW" if "tomorrow" in text else None)

        return StylingIntent(
            occasion=occasion,
            formality=formality,
            colors=colors,
            time_context=time_context,
        )
