"""Real holistic Aesthetic Provider via OpenRouter.

Judges a candidate outfit as a single composition in ONE LLM call — not averaged from
pairwise checks the way visual_harmony is — since a real stylist reasons about a whole look
(colour story, proportion, focal point, styled cohesion), not "does A clash with B, times
every pair." Direct httpx call, matching the established pattern used elsewhere in this
codebase (app/providers/vlm/openrouter.py, app/styling/swap.py's instruction interpretation)
rather than a full Provider ABC wrapping a shared client.
"""

import json
from typing import List, Optional
import httpx
from app.config import settings
from app.observability import logger
from app.providers.base import BaseAestheticProvider
from app.schemas.styling import AestheticScoreResult, GarmentSummary, StylingContext


class OpenRouterAestheticProvider(BaseAestheticProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = settings.OPENROUTER_MODEL,
        base_url: str = settings.OPENROUTER_BASE_URL,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name
        self.base_url = base_url
        self.model_version = "v1"

    def _describe_garment(self, g: GarmentSummary) -> str:
        attrs = g.attributes or {}
        colour = ", ".join(attrs.get("colour", [])) or "unknown colour"
        pattern = attrs.get("pattern", "solid")
        subcat = (attrs.get("subcategory") or g.subcategory or g.category or "item").replace("_", " ")
        fit = attrs.get("fit")
        material = attrs.get("material")
        extras = ", ".join(v for v in (fit, material) if v)
        role = (g.role or g.category or "").lower()
        return f"{role}: {colour} {pattern} {subcat}" + (f" ({extras})" if extras else "")

    async def score_outfit(
        self,
        garments: List[GarmentSummary],
        context: StylingContext,
    ) -> AestheticScoreResult:
        if len(garments) < 2:
            return AestheticScoreResult(
                score=0.75,
                reasoning="Single-garment outfit; no composition to judge.",
                model=self.model_name,
                model_version=self.model_version,
            )
        if not self.api_key:
            return AestheticScoreResult(
                score=0.6,
                reasoning="No OpenRouter API key configured; neutral fallback score.",
                model=self.model_name,
                model_version=self.model_version,
            )

        pieces_desc = "\n".join(f"- {self._describe_garment(g)}" for g in garments)
        occasion = context.intent.occasion or "no specific occasion stated"
        formality = context.intent.formality or "no specific formality stated"

        prompt = f"""You are a professional fashion stylist judging ONE candidate outfit as a complete look — \
not checking whether the pieces merely avoid clashing, but whether a stylist would call this a genuinely \
well put-together, aesthetically pleasing outfit. Consider the whole composition: colour story (not just \
"do these clash" but "does this palette feel intentional and elevated"), proportion and silhouette balance, \
whether there's a clear focal point versus visual competition, and overall styled cohesion. Judge the \
combination AS PRESENTED — do not invent, add, or substitute any garment.

Occasion: {occasion}. Formality: {formality}.

Outfit pieces:
{pieces_desc}

Output ONLY JSON:
{{"score": 0-10 (0 = looks thrown together, 10 = magazine-editorial styled), "reasoning": "one or two sentences, specific to THESE exact pieces, not generic"}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Wardrobe Styling Pipeline",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self.base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)
            score = max(0.0, min(1.0, float(data.get("score", 5)) / 10.0))
            reasoning = str(data.get("reasoning", "") or "")
            return AestheticScoreResult(
                score=score,
                reasoning=reasoning,
                model=self.model_name,
                model_version=self.model_version,
            )
        except Exception as e:
            logger.warning(f"Aesthetic scoring call failed: {e}")
            return AestheticScoreResult(
                score=0.6,
                reasoning=f"Aesthetic judgment unavailable ({type(e).__name__}); neutral fallback score.",
                model=self.model_name,
                model_version=self.model_version,
            )
