"""Thin client for Hopit's own hosted MODA API (distinct from the RunPod-hosted MODA_NER
model in app/providers/attributes/moda_ner.py, which runs the same underlying model on our
own infra). Used to call capabilities that only exist on Hopit's side, starting with outfit
ranking (/v1/outfits:rank) for the pipeline-comparison tab and review split-view.

Confirmed reachable with the existing key as of the styling-endpoint check (previously
403 scope_forbidden — see scripts/moda_ingest_wardrobes.py's docstring for that history).
"""

from typing import Any, Dict, List, Optional
import httpx
from app.config import settings


class HopitError(Exception):
    """A real failure calling Hopit's API (network, auth, 4xx/5xx) — never silently swallowed
    into a fake empty result, so a comparison view can show the caller what actually happened."""


class HopitClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or settings.MODA_API or "").rstrip("/")
        self.api_key = api_key or settings.MODA_KEY

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    async def rank_outfits(
        self,
        candidates: List[str],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        outfit_count: int = 3,
    ) -> Dict[str, Any]:
        if not self.configured:
            raise HopitError("MODA_API / MODA_KEY are not configured")
        payload: Dict[str, Any] = {"candidates": candidates, "outfit_count": outfit_count}
        if query:
            payload["query"] = query
        if context:
            payload["context"] = context
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/v1/outfits:rank", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise HopitError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()
