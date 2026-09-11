"""Mock Aesthetic Provider: deterministic score derived from the same pairwise colour-theory
rules used elsewhere, so behaviour stays sensible without a real LLM configured — mirrors how
MockSemanticValidatorProvider derives its answer from an already-computed signal rather than
a hardcoded constant."""

from itertools import combinations
from typing import List
from app.providers.base import BaseAestheticProvider
from app.rules.visual import evaluate_visual_rules
from app.schemas.styling import AestheticScoreResult, GarmentSummary, StylingContext


class MockAestheticProvider(BaseAestheticProvider):
    def __init__(self, model_name: str = "mock-aesthetic-provider", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def score_outfit(
        self,
        garments: List[GarmentSummary],
        context: StylingContext,
    ) -> AestheticScoreResult:
        if len(garments) < 2:
            return AestheticScoreResult(
                score=0.75,
                reasoning="Single-garment outfit; no pairwise composition to judge.",
                model=self.model_name,
                model_version=self.model_version,
            )

        scores = []
        for a, b in combinations(garments, 2):
            _confident, _decision, score, _reason, _ver = evaluate_visual_rules(a.attributes or {}, b.attributes or {})
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.7

        return AestheticScoreResult(
            score=avg,
            reasoning="Mock aesthetic provider: derived from pairwise colour-theory rules (no LLM configured).",
            model=self.model_name,
            model_version=self.model_version,
        )
