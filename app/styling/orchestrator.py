"""Styling Pipeline Orchestrator: ties all 10 stages together end-to-end.

request_text/anchors -> normalize -> filter -> retrieve -> combine+compatibility
-> rank -> semantic validate -> generate+visually validate image -> persist -> respond.

Image generation/visual validation only ever run on the final top-k outfits,
never on the full candidate/combination set (PRD Section 26).

Every stage's timing and a human-readable summary is captured into `self.trace`
so the frontend can render a step-by-step pipeline detail view, mirroring the
Image Ingestion Pipeline's live stage cards.
"""

import asyncio
import hashlib
import io
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.providers.normalizer import get_request_normalizer_provider
from app.schemas.styling import (
    GarmentSummary,
    OutfitResult,
    StageTrace,
    StylingContext,
    StylingIntent,
    StylingRecommendationRequest,
    StylingRecommendationResponse,
    ValidationResult,
)
from app.storage.base import StorageClient
from app.styling.combinator import build_outfit_combinations
from app.styling.filtering import filter_candidates, get_anchor_garments
from app.styling.imaging import generate_and_run_gates
from app.styling.ranking import rank_combinations
from app.styling.retrieval import resolve_role, retrieve_by_role
from app.styling.semantic_validation import validate_outfits

# How many extra semantically-validated candidates beyond top_k to keep as a fallback
# pool, so a candidate that fails the post-generation gates can be replaced by the
# next-best one instead of shrinking the final shortlist (SPEC.md Section 36).
GATE_FALLBACK_DEPTH = 2


def garment_to_summary(garment: Garment, role: str = "") -> GarmentSummary:
    canonical_url = f"/api/v1/wardrobe/images/{garment.canonical_image_id}/bytes" if garment.canonical_image_id else None
    return GarmentSummary(
        garment_id=garment.id,
        category=garment.category,
        subcategory=garment.subcategory,
        garment_class=garment.garment_class,
        role=role or resolve_role(garment) or "",
        attributes=garment.attributes_json,
        canonical_image_url=canonical_url,
        status=garment.status,
        quality_status=garment.quality_status,
        created_at=garment.created_at.isoformat() if garment.created_at else None,
    )


async def _persist_generated_image(
    session: AsyncSession,
    storage: StorageClient,
    tenant_id: str,
    member_id: str,
    image_bytes: bytes,
) -> str:
    try:
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    sha256_hash = hashlib.sha256(image_bytes).hexdigest()
    storage_key = f"outfits/{tenant_id}/{sha256_hash[:16]}_{uuid.uuid4().hex[:8]}.jpg"
    object_uri = await storage.put_object(storage_key, image_bytes, content_type="image/jpeg")

    image_asset = ImageAsset(
        tenant_id=tenant_id,
        member_id=member_id,
        object_uri=object_uri,
        mime_type="image/jpeg",
        width=width,
        height=height,
        sha256=sha256_hash,
    )
    session.add(image_asset)
    await session.flush()
    return image_asset.id


class StylingOrchestrator:
    def __init__(
        self,
        session: AsyncSession,
        storage: StorageClient,
        on_stage: Optional[Callable[[StageTrace], Awaitable[None]]] = None,
    ):
        self.session = session
        self.storage = storage
        self.trace: List[StageTrace] = []
        # Optional callback invoked immediately after each stage completes (used by the
        # streaming endpoint to push live progress to the frontend as it actually happens,
        # rather than only after the whole request finishes).
        self.on_stage = on_stage

    async def _record(self, stage: str, title: str, summary: Dict[str, Any], status: str, started_at: float) -> None:
        entry = StageTrace(
            stage=stage,
            title=title,
            status=status,
            duration_ms=round((time.perf_counter() - started_at) * 1000.0, 2),
            summary=summary,
        )
        self.trace.append(entry)
        if self.on_stage:
            await self.on_stage(entry)

    async def run(self, request: StylingRecommendationRequest) -> StylingRecommendationResponse:
        # Hard constraint checks first (code, not AI) — not its own numbered stage but must run before Stage 1
        t0 = time.perf_counter()
        anchors = await get_anchor_garments(
            self.session, request.tenant_id, request.member_id, request.anchor_garment_ids
        )
        anchor_categories = [a.category for a in anchors if a.category]

        # Stage 1: Request Normalisation (LLM)
        intent: StylingIntent = StylingIntent()
        normalizer_used = "none (no request_text supplied)"
        if request.request_text:
            normalizer = get_request_normalizer_provider()
            normalizer_used = getattr(normalizer, "model_name", type(normalizer).__name__)
            intent = await normalizer.normalize(request.request_text, anchor_categories)
        await self._record(
            "STAGE_01_NORMALISATION",
            "Request Normalisation",
            {
                "request_text": request.request_text,
                "anchor_garment_ids": request.anchor_garment_ids or [],
                "provider": normalizer_used,
                "intent": intent.model_dump(mode="json"),
            },
            "SUCCEEDED",
            t0,
        )

        # Stage 2: Contextual Analysis (V1: intent + neutral stub signals, no user profile data yet)
        t1 = time.perf_counter()
        user_preferences = {}
        if request.boldness_preference is not None:
            user_preferences["boldness_preference"] = request.boldness_preference
        context = StylingContext(intent=intent, user_preferences=user_preferences)
        await self._record(
            "STAGE_02_CONTEXT",
            "Contextual Analysis",
            {
                "note": "No StyleProfile/behavioral history yet — context is the normalised intent plus neutral stubs.",
                "context": context.model_dump(mode="json"),
            },
            "SUCCEEDED",
            t1,
        )

        # Stage 3: Wardrobe Behaviour (stub — no WearLog data yet)
        t2 = time.perf_counter()
        await self._record(
            "STAGE_03_WARDROBE_BEHAVIOUR",
            "Wardrobe Behaviour Analysis",
            {
                "note": "No WearLog/StyleProfile tables yet — every garment receives a neutral stub score (0.5).",
                "neutral_score": 0.5,
            },
            "SUCCEEDED",
            t2,
        )

        # Stage 4: DB Filtering
        t3 = time.perf_counter()
        candidates = await filter_candidates(self.session, request.tenant_id, request.member_id, intent)
        garments_by_id: Dict[str, Garment] = {g.id: g for g in candidates}
        for a in anchors:
            garments_by_id[a.id] = a
        await self._record(
            "STAGE_04_FILTERING",
            "Attribute Candidate Filtering",
            {
                "tenant_id": request.tenant_id,
                "member_id": request.member_id,
                "candidates_after_filter": len(candidates),
                "anchors_locked_in": [a.id for a in anchors],
                "filters_applied": ["status=COMPLETED", "quality_status in (APPROVED, PENDING)"]
                + (["color preference (soft)"] if intent.colors else []),
            },
            "SUCCEEDED",
            t3,
        )

        # Stage 5: Candidate Retrieval (role-aware)
        t4 = time.perf_counter()
        role_candidates = await retrieve_by_role(self.session, candidates, anchors)
        await self._record(
            "STAGE_05_RETRIEVAL",
            "Candidate Retrieval",
            {
                "max_candidates_per_role": settings.STYLING_MAX_CANDIDATES_PER_ROLE,
                "candidates_per_role": {role: len(items) for role, items in role_candidates.items()},
                "candidates_per_role_detail": {
                    role: [{"garment_id": g.id, "subcategory": g.subcategory, "score": round(score, 3)} for g, score in items]
                    for role, items in role_candidates.items()
                },
                "method": "cosine similarity to anchor embeddings (Python/numpy) or versatility fallback",
            },
            "SUCCEEDED",
            t4,
        )

        # Stage 6 + 7: Combination assembly, compatibility rules + VLM fallback, weighted ranking
        t5 = time.perf_counter()
        combos = build_outfit_combinations(role_candidates, anchors)
        ranking_trace = await rank_combinations(combos, intent, context, top_k=max(request.top_k * 2, request.top_k + 2))
        rejected_incompatible = ranking_trace.total_evaluated - ranking_trace.total_compatible
        await self._record(
            "STAGE_06_COMPATIBILITY",
            "Candidate Compatibility Analysis",
            {
                "combinations_built": len(combos),
                "combinations_evaluated": ranking_trace.total_evaluated,
                "rejected_incompatible": rejected_incompatible,
                "pairing_rejected": ranking_trace.pairing_rejected,
                "surviving_combinations": ranking_trace.total_compatible,
                "method": "deterministic pairing/layering/structural rules (hard reject) + visual rules/VLM (soft penalty only)",
            },
            "SUCCEEDED",
            t5,
        )

        t6 = time.perf_counter()
        await self._record(
            "STAGE_07_RANKING",
            "Outfit Ranking & Shortlist",
            {
                "scorer_version": settings.STYLING_SCORER_VERSION,
                "candidates_ranked": ranking_trace.total_compatible,
                "shortlist_before_validation": [
                    {"garment_ids": o.garment_ids, "final_score": round(o.scores.final_score, 3)}
                    for o in ranking_trace.outfits
                ],
            },
            "SUCCEEDED",
            t6,
        )

        # Stage 8: Semantic Validation (drops FAIL, keeps NEEDS_REVIEW).
        # Validates a small pool beyond top_k so a candidate that later fails the
        # post-generation gates (Stage 10) can be replaced instead of shrinking the shortlist.
        t7 = time.perf_counter()
        validated, dropped_for_fail = await validate_outfits(
            ranking_trace.outfits, context, garments_by_id, top_k=request.top_k + GATE_FALLBACK_DEPTH
        )
        await self._record(
            "STAGE_08_SEMANTIC_VALIDATION",
            "Semantic Validation",
            {
                "outfits_validated": len(ranking_trace.outfits),
                "dropped_for_fail": dropped_for_fail,
                "accepted": len(validated),
                "results": [
                    {"garment_ids": o.garment_ids, "status": r.status.value, "confidence": r.confidence}
                    for o, r in validated
                ],
            },
            "SUCCEEDED",
            t7,
        )

        # Persist StylingRequest (trace finalized after stages 9/10 below)
        styling_request = StylingRequest(
            tenant_id=request.tenant_id,
            member_id=request.member_id,
            raw_text=request.request_text,
            anchor_garment_ids=request.anchor_garment_ids or [],
            normalized_intent=intent.model_dump(mode="json"),
            context=context.model_dump(mode="json"),
        )
        self.session.add(styling_request)
        await self.session.flush()

        outfit_results: List[OutfitResult] = []
        imaging_summaries = []
        gate_summaries = []
        candidates_skipped = 0

        t8 = time.perf_counter()

        # Stage 9: Outfit Image Generation. Stage 10: Visual Gate + Semantic Gate, in
        # parallel, on the generated result (SPEC.md Section 36). Every candidate in the
        # fallback pool is generated/gated CONCURRENTLY (asyncio.gather) rather than one
        # at a time — each candidate is independent, so there's no reason to pay the
        # latency of a failed candidate's retries before even starting the next one.
        gate_results = await asyncio.gather(
            *(
                generate_and_run_gates(context, outfit_candidate, [garments_by_id[gid] for gid in outfit_candidate.garment_ids], self.storage)
                for outfit_candidate, _semantic_result in validated
            )
        )

        rank = 0
        for (outfit_candidate, semantic_result), (image_bytes, visual_gate, semantic_gate, passed) in zip(validated, gate_results):
            if len(outfit_results) >= request.top_k:
                break  # already have enough passing outfits; remaining pool is unused fallback headroom

            garments = [garments_by_id[gid] for gid in outfit_candidate.garment_ids]

            # Persist whatever was generated regardless of pass/fail, so rejected candidates
            # remain inspectable (e.g. in the styling pipeline's stage-detail UI) instead of
            # being silently discarded — only the ACCEPTED outfits become real Outfit rows.
            candidate_image_url = None
            candidate_image_id = None
            if image_bytes:
                candidate_image_id = await _persist_generated_image(
                    self.session, self.storage, request.tenant_id, request.member_id, image_bytes
                )
                candidate_image_url = f"/api/v1/wardrobe/images/{candidate_image_id}/bytes"

            gate_summaries.append(
                {
                    "garment_ids": outfit_candidate.garment_ids,
                    "roles": outfit_candidate.roles,
                    "passed": passed,
                    "visual_score": visual_gate.score if visual_gate else None,
                    "visual_feedback": visual_gate.feedback if visual_gate else {},
                    "semantic_status": semantic_gate.status if semantic_gate else "SKIPPED",
                    "semantic_violations": semantic_gate.violations if semantic_gate else [],
                    "generated_image_url": candidate_image_url,
                }
            )

            if not passed:
                candidates_skipped += 1
                continue

            rank += 1
            generated_image_url = candidate_image_url
            generated_image_id = candidate_image_id

            imaging_summaries.append({"rank": rank, "image_generated": True})

            outfit_row = Outfit(
                request_id=styling_request.id,
                tenant_id=request.tenant_id,
                member_id=request.member_id,
                rank=rank,
                score_breakdown=outfit_candidate.scores.model_dump(mode="json"),
                final_score=outfit_candidate.scores.final_score,
                compatibility_reason=outfit_candidate.compatibility_reason,
                semantic_validation=semantic_result.model_dump(mode="json"),
                visual_validation=visual_gate.model_dump(mode="json"),
                generated_image_semantic_validation=semantic_gate.model_dump(mode="json"),
                generated_image_id=generated_image_id,
            )
            self.session.add(outfit_row)
            await self.session.flush()

            for garment in garments:
                self.session.add(
                    OutfitGarment(
                        outfit_id=outfit_row.id,
                        garment_id=garment.id,
                        role=outfit_candidate.roles.get(garment.id, resolve_role(garment) or "UNKNOWN"),
                    )
                )

            outfit_results.append(
                OutfitResult(
                    outfit_id=outfit_row.id,
                    rank=rank,
                    garments=[garment_to_summary(g, outfit_candidate.roles.get(g.id, "")) for g in garments],
                    roles=outfit_candidate.roles,
                    scores=outfit_candidate.scores,
                    compatibility_reason=outfit_candidate.compatibility_reason,
                    semantic_validation=semantic_result,
                    generated_image_url=generated_image_url,
                    visual_gate=visual_gate,
                    generation_semantic_gate=semantic_gate,
                )
            )

        await self._record(
            "STAGE_09_IMAGE_GENERATION",
            "Outfit Image Generation",
            {
                "provider": settings.STYLING_OUTFIT_IMAGE_PROVIDER,
                "max_retries": settings.STYLING_IMAGE_MAX_RETRIES,
                "results": imaging_summaries,
                "candidates_skipped_after_gate_failure": candidates_skipped,
            },
            "SUCCEEDED",
            t8,
        )
        await self._record(
            "STAGE_10_GATES",
            "Visual Gate + Semantic Gate (parallel, on generated image)",
            {
                "visual_provider": settings.STYLING_VISUAL_VALIDATOR_PROVIDER,
                "semantic_provider": settings.STYLING_VALIDATOR_PROVIDER,
                "visual_pass_threshold": 6.0,
                "results": gate_summaries,
            },
            "SUCCEEDED",
            t8,
        )

        styling_request.trace = [entry.model_dump(mode="json") for entry in self.trace]
        await self.session.commit()

        return StylingRecommendationResponse(
            request_id=styling_request.id,
            intent=intent,
            outfits=outfit_results,
            trace=self.trace,
        )
