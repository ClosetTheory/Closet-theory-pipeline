"""Styling Pipeline Orchestrator: ties all 10 stages together end-to-end.

request_text/anchors -> normalize -> filter -> retrieve -> combine+compatibility
-> rank -> semantic validate -> generate+visually validate image -> persist -> respond.

Image generation/visual validation only ever run on the final top-k outfits,
never on the full candidate/combination set (PRD Section 26).

Every stage's timing and a human-readable summary is captured into `self.trace`
so the frontend can render a step-by-step pipeline detail view, mirroring the
Image Ingestion Pipeline's live stage cards.
"""

import hashlib
import io
import time
import uuid
from typing import Any, Dict, List
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
from app.styling.imaging import generate_and_validate_outfit_image
from app.styling.ranking import rank_combinations
from app.styling.retrieval import resolve_role, retrieve_by_role
from app.styling.semantic_validation import validate_outfits


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
    def __init__(self, session: AsyncSession, storage: StorageClient):
        self.session = session
        self.storage = storage
        self.trace: List[StageTrace] = []

    def _record(self, stage: str, title: str, summary: Dict[str, Any], status: str, started_at: float) -> None:
        self.trace.append(
            StageTrace(
                stage=stage,
                title=title,
                status=status,
                duration_ms=round((time.perf_counter() - started_at) * 1000.0, 2),
                summary=summary,
            )
        )

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
        self._record(
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
        context = StylingContext(intent=intent)
        self._record(
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
        self._record(
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
        self._record(
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
        self._record(
            "STAGE_05_RETRIEVAL",
            "Candidate Retrieval",
            {
                "max_candidates_per_role": settings.STYLING_MAX_CANDIDATES_PER_ROLE,
                "candidates_per_role": {role: len(items) for role, items in role_candidates.items()},
                "method": "cosine similarity to anchor embeddings (Python/numpy) or versatility fallback",
            },
            "SUCCEEDED",
            t4,
        )

        # Stage 6 + 7: Combination assembly, compatibility rules + VLM fallback, weighted ranking
        t5 = time.perf_counter()
        combos = build_outfit_combinations(role_candidates, anchors)
        ranking_trace = await rank_combinations(combos, intent, top_k=max(request.top_k * 2, request.top_k + 2))
        rejected_incompatible = ranking_trace.total_evaluated - ranking_trace.total_compatible
        self._record(
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
        self._record(
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

        # Stage 8: Semantic Validation (drops FAIL, keeps NEEDS_REVIEW)
        t7 = time.perf_counter()
        validated, dropped_for_fail = await validate_outfits(
            ranking_trace.outfits, context, garments_by_id, top_k=request.top_k
        )
        self._record(
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

        t8 = time.perf_counter()
        for rank, (outfit_candidate, semantic_result) in enumerate(validated, start=1):
            garments = [garments_by_id[gid] for gid in outfit_candidate.garment_ids]

            # Stage 9 + 10: Outfit Image Generation + Visual Validation (final outfits only)
            image_bytes, visual_result = await generate_and_validate_outfit_image(
                garments, outfit_candidate.roles, self.storage
            )

            generated_image_id = None
            generated_image_url = None
            if image_bytes:
                generated_image_id = await _persist_generated_image(
                    self.session, self.storage, request.tenant_id, request.member_id, image_bytes
                )
                generated_image_url = f"/api/v1/wardrobe/images/{generated_image_id}/bytes"

            imaging_summaries.append(
                {
                    "rank": rank,
                    "image_generated": bool(generated_image_id),
                    "visual_validation_status": visual_result.status.value if visual_result else "SKIPPED",
                }
            )

            outfit_row = Outfit(
                request_id=styling_request.id,
                tenant_id=request.tenant_id,
                member_id=request.member_id,
                rank=rank,
                score_breakdown=outfit_candidate.scores.model_dump(mode="json"),
                final_score=outfit_candidate.scores.final_score,
                compatibility_reason=outfit_candidate.compatibility_reason,
                semantic_validation=semantic_result.model_dump(mode="json"),
                visual_validation=visual_result.model_dump(mode="json") if visual_result else {},
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
                    visual_validation=visual_result,
                )
            )

        self._record(
            "STAGE_09_IMAGE_GENERATION",
            "Outfit Image Generation",
            {
                "provider": settings.STYLING_OUTFIT_IMAGE_PROVIDER,
                "max_retries": settings.STYLING_IMAGE_MAX_RETRIES,
                "results": imaging_summaries,
            },
            "SUCCEEDED",
            t8,
        )
        self._record(
            "STAGE_10_VISUAL_VALIDATION",
            "Visual Validation",
            {
                "provider": settings.STYLING_VISUAL_VALIDATOR_PROVIDER,
                "results": imaging_summaries,
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
