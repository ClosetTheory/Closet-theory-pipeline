"""Compatibility evaluation API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_db_session, get_storage
from app.config import settings
from app.models.compatibility import CompatibilityResult
from app.models.garment import Garment
from app.pipeline.image_resolution import resolve_garment_image_bytes
from app.providers.vlm import get_vlm_provider
from app.rules.layering import evaluate_layering_compatibility
from app.rules.structural import evaluate_structural_compatibility
from app.rules.visual import evaluate_visual_rules
from app.storage.base import StorageClient
from app.schemas.compatibility import (
    CompatibilityDecision,
    CompatibilityEvaluateRequest,
    CompatibilityEvaluateResponse,
    CompatibilityItemResult,
    CompatibilityType,
)

router = APIRouter(prefix="/wardrobe/compatibility", tags=["Compatibility"])


@router.post("", response_model=CompatibilityEvaluateResponse)
async def evaluate_compatibility(
    request: CompatibilityEvaluateRequest,
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """
    Evaluates compatibility between two canonical garments across layering, structural, and visual criteria.
    """
    garment_a = await session.get(Garment, request.garment_a_id)
    if not garment_a:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment A '{request.garment_a_id}' not found",
        )

    garment_b = await session.get(Garment, request.garment_b_id)
    if not garment_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Garment B '{request.garment_b_id}' not found",
        )

    eval_types = request.types or [
        CompatibilityType.LAYERING,
        CompatibilityType.STRUCTURAL,
        CompatibilityType.VISUAL,
    ]

    results = []
    overall_incompatible = False
    overall_review_needed = False

    attrs_a = {**garment_a.attributes_json, "category": garment_a.category}
    attrs_b = {**garment_b.attributes_json, "category": garment_b.category}

    # 1. Layering
    if CompatibilityType.LAYERING in eval_types:
        decision, score, reason, ver = evaluate_layering_compatibility(attrs_a, attrs_b)
        results.append(
            CompatibilityItemResult(
                compatibility_type=CompatibilityType.LAYERING,
                decision=CompatibilityDecision(decision),
                score=score,
                reason=reason,
                algorithm_version=ver,
            )
        )
        if decision == "INCOMPATIBLE":
            overall_incompatible = True
        elif decision == "REVIEW_REQUIRED":
            overall_review_needed = True

    # 2. Structural
    if CompatibilityType.STRUCTURAL in eval_types:
        decision, score, reason, ver = evaluate_structural_compatibility(attrs_a, attrs_b)
        results.append(
            CompatibilityItemResult(
                compatibility_type=CompatibilityType.STRUCTURAL,
                decision=CompatibilityDecision(decision),
                score=score,
                reason=reason,
                algorithm_version=ver,
            )
        )
        if decision == "INCOMPATIBLE":
            overall_incompatible = True
        elif decision == "REVIEW_REQUIRED":
            overall_review_needed = True

    # 3. Visual
    if CompatibilityType.VISUAL in eval_types:
        confident, decision, score, reason, ver = evaluate_visual_rules(attrs_a, attrs_b)
        model_ver = None

        if not confident:
            # Fallback to VLM — needs the real images, not just the attribute strings
            image_a_bytes = await resolve_garment_image_bytes(storage, garment_a)
            image_b_bytes = await resolve_garment_image_bytes(storage, garment_b)
            vlm = get_vlm_provider()
            model_ver = f"{vlm.model_name}:{vlm.model_version}"
            decision, score, reason = await vlm.evaluate_visual_compatibility(
                image_a_bytes, image_b_bytes, attrs_a, attrs_b
            )

        results.append(
            CompatibilityItemResult(
                compatibility_type=CompatibilityType.VISUAL,
                decision=CompatibilityDecision(decision),
                score=score,
                reason=reason,
                algorithm_version=ver,
                model_version=model_ver,
            )
        )
        if decision == "INCOMPATIBLE":
            overall_incompatible = True
        elif decision == "REVIEW_REQUIRED":
            overall_review_needed = True

    overall_decision = (
        CompatibilityDecision.INCOMPATIBLE
        if overall_incompatible
        else CompatibilityDecision.REVIEW_REQUIRED
        if overall_review_needed
        else CompatibilityDecision.COMPATIBLE
    )

    return CompatibilityEvaluateResponse(
        garment_a_id=garment_a.id,
        garment_b_id=garment_b.id,
        overall_decision=overall_decision,
        results=results,
    )
