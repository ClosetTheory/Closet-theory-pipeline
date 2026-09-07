"""Swap/replace a single garment (by role) within an already-generated outfit.

Two entry points, sharing one code path (_apply_swap):
  - swap_garment_direct: caller already knows which real garment_id they want.
  - swap_garment_by_chat: caller describes it in free text ("swap the shoes for something more
    casual"); an LLM call identifies which role they mean, then a real candidate from that
    member's own wardrobe is picked automatically.

Either way: never invents a garment (always a real, member-owned, COMPLETED row), re-checks
compatibility with the rest of the outfit (hard-rejects a genuinely incompatible replacement
rather than silently allowing a broken outfit), and regenerates the outfit's composite image +
visual/semantic gates, since the visual composition actually changed.
"""

import json
from typing import List, Optional, Tuple
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models.garment import Garment
from app.models.styling import Outfit, OutfitGarment, StylingRequest
from app.observability import logger
from app.schemas.styling import OutfitCandidate, OutfitResult, ScoreBreakdown, StylingContext
from app.storage.base import StorageClient
from app.styling.compatibility import evaluate_pair_compatibility
from app.styling.imaging import generate_and_run_gates
from app.styling.orchestrator import persist_generated_image
from app.styling.replay import build_outfit_result

# How many of the member's other same-role garments to actually score against the rest of the
# outfit for a chat-swap — each one costs a real (deterministic-rules-first) compatibility
# check, so this bounds worst-case latency rather than scoring an entire large wardrobe.
MAX_CHAT_SWAP_CANDIDATES = 8


class SwapError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def _load_outfit(session: AsyncSession, tenant_id: str, member_id: str, outfit_id: str) -> Tuple[Outfit, List[OutfitGarment], List[Garment]]:
    outfit = await session.get(Outfit, outfit_id)
    if not outfit:
        raise SwapError(f"Outfit '{outfit_id}' not found", 404)
    if outfit.tenant_id != tenant_id or outfit.member_id != member_id:
        raise SwapError("This outfit belongs to another account.", 403)

    og_rows = list((await session.execute(select(OutfitGarment).where(OutfitGarment.outfit_id == outfit_id))).scalars().all())
    garment_ids = [og.garment_id for og in og_rows]
    garments_by_id = {g.id: g for g in (await session.execute(select(Garment).where(Garment.id.in_(garment_ids)))).scalars().all()}
    ordered_garments = [garments_by_id[gid] for gid in garment_ids if gid in garments_by_id]
    return outfit, og_rows, ordered_garments


async def _compatibility_with_rest(candidate: Garment, other_garments: List[Garment]) -> Tuple[str, float, List[str]]:
    """Same pairwise checks used during ranking (app/styling/compatibility.py), run between
    `candidate` and every OTHER garment that would remain in the outfit."""
    worst = "COMPATIBLE"
    scores = []
    reasons = []
    for other in other_garments:
        decision, score, reason, _hard_reject_source = await evaluate_pair_compatibility(candidate, other)
        scores.append(score)
        reasons.append(f"{candidate.subcategory}+{other.subcategory}: {reason}")
        if decision == "INCOMPATIBLE":
            worst = "INCOMPATIBLE"
        elif decision == "REVIEW_REQUIRED" and worst != "INCOMPATIBLE":
            worst = "REVIEW_REQUIRED"
    avg_score = sum(scores) / len(scores) if scores else 1.0  # nothing else in the outfit -> trivially fine
    return worst, avg_score, reasons


async def _validate_replacement_garment(session: AsyncSession, tenant_id: str, member_id: str, new_garment_id: str, exclude_ids: set) -> Garment:
    garment = await session.get(Garment, new_garment_id)
    if not garment or garment.tenant_id != tenant_id or garment.member_id != member_id:
        raise SwapError(f"Garment '{new_garment_id}' not found or not accessible to this member.", 404)
    if garment.status != "COMPLETED":
        raise SwapError(f"Garment '{new_garment_id}' isn't fully processed yet (status={garment.status}).", 400)
    if garment.id in exclude_ids:
        raise SwapError(f"Garment '{new_garment_id}' is already part of this outfit.", 400)
    return garment


async def _apply_swap(
    session: AsyncSession,
    storage: StorageClient,
    tenant_id: str,
    member_id: str,
    outfit_id: str,
    role: str,
    new_garment: Garment,
) -> OutfitResult:
    outfit, og_rows, garments = await _load_outfit(session, tenant_id, member_id, outfit_id)

    target_og = next((og for og in og_rows if og.role.upper() == role.upper()), None)
    if not target_og:
        available = ", ".join(sorted({og.role for og in og_rows})) or "none"
        raise SwapError(f"This outfit has no '{role}' role to replace. Roles present: {available}", 404)

    other_garments = [g for g in garments if g.id != target_og.garment_id]

    decision, score, reasons = await _compatibility_with_rest(new_garment, other_garments)
    if decision == "INCOMPATIBLE":
        raise SwapError(
            f"'{new_garment.subcategory or new_garment.id}' isn't compatible with the rest of "
            f"this outfit: {'; '.join(reasons)}",
            409,
        )

    # Apply the swap
    target_og.garment_id = new_garment.id
    updated_garments = other_garments + [new_garment]
    roles_map = {g.id: (role if g.id == new_garment.id else next((o.role for o in og_rows if o.garment_id == g.id), "")) for g in updated_garments}

    scores = dict(outfit.score_breakdown or {})
    scores["compatibility"] = score
    outfit.score_breakdown = scores
    outfit.compatibility_reason = "; ".join(reasons) if reasons else None

    # Regenerate the composite image + gates — the visual composition genuinely changed.
    styling_request = await session.get(StylingRequest, outfit.request_id)
    context = (
        StylingContext.model_validate(styling_request.context)
        if styling_request and styling_request.context
        else StylingContext(intent={})
    )
    candidate = OutfitCandidate(
        outfit_id=outfit.id,
        garment_ids=[g.id for g in updated_garments],
        roles=roles_map,
        compatibility_reason=outfit.compatibility_reason,
        scores=ScoreBreakdown.model_validate(scores),
    )
    image_bytes, visual_gate, semantic_gate, _passed = await generate_and_run_gates(context, candidate, updated_garments, storage)
    if image_bytes:
        outfit.generated_image_id = await persist_generated_image(session, storage, tenant_id, member_id, image_bytes)
    if visual_gate:
        outfit.visual_validation = visual_gate.model_dump(mode="json")
    if semantic_gate:
        outfit.generated_image_semantic_validation = semantic_gate.model_dump(mode="json")

    await session.commit()
    await session.refresh(outfit)
    return await build_outfit_result(session, outfit)


async def swap_garment_direct(
    session: AsyncSession, storage: StorageClient, tenant_id: str, member_id: str,
    outfit_id: str, role: str, new_garment_id: str,
) -> OutfitResult:
    _outfit, _og_rows, garments = await _load_outfit(session, tenant_id, member_id, outfit_id)
    new_garment = await _validate_replacement_garment(session, tenant_id, member_id, new_garment_id, {g.id for g in garments})
    return await _apply_swap(session, storage, tenant_id, member_id, outfit_id, role, new_garment)


async def _interpret_swap_instruction(instruction: str, current_roles: List[str]) -> Tuple[str, str]:
    """Returns (role, style_hint). Raises SwapError rather than guessing — a wrong role guess
    would swap the wrong garment, which is worse than asking the caller to be more specific."""
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise SwapError("Chat-based swap requires OPENROUTER_API_KEY to be configured.", 503)

    roles_str = ", ".join(sorted(set(current_roles)))
    prompt = f"""The current outfit has these garment roles: {roles_str}.
A user wants to change something about the outfit. Their instruction: "{instruction}"

Identify which ONE role they want to replace, and any style/attribute hint for the replacement
(colour, formality, pattern, anything descriptive they mentioned). Output ONLY raw JSON:
{{"role": "<exactly one of: {roles_str}>", "style_hint": "short phrase, or empty string if none"}}"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{settings.OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
    except SwapError:
        raise
    except Exception as e:
        logger.warning(f"Swap instruction interpretation failed: {e}")
        raise SwapError(f"Couldn't understand the swap instruction: {e}", 502)

    role = str(data.get("role", "")).strip().upper()
    if role not in {r.upper() for r in current_roles}:
        raise SwapError(
            f"Could not confidently tell which part of the outfit to change from: \"{instruction}\". "
            f"Try naming it directly, e.g. \"replace the footwear\". Roles present: {roles_str}",
            422,
        )
    return role, str(data.get("style_hint", "") or "")


async def swap_garment_by_chat(
    session: AsyncSession, storage: StorageClient, tenant_id: str, member_id: str,
    outfit_id: str, instruction: str,
) -> OutfitResult:
    outfit, og_rows, garments = await _load_outfit(session, tenant_id, member_id, outfit_id)
    role, style_hint = await _interpret_swap_instruction(instruction, [og.role for og in og_rows])

    target_og = next((og for og in og_rows if og.role.upper() == role.upper()), None)
    other_garments = [g for g in garments if not target_og or g.id != target_og.garment_id]
    exclude_ids = {g.id for g in garments}

    stmt = select(Garment).where(
        Garment.tenant_id == tenant_id, Garment.member_id == member_id,
        Garment.status == "COMPLETED", Garment.category == role,
    )
    same_role_garments = [g for g in (await session.execute(stmt)).scalars().all() if g.id not in exclude_ids]
    if not same_role_garments:
        raise SwapError(f"No other {role.lower()} garments found in this member's wardrobe to swap in.", 404)

    hint_lower = style_hint.lower()
    scored: List[Tuple[Garment, float]] = []
    for cand in same_role_garments[:MAX_CHAT_SWAP_CANDIDATES]:
        decision, score, _reasons = await _compatibility_with_rest(cand, other_garments)
        if decision == "INCOMPATIBLE":
            continue
        attrs = cand.attributes_json or {}
        keyword_bonus = 0.15 if hint_lower and any(
            kw and kw.lower() in hint_lower for kw in [attrs.get("pattern"), attrs.get("fit"), *attrs.get("colour", [])]
        ) else 0.0
        scored.append((cand, score + keyword_bonus))

    if not scored:
        raise SwapError(f"None of this member's other {role.lower()} garments are compatible with the rest of this outfit.", 409)
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_candidate = scored[0][0]

    return await _apply_swap(session, storage, tenant_id, member_id, outfit_id, role, best_candidate)


async def list_swap_candidates(session: AsyncSession, tenant_id: str, member_id: str, outfit_id: str, role: str) -> List[Garment]:
    """The member's own real garments of `role` not already in this outfit — for a 'browse and
    pick' UI, as an alternative to naming a garment_id directly or describing it in chat."""
    _outfit, _og_rows, garments = await _load_outfit(session, tenant_id, member_id, outfit_id)
    exclude_ids = {g.id for g in garments}
    stmt = select(Garment).where(
        Garment.tenant_id == tenant_id, Garment.member_id == member_id,
        Garment.status == "COMPLETED", Garment.category == role.upper(),
    )
    return [g for g in (await session.execute(stmt)).scalars().all() if g.id not in exclude_ids]
