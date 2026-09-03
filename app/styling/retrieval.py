"""Styling Stage 5: Candidate Retrieval (role-aware, embedding similarity).

Cosine similarity is computed in plain Python/numpy over the already-filtered
(small) candidate set rather than a pgvector `<=>` SQL query — this avoids the
PortableVector TypeDecorator/comparator complications and works identically on
Postgres and SQLite (tests), at the cost of not scaling to huge wardrobes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models.embedding import GarmentEmbedding
from app.models.garment import Garment
from app.rules.garment_class import bundle_garment_class, infer_garment_class_from_subcategory

RoleCandidates = Dict[str, List[Tuple[Garment, float]]]


def resolve_role(garment: Garment) -> Optional[str]:
    if garment.category:
        return garment.category
    garment_class = garment.garment_class or infer_garment_class_from_subcategory(garment.subcategory or "")
    category, _, requires_review = bundle_garment_class(garment_class)
    return None if requires_review else category


async def _load_embeddings(session: AsyncSession, garment_ids: List[str]) -> Dict[str, List[float]]:
    if not garment_ids:
        return {}
    stmt = select(GarmentEmbedding).where(GarmentEmbedding.garment_id.in_(garment_ids))
    res = await session.execute(stmt)
    by_garment: Dict[str, List[float]] = {}
    for row in res.scalars().all():
        # Keep the most recently created embedding per garment (rows aren't ordered here, last-write-wins is fine for V1)
        by_garment[row.garment_id] = row.embedding
    return by_garment


def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def retrieve_by_role(
    session: AsyncSession,
    candidates: List[Garment],
    anchors: List[Garment],
    max_per_role: int = settings.STYLING_MAX_CANDIDATES_PER_ROLE,
) -> RoleCandidates:
    """Groups candidates by canonical role, scores by similarity to anchors (or neutral), caps per role."""
    anchor_ids = {a.id for a in anchors}
    non_anchor_candidates = [g for g in candidates if g.id not in anchor_ids]

    all_ids = [g.id for g in non_anchor_candidates] + [a.id for a in anchors]
    embeddings = await _load_embeddings(session, all_ids)
    anchor_vectors = [embeddings[a.id] for a in anchors if a.id in embeddings]

    role_map: RoleCandidates = {}
    for garment in non_anchor_candidates:
        role = resolve_role(garment)
        if role is None:
            continue

        vec = embeddings.get(garment.id)
        if vec and anchor_vectors:
            score = max(_cosine(vec, av) for av in anchor_vectors)
        else:
            score = float((garment.attributes_json or {}).get("versatility", 0.5))

        role_map.setdefault(role, []).append((garment, score))

    for role, scored in role_map.items():
        scored.sort(key=lambda pair: pair[1], reverse=True)
        role_map[role] = scored[:max_per_role]

    return role_map
