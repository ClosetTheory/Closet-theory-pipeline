"""Live pipeline-comparison endpoints: our own attribute/styling pipeline vs. Hopit's, run
side by side on the same input. Deliberately stateless — nothing here writes a new table or
column (this project has no migrations, and a schema change against the live database is not
something a comparison feature should be doing quietly). Every call recomputes both sides live;
nothing about "our" side changes behaviour for the rest of the app.

Two comparisons:
  - Attribute extraction: our configured provider (get_attribute_provider(), whatever
    ATTRIBUTE_PROVIDER is set to) vs. MODA_NER's structural extraction (Hopit's own model,
    run on our RunPod endpoint — the same call exercised in
    benchmarks/moda_vs_gemini_attribute_extraction/) on the crop track, which is the correct
    track for a single already-cropped garment photo.
  - Styling: our StylingOrchestrator vs. Hopit's hosted /v1/outfits:rank, given the same
    candidate pool and request text.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import ActingScope, get_acting_scope, get_db_session, get_storage
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.providers.attributes import get_attribute_provider
from app.providers.attributes.moda_ner import ModaNerAttributeExtractorProvider
from app.providers.hopit_client import HopitClient, HopitError
from app.schemas.attributes import AttributeValidationError
from app.schemas.styling import StylingRecommendationRequest
from app.storage.base import StorageClient
from app.styling.orchestrator import StylingOrchestrator

router = APIRouter(prefix="/hopit", tags=["Hopit comparison"])


def _attrs_or_error(fn):
    async def wrapped(*args, **kwargs):
        try:
            return {"ok": True, "attributes": (await fn(*args, **kwargs))}
        except AttributeValidationError as e:
            return {"ok": False, "error": f"{e.stage}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrapped


async def _extract_ours(image_bytes: bytes, image_type: Optional[str]) -> Dict[str, Any]:
    provider = get_attribute_provider()
    result = await provider.extract_attributes(image_bytes, image_type=image_type)
    return {
        "provider": type(provider).__name__,
        "model": getattr(provider, "model_name", "unknown"),
        **(result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)),
    }


async def _extract_moda(image_bytes: bytes) -> Dict[str, Any]:
    """MODA_NER's raw structural output, no OpenRouter top-up — the same "MODA on its own"
    call used in the attribute-extraction benchmark, not the production hybrid."""
    provider = ModaNerAttributeExtractorProvider()
    raw = await provider._extract_structural(image_bytes, track="crop")
    return {"provider": "ModaNerAttributeExtractorProvider", "track": "crop", **raw}


async def _resolve_garment_image(garment_id: str, scope: ActingScope, session: AsyncSession, storage: StorageClient):
    garment = await session.get(Garment, garment_id)
    if not garment or garment.tenant_id != scope.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")
    image_asset_id = garment.canonical_image_id or garment.source_image_id
    image_asset = await session.get(ImageAsset, image_asset_id)
    if not image_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment has no stored image")
    image_bytes = await storage.get_object(image_asset.object_uri)
    return garment, image_bytes


class AttributeCompareResponse(BaseModel):
    garment_id: Optional[str] = None
    ours: Dict[str, Any]
    moda: Dict[str, Any]


@router.post("/attributes/compare/upload", response_model=AttributeCompareResponse)
async def compare_attributes_upload(file: UploadFile = File(...)):
    """Ephemeral comparison, no garment created and nothing persisted — for the Hopit tab's
    "try an image" section, before deciding whether to actually save it via the normal
    POST /wardrobe/images + POST /wardrobe/garments flow."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    ours = await _attrs_or_error(_extract_ours)(image_bytes, None)
    moda = await _attrs_or_error(_extract_moda)(image_bytes)
    return AttributeCompareResponse(ours=ours, moda=moda)


@router.post("/attributes/compare/{garment_id}", response_model=AttributeCompareResponse)
async def compare_attributes_existing(
    garment_id: str,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Same comparison, but against a garment already in the catalogue — what the Review tab's
    split view uses. "Ours" here is re-run live rather than read back from the stored
    attributes_json, so both sides of the comparison are on equal footing (same call shape,
    same moment in time)."""
    garment, image_bytes = await _resolve_garment_image(garment_id, scope, session, storage)
    ours = await _attrs_or_error(_extract_ours)(image_bytes, garment.image_type)
    moda = await _attrs_or_error(_extract_moda)(image_bytes)
    return AttributeCompareResponse(garment_id=garment_id, ours=ours, moda=moda)


class StylingCompareRequest(BaseModel):
    request_text: Optional[str] = Field(default=None, description="Free-text styling request")
    candidates: Optional[List[str]] = Field(default=None, description="Garment ids to consider; defaults to the whole wardrobe in scope")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Hopit context, e.g. {occasion, season, temperature_c}")
    outfit_count: int = Field(default=3, ge=1, le=10)


class StylingCompareResponse(BaseModel):
    ours: Dict[str, Any]
    hopit: Dict[str, Any]


@router.post("/styling/compare", response_model=StylingCompareResponse)
async def compare_styling(
    body: StylingCompareRequest,
    scope: ActingScope = Depends(get_acting_scope),
    session: AsyncSession = Depends(get_db_session),
    storage: StorageClient = Depends(get_storage),
):
    """Runs our own StylingOrchestrator and Hopit's /v1/outfits:rank against the same candidate
    pool and request text. Candidates default to the acting scope's whole COMPLETED wardrobe —
    same query shape as GET /wardrobe/garments."""
    candidates = body.candidates
    if not candidates:
        rows = (
            await session.execute(
                select(Garment.id).where(Garment.tenant_id == scope.tenant_id, Garment.status == "COMPLETED").limit(200)
            )
        ).scalars().all()
        candidates = list(rows)
    if not candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No completed garments in this wardrobe to style")

    # ours
    try:
        orchestrator = StylingOrchestrator(session, storage)
        ours_result = await orchestrator.run(
            StylingRecommendationRequest(request_text=body.request_text, top_k=body.outfit_count),
            scope.tenant_id, scope.member_id,
        )
        ours = {"ok": True, "result": ours_result.model_dump(mode="json")}
    except Exception as e:
        ours = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # hopit
    try:
        hopit_client = HopitClient()
        hopit_result = await hopit_client.rank_outfits(
            candidates=candidates, query=body.request_text, context=body.context, outfit_count=body.outfit_count,
        )
        hopit = {"ok": True, "result": hopit_result}
    except HopitError as e:
        hopit = {"ok": False, "error": str(e)}
    except Exception as e:
        hopit = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return StylingCompareResponse(ours=ours, hopit=hopit)
