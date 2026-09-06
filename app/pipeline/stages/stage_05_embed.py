"""Stage 5: Image Embedding (MODA SigLIP Distilled).

Computes normalized vector embedding for canonical garment image and persists to pgvector.
"""

import numpy as np
from app.config import settings
from app.models.embedding import GarmentEmbedding
from app.models.image_asset import ImageAsset
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.embedding import get_embedding_provider


class Stage05Embed(BaseStage):
    stage_name = PipelineStage.STAGE_05_EMBED.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Prefer canonical image; fallback to primary crop. Deliberately checks the scalar FK
        # (canonical_image_id) and fetches by id rather than trusting the lazy `canonical_image`
        # ORM relationship — that relationship can still read as falsy here even after Stage 4
        # set canonical_image_id moments earlier on this same in-memory garment object, since
        # the orchestrator never commits between stages (only flushes) and a relationship
        # accessed for the first time can cache before/without picking up the fresh FK.
        # Confirmed live: 3 sibling garments from the same source photo all embedded from the
        # shared raw source image instead of their own (genuinely distinct) canonical images,
        # producing identical embeddings for different garments.
        image_uri = None
        if ctx.garment.canonical_image_id:
            canonical_asset = await ctx.session.get(ImageAsset, ctx.garment.canonical_image_id)
            if canonical_asset:
                image_uri = canonical_asset.object_uri
        if not image_uri and ctx.garment.garment_crop_refs:
            image_uri = ctx.garment.garment_crop_refs[0]
        if not image_uri:
            image_uri = ctx.garment.source_image.object_uri

        image_bytes = await ctx.storage.get_object(image_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        provider = get_embedding_provider()
        vector = await provider.embed(image_bytes)

        # Validate dimension
        if len(vector) != settings.EMBEDDING_DIMENSION:
            return StageExecutionResult(
                status="FAILED",
                input_refs={"image_uri": image_uri},
                output_refs={},
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="embed_v1",
                error=f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, got {len(vector)}",
            )

        # Validate unit length (L2 normalization check: ||v|| ~ 1.0)
        norm = np.linalg.norm(vector)
        if abs(norm - 1.0) > 0.05:
            # Re-normalize if slight numerical drift
            normalized = (np.array(vector) / norm).tolist()
            vector = normalized

        embedding_record = GarmentEmbedding(
            garment_id=ctx.garment.id,
            embedding=vector,
            model=provider.model_name,
            model_version=provider.model_version,
            dimension=len(vector),
            source_image_version=ctx.garment.pipeline_version,
        )
        ctx.session.add(embedding_record)
        await ctx.session.flush()

        norm_val = round(float(np.linalg.norm(vector)), 4)
        vector_rounded = [round(float(x), 5) for x in vector]

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"image_uri": image_uri},
            output_refs={
                "embedding_id": embedding_record.id,
                "dimension": len(vector),
                "norm": norm_val,
                "vector": vector_rounded,
                "preview": vector_rounded[:32],
            },
            input_hash=input_hash,
            model=provider.model_name,
            model_version=provider.model_version,
            algorithm_version="embed_v1",
        )
