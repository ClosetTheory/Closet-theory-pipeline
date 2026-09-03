"""Stage 5: Image Embedding (MODA SigLIP Distilled).

Computes normalized vector embedding for canonical garment image and persists to pgvector.
"""

import numpy as np
from app.config import settings
from app.models.embedding import GarmentEmbedding
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.embedding import get_embedding_provider


class Stage05Embed(BaseStage):
    stage_name = PipelineStage.STAGE_05_EMBED.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Prefer canonical image; fallback to primary crop
        image_uri = None
        if ctx.garment.canonical_image:
            image_uri = ctx.garment.canonical_image.object_uri
        elif ctx.garment.garment_crop_refs:
            image_uri = ctx.garment.garment_crop_refs[0]
        else:
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

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"image_uri": image_uri},
            output_refs={
                "embedding_id": embedding_record.id,
                "dimension": len(vector),
                "norm": round(float(np.linalg.norm(vector)), 4),
            },
            input_hash=input_hash,
            model=provider.model_name,
            model_version=provider.model_version,
            algorithm_version="embed_v1",
        )
