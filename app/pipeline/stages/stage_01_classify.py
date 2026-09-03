"""Stage 1: Image Classifier (MobileNetV3).

Determines whether the image is CATALOG, CROP, or FULL_BODY.
Enforces confidence threshold routing to review.
"""

from app.config import settings
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.classifier import get_classifier_provider


class Stage01Classify(BaseStage):
    stage_name = PipelineStage.STAGE_01_CLASSIFY.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        source_image = ctx.garment.source_image
        image_bytes = await ctx.storage.get_object(source_image.object_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        provider = get_classifier_provider()
        classification = await provider.classify(image_bytes)

        # PRD Section 8: Unsupported/low-confidence cases routed to review
        threshold = settings.CLASSIFIER_CONFIDENCE_THRESHOLD
        if classification.confidence < threshold:
            status = "REVIEW_REQUIRED"
            quality_status = "REVIEW_REQUIRED"
            error = f"Low classifier confidence ({classification.confidence:.2f} < {threshold:.2f})"
        else:
            status = "SUCCEEDED"
            quality_status = "APPROVED"
            error = None

        # Update garment image type
        ctx.garment.image_type = classification.image_type.value

        return StageExecutionResult(
            status=status,
            input_refs={"source_image_id": source_image.id, "object_uri": source_image.object_uri},
            output_refs={
                "image_type": classification.image_type.value,
                "confidence": classification.confidence,
            },
            input_hash=input_hash,
            model=classification.model,
            model_version=classification.model_version,
            algorithm_version="classifier_v1",
            error=error,
            quality_status=quality_status,
        )
