"""Stage 3: Image to Garment Attributes.

Extracts structured garment attributes through provider with strict 7-step validation.
"""

from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.attributes import get_attribute_provider
from app.schemas.attributes import AttributeValidationError, validate_extracted_attributes


class Stage03Attributes(BaseStage):
    stage_name = PipelineStage.STAGE_03_ATTRIBUTES.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Select best available visual crop: primary garment crop or source image
        image_uri = (
            ctx.garment.garment_crop_refs[0]
            if ctx.garment.garment_crop_refs
            else ctx.garment.source_image.object_uri
        )
        image_bytes = await ctx.storage.get_object(image_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        provider = get_attribute_provider()

        try:
            attributes = await provider.extract_attributes(image_bytes)
            # Persist canonical attributes to garment entity
            ctx.garment.attributes_json = attributes.model_dump(mode="json")
            ctx.garment.subcategory = attributes.subcategory

            return StageExecutionResult(
                status="SUCCEEDED",
                input_refs={"image_uri": image_uri},
                output_refs=attributes.model_dump(mode="json"),
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="attr_pipeline_v1",
                quality_status="APPROVED",
            )
        except AttributeValidationError as ave:
            # PRD Section 21: Attribute validation failure routed to human review
            return StageExecutionResult(
                status="REVIEW_REQUIRED",
                input_refs={"image_uri": image_uri},
                output_refs={},
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="attr_pipeline_v1",
                error=f"Attribute validation failed at {ave.stage}: {ave.message}",
                quality_status="REVIEW_REQUIRED",
            )
        except Exception as e:
            return StageExecutionResult(
                status="FAILED",
                input_refs={"image_uri": image_uri},
                output_refs={},
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="attr_pipeline_v1",
                error=f"Extraction provider failure: {str(e)}",
                quality_status="REJECTED",
            )
