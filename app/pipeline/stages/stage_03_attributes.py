"""Stage 3: Image to Garment Attributes.

Extracts structured garment attributes through provider with strict 7-step validation.
"""

from app.config import settings
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.attributes import get_attribute_provider
from app.providers.verification import verify_attributes_against_image
from app.schemas.attributes import AttributeValidationError, validate_extracted_attributes


class Stage03Attributes(BaseStage):
    stage_name = PipelineStage.STAGE_03_ATTRIBUTES.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Stage 2 no longer pixel-crops — garment_crop_refs always points at the full source
        # photo. `detected_label` (set by Stage 2, None for single-garment/catalog images) tells
        # the provider which garment within that photo to describe.
        image_uri = ctx.garment.source_image.object_uri
        garment_label = ctx.garment.detected_label
        image_bytes = await ctx.storage.get_object(image_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        provider = get_attribute_provider()
        max_retries = settings.ATTRIBUTE_MAX_RETRIES
        verification_history = []

        try:
            attributes = None
            for attempt in range(1, max_retries + 1):
                attributes = await provider.extract_attributes(
                    image_bytes, image_type=ctx.garment.image_type, garment_label=garment_label
                )

                # Second-opinion verification (Gemini via settings.VISION_VERIFIER_MODEL — a
                # different model/vendor than MODA_NER or GPT-4o) against the actual image:
                # catches extraction errors invisible to the rule-based cross-field checks in
                # validate_extracted_attributes() (e.g. a color/garment-type that's simply not
                # what's in the photo), independent of whichever provider produced them.
                is_match, score, reason, mismatches = await verify_attributes_against_image(image_bytes, attributes)
                verification_history.append({
                    "attempt": attempt,
                    "is_valid": is_match,
                    "score": score,
                    "reason": reason,
                    "mismatches": mismatches,
                    "verifier_model": settings.VISION_VERIFIER_MODEL,
                })
                if is_match and score >= settings.ATTRIBUTE_VERIFICATION_THRESHOLD:
                    break

            # Persist canonical attributes to garment entity
            ctx.garment.attributes_json = attributes.model_dump(mode="json")
            ctx.garment.subcategory = attributes.subcategory

            last_check = verification_history[-1]
            status = "SUCCEEDED" if (last_check["is_valid"] and last_check["score"] >= settings.ATTRIBUTE_VERIFICATION_THRESHOLD) else "REVIEW_REQUIRED"
            output_data = attributes.model_dump(mode="json")
            output_data["verification_history"] = verification_history

            return StageExecutionResult(
                status=status,
                input_refs={"image_uri": image_uri},
                output_refs=output_data,
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="attr_pipeline_v1",
                error=None if status == "SUCCEEDED" else f"Attribute verification failed after {max_retries} attempt(s): {last_check['reason']}",
                quality_status="APPROVED" if status == "SUCCEEDED" else "REVIEW_REQUIRED",
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
