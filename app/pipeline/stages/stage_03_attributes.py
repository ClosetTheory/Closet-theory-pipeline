"""Stage 3: Image to Garment Attributes.

Extracts structured garment attributes through provider with strict 7-step validation.
"""

from app.config import settings
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.attributes import get_attribute_provider
from app.providers.attributes.gemini import GeminiAttributeExtractorProvider
from app.providers.verification import verify_attributes_against_image
from app.schemas.attributes import AttributeValidationError, validate_extracted_attributes


class Stage03Attributes(BaseStage):
    stage_name = PipelineStage.STAGE_03_ATTRIBUTES.value

    def _providers_for_attempts(self, max_retries: int):
        """
        Retrying a failed extraction with the SAME model tends to reproduce the same mistake —
        it's the same model looking at the same image with the same blind spot (confirmed live:
        a kurta/dupatta outfit was called "oxford shirt"/"blazer"/"heels" identically on both
        attempts before this fix). Attempt 1 uses the configured primary provider; every retry
        after that uses a genuinely different vendor so it's a real second opinion, not an echo.
        """
        primary = get_attribute_provider()
        alternate = GeminiAttributeExtractorProvider()
        if isinstance(primary, GeminiAttributeExtractorProvider):
            from app.providers.vlm.openrouter import OpenRouterGPTProvider

            alternate = OpenRouterGPTProvider(
                api_key=settings.OPENROUTER_API_KEY,
                model_name=settings.OPENROUTER_MODEL,
                base_url=settings.OPENROUTER_BASE_URL,
            )
        providers = [primary] + [alternate] * max(0, max_retries - 1)
        return providers[:max_retries]

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Stage 2 no longer pixel-crops — garment_crop_refs always points at the full source
        # photo. `detected_label` (set by Stage 2, None for single-garment/catalog images) tells
        # the provider which garment within that photo to describe.
        image_uri = ctx.garment.source_image.object_uri
        garment_label = ctx.garment.detected_label
        image_bytes = await ctx.storage.get_object(image_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        max_retries = settings.ATTRIBUTE_MAX_RETRIES
        attempt_providers = self._providers_for_attempts(max_retries)
        verification_history = []
        last_provider = attempt_providers[0]
        last_validation_error = None

        try:
            attributes = None
            for attempt, provider in enumerate(attempt_providers, start=1):
                last_provider = provider
                try:
                    attributes = await provider.extract_attributes(
                        image_bytes, image_type=ctx.garment.image_type, garment_label=garment_label
                    )
                except AttributeValidationError as ave:
                    # A hard schema failure from ONE model (e.g. an enum value it invented)
                    # must not sink the whole retry loop before the other model gets a chance —
                    # only give up if every attempt fails this way.
                    last_validation_error = ave
                    verification_history.append({
                        "attempt": attempt,
                        "is_valid": False,
                        "score": 0.0,
                        "reason": f"Extraction schema error ({ave.stage}): {ave.message}",
                        "mismatches": [],
                        "extractor_model": getattr(provider, "model_name", "unknown"),
                        "verifier_model": settings.VISION_VERIFIER_MODEL,
                    })
                    continue

                # Second-opinion verification (Gemini via settings.VISION_VERIFIER_MODEL — a
                # different model/vendor than MODA_NER or GPT-4o) against the actual image:
                # catches extraction errors invisible to the rule-based cross-field checks in
                # validate_extracted_attributes() (e.g. a color/garment-type that's simply not
                # what's in the photo), independent of whichever provider produced them.
                is_match, score, reason, mismatches = await verify_attributes_against_image(
                    image_bytes, attributes, garment_label=garment_label
                )
                verification_history.append({
                    "attempt": attempt,
                    "is_valid": is_match,
                    "score": score,
                    "reason": reason,
                    "mismatches": mismatches,
                    "extractor_model": getattr(provider, "model_name", "unknown"),
                    "verifier_model": settings.VISION_VERIFIER_MODEL,
                })
                if is_match and score >= settings.ATTRIBUTE_VERIFICATION_THRESHOLD:
                    break

            if attributes is None:
                # Every attempt raised a hard schema error — nothing usable was ever extracted.
                raise last_validation_error

            # Persist canonical attributes to garment entity
            ctx.garment.attributes_json = attributes.model_dump(mode="json")
            ctx.garment.subcategory = attributes.subcategory
            ctx.garment.gender = attributes.gender.value

            last_check = verification_history[-1]
            status = "SUCCEEDED" if (last_check["is_valid"] and last_check["score"] >= settings.ATTRIBUTE_VERIFICATION_THRESHOLD) else "REVIEW_REQUIRED"
            output_data = attributes.model_dump(mode="json")
            output_data["verification_history"] = verification_history
            output_data["prompt"] = getattr(last_provider, "_last_prompt", "")

            return StageExecutionResult(
                status=status,
                input_refs={"image_uri": image_uri},
                output_refs=output_data,
                input_hash=input_hash,
                model=getattr(last_provider, "model_name", "unknown"),
                model_version=getattr(last_provider, "model_version", "v1"),
                algorithm_version="attr_pipeline_v2_cross_model_retry",
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
                model=getattr(last_provider, "model_name", "unknown"),
                model_version=getattr(last_provider, "model_version", "v1"),
                algorithm_version="attr_pipeline_v2_cross_model_retry",
                error=f"Attribute validation failed at {ave.stage}: {ave.message}",
                quality_status="REVIEW_REQUIRED",
            )
        except Exception as e:
            return StageExecutionResult(
                status="FAILED",
                input_refs={"image_uri": image_uri},
                output_refs={},
                input_hash=input_hash,
                model=getattr(last_provider, "model_name", "unknown"),
                model_version=getattr(last_provider, "model_version", "v1"),
                algorithm_version="attr_pipeline_v2_cross_model_retry",
                error=f"Extraction provider failure: {str(e)}",
                quality_status="REJECTED",
            )
