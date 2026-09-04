"""Stage 4: Image Digitisation (FLUX.2).

Creates standardized clean canonical imagery with validation-and-retry loop.
Preserves garment identity without overwriting the original source.
"""

import hashlib
import io
from PIL import Image
from app.config import settings
from app.models.image_asset import ImageAsset
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.digitisation import get_digitisation_provider
from app.schemas.attributes import GarmentAttributes


class Stage04Digitise(BaseStage):
    stage_name = PipelineStage.STAGE_04_DIGITISE.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        # Stage 2 no longer pixel-crops — garment_crop_refs always points at the full source
        # photo. `detected_label` (set by Stage 2, None for single-garment/catalog images) tells
        # the provider which garment within that photo to isolate and reproduce.
        crop_uri = ctx.garment.source_image.object_uri
        garment_label = ctx.garment.detected_label
        crop_bytes = await ctx.storage.get_object(crop_uri)
        input_hash = compute_stage_input_hash(crop_bytes)

        attributes = GarmentAttributes.model_validate(ctx.garment.attributes_json)
        provider = get_digitisation_provider()

        max_retries = settings.DIGITISATION_MAX_RETRIES
        quality_threshold = settings.DIGITISATION_QUALITY_THRESHOLD

        accepted = False
        last_error = None
        last_result = None
        canonical_bytes = None
        verification_history = []

        for attempt in range(1, max_retries + 1):
            digit_res = await provider.digitise(crop_bytes, attributes, attempt=attempt, garment_label=garment_label)
            last_result = digit_res

            # Generate synthetic or actual canonical image bytes
            # If provider did not save bytes directly, render/fetch them:
            if hasattr(provider, "_last_generated_bytes") and provider._last_generated_bytes:
                canonical_bytes = provider._last_generated_bytes
            else:
                # Standardized canonical bytes representation
                buf = io.BytesIO()
                with Image.new("RGB", (768, 1024), color=(250, 250, 250)) as img:
                    img.save(buf, format="JPEG", quality=95)
                    canonical_bytes = buf.getvalue()

            is_valid, quality_score, reason = await provider.validate_digitisation(
                crop_bytes, canonical_bytes, attributes, garment_label=garment_label
            )
            verifier_info = getattr(provider, "_last_verification", None) or {}
            verification_history.append({
                "attempt": attempt,
                "is_valid": is_valid,
                "score": quality_score,
                "reason": reason,
                "mismatches": verifier_info.get("mismatches", []),
                "verifier_model": verifier_info.get("model", "unknown"),
            })

            if is_valid and quality_score >= quality_threshold:
                accepted = True
                last_result.quality_score = quality_score
                break
            else:
                last_error = f"Attempt {attempt}/{max_retries} validation failed: {reason}"

        if not accepted:
            # PRD Section 21: Poor digitisation routes to human review
            return StageExecutionResult(
                status="REVIEW_REQUIRED",
                input_refs={"crop_uri": crop_uri},
                output_refs={"attempts": max_retries, "reason": last_error, "verification_history": verification_history},
                input_hash=input_hash,
                model=provider.model_name,
                model_version=provider.model_version,
                algorithm_version="digitise_v1",
                error=last_error,
                quality_status="REVIEW_REQUIRED",
            )

        # Store canonical image in object storage immutably
        canonical_key = f"canonical/{ctx.garment.tenant_id}/{ctx.garment.id}_canonical.jpg"
        canonical_uri = await ctx.storage.put_object(canonical_key, canonical_bytes, content_type="image/jpeg")

        # Create ImageAsset record for canonical image
        with Image.open(io.BytesIO(canonical_bytes)) as c_img:
            cw, ch = c_img.size
        canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()

        canonical_asset = ImageAsset(
            tenant_id=ctx.garment.tenant_id,
            member_id=ctx.garment.member_id,
            object_uri=canonical_uri,
            mime_type="image/jpeg",
            width=cw,
            height=ch,
            sha256=canonical_sha,
        )
        ctx.session.add(canonical_asset)
        await ctx.session.flush()

        # Link canonical image to garment without touching source image
        ctx.garment.canonical_image_id = canonical_asset.id

        prompt = getattr(provider, "_last_prompt", "")
        negative_prompt = getattr(provider, "_last_negative_prompt", "")

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"crop_uri": crop_uri},
            output_refs={
                "canonical_image_id": canonical_asset.id,
                "canonical_image_uri": canonical_uri,
                "quality_score": last_result.quality_score,
                "attempts": last_result.attempts,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "verification_history": verification_history,
            },
            input_hash=input_hash,
            model=last_result.model,
            model_version=last_result.model_version,
            algorithm_version="digitise_v1",
            quality_status="APPROVED",
        )
