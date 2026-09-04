"""Stage 2: Facial/Person Detection and Garment Region Labeling.

Localizes faces, identifies distinct garment regions, and creates a visual overlay — but does
NOT pixel-crop them. Every garment (primary + spawned siblings) points at the same full source
photo; a `detected_label` field on each Garment record carries which garment it represents, so
Stage 3/4 can isolate the right item within the full photo instead of reasoning from a small,
decontextualized crop. Tested this session: full-image + label routing beat pixel-cropping on
both attribute accuracy and digitisation fidelity, because the model keeps true scene context
(real proportions, drape, adjacent-garment boundaries) — see the plan file for the comparisons.
"""

import io
from PIL import Image, ImageDraw
from sqlalchemy import select
from app.config import settings
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import GarmentState, PipelineStage
from app.providers.detection import get_detection_provider
from app.providers.verification import verify_region_presence


from app.schemas.pipeline import ImageType


class Stage02Crop(BaseStage):
    stage_name = PipelineStage.STAGE_02_CROP.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        source_image = ctx.garment.source_image
        image_bytes = await ctx.storage.get_object(source_image.object_uri)
        input_hash = compute_stage_input_hash(image_bytes)
        image_type = ctx.garment.image_type

        provider = get_detection_provider()
        detection = await provider.detect_and_crop(image_bytes)

        # A classifier label of CATALOG/CROP is a hint that the photo is already a clean
        # garment-only shot, not a guarantee — many "catalog" photos still show a person/face
        # (e.g. a model wearing the item). Only take the fast skip path when detection backs
        # that hint up (no face actually found); otherwise fall through, same as any FULL_BODY image.
        is_catalog_like = image_type in (ImageType.CATALOG.value, ImageType.CROP.value, "CATALOG", "CROP")
        if is_catalog_like and not detection.face_box:
            ctx.garment.garment_crop_refs = [source_image.object_uri]
            return StageExecutionResult(
                status="SUCCEEDED",
                input_refs={"source_image_id": source_image.id},
                output_refs={
                    "skipped": True,
                    "reason": f"Image classified as {image_type} and no face detected. Cropping skipped.",
                    "person_detected": detection.person_detected,
                    "face_box": None,
                    "garment_crop_refs": [source_image.object_uri],
                    "annotated_overlay_uri": source_image.object_uri,
                },
                input_hash=input_hash,
                model="Rule-Bypass",
                model_version="v1",
                algorithm_version="crop_skip_v1",
            )

        annotated_uri = None
        region_verifications = []
        kept_regions = []  # [(label, is_primary)], in detection order

        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as pil_img:
            img_w, img_h = pil_img.size

            # Create annotated visualization image (still useful for the demo UI — it's a
            # visualization aid, not an input to Stage 3/4, which now always work from the
            # full, uncropped photo).
            annotated_img = pil_img.copy()
            draw = ImageDraw.Draw(annotated_img)

            # Draw face box in bright green if present
            if detection.face_box:
                fx1, fy1, fx2, fy2 = detection.face_box
                draw.rectangle([fx1, fy1, fx2, fy2], outline=(16, 185, 129), width=4)
                draw.rectangle([fx1, max(0, fy1 - 22), fx1 + 80, fy1], fill=(16, 185, 129))
                draw.text((fx1 + 5, max(0, fy1 - 18)), "FACE", fill=(255, 255, 255))

            if detection.garment_regions:
                for idx, region in enumerate(detection.garment_regions):
                    box = region.box  # [x1, y1, x2, y2]
                    x1 = max(0, min(box[0], img_w - 1))
                    y1 = max(0, min(box[1], img_h - 1))
                    x2 = max(x1 + 1, min(box[2], img_w))
                    y2 = max(y1 + 1, min(box[3], img_h))

                    # Second-opinion verification (Gemini via settings.VISION_VERIFIER_MODEL —
                    # a different model than whichever produced this detection box) against the
                    # FULL photo: confirms this region actually corresponds to a real, visible
                    # garment, not a detector false-positive (the exact failure mode that once
                    # hallucinated a "belt" from an isolated crop of bare pavement — with full
                    # photo context, the model can correctly say "there's nothing there").
                    is_present, presence_score, presence_reason = await verify_region_presence(image_bytes, region.label)
                    is_primary = idx == 0
                    keep = is_present and presence_score >= settings.CROP_VERIFICATION_THRESHOLD
                    region_verifications.append({
                        "index": idx,
                        "label": region.label,
                        "is_valid": is_present,
                        "score": presence_score,
                        "reason": presence_reason,
                        "kept": keep or is_primary,
                    })

                    if not keep and not is_primary:
                        # A non-primary region that fails the second-opinion check is dropped
                        # entirely — never spawned as a sibling garment. The primary garment is
                        # always kept even if flagged, since downstream stages need *something*
                        # to work with; its REVIEW_REQUIRED-worthiness is left to Stage 3's own
                        # attribute-verification instead.
                        draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=3)
                        continue

                    kept_regions.append((region.label, is_primary))

                    # Draw garment bounding box on annotated image in electric blue
                    draw.rectangle([x1, y1, x2, y2], outline=(59, 130, 246), width=4)
                    draw.rectangle([x1, max(0, y1 - 22), x1 + 120, y1], fill=(59, 130, 246))
                    draw.text((x1 + 5, max(0, y1 - 18)), region.label.upper(), fill=(255, 255, 255))

            # Save annotated image for frontend visual feedback
            ann_buf = io.BytesIO()
            annotated_img.save(ann_buf, format="JPEG", quality=90)
            ann_key = f"crops/{ctx.garment.tenant_id}/{ctx.garment.id}_annotated.jpg"
            annotated_uri = await ctx.storage.put_object(ann_key, ann_buf.getvalue(), content_type="image/jpeg")

        # Every garment (this one + every spawned sibling) points at the SAME full source photo
        # — no pixel crop is stored. `detected_label` is what tells Stage 3/4 which garment to
        # isolate within that photo. Not enqueued here: the orchestrator only commits once, at
        # the very end of the whole run — enqueuing before that commit would let the worker's
        # separate DB session try to load a row that isn't durable yet. See PipelineOrchestrator.run().
        ctx.garment.garment_crop_refs = [source_image.object_uri]
        ctx.garment.detected_label = kept_regions[0][0] if kept_regions else None
        spawned_garment_ids = []

        # Idempotency: re-running this stage (e.g. a retry/force re-run, or the live-demo
        # /step endpoint being invoked twice) must not spawn a second batch of duplicate
        # siblings for regions already split out. Since every sibling now shares the same
        # garment_crop_refs, dedup is keyed on (source_image_id, detected_label) instead of
        # crop URI.
        existing_siblings_stmt = select(Garment.detected_label).where(
            Garment.source_image_id == ctx.garment.source_image_id,
            Garment.id != ctx.garment.id,
        )
        existing_labels = {label for (label,) in (await ctx.session.execute(existing_siblings_stmt)).all() if label}

        for sibling_label, is_primary in kept_regions[1:]:
            if sibling_label in existing_labels:
                continue
            sibling = Garment(
                tenant_id=ctx.garment.tenant_id,
                member_id=ctx.garment.member_id,
                source_image_id=ctx.garment.source_image_id,
                image_type=ImageType.CROP.value,
                garment_crop_refs=[source_image.object_uri],
                detected_label=sibling_label,
                status=GarmentState.CROPPED.value,
                quality_status="APPROVED",
            )
            ctx.session.add(sibling)
            await ctx.session.flush()
            spawned_garment_ids.append(sibling.id)

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"source_image_id": source_image.id},
            output_refs={
                "person_detected": detection.person_detected,
                "face_box": detection.face_box,
                "garment_crop_refs": [source_image.object_uri],
                "detected_label": ctx.garment.detected_label,
                "detected_labels": [label for label, _ in kept_regions],
                "annotated_overlay_uri": annotated_uri,
                "spawned_garment_ids": spawned_garment_ids,
                "region_verifications": region_verifications,
            },
            input_hash=input_hash,
            model=detection.model,
            model_version=detection.model_version,
            algorithm_version="crop_v2_no_pixel_crop",
        )
