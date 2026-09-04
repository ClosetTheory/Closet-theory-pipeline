"""Stage 2: Facial/Person Detection and Garment Crop (RetinaFace + SAM / OpenCV).

Localizes faces, isolates garment regions, generates clean crops, creates visual overlays, and stores crop refs.
"""

import io
from PIL import Image, ImageDraw
from sqlalchemy import select
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import GarmentState, PipelineStage
from app.providers.detection import get_detection_provider


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
        # that hint up (no face actually found); otherwise fall through and crop for real, same
        # as any FULL_BODY image.
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

        crop_refs = []
        annotated_uri = None

        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as pil_img:
            img_w, img_h = pil_img.size

            # Create annotated visualization image
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
                    # Pad a little beyond the model's predicted box — a tight/exact box
                    # routinely clips a sleeve edge, hem, or shoe tip, and the digitisation
                    # stage then has to draw that missing part from nothing.
                    box_w, box_h = box[2] - box[0], box[3] - box[1]
                    pad_x, pad_y = max(4, int(box_w * 0.06)), max(4, int(box_h * 0.06))
                    x1 = max(0, min(box[0] - pad_x, img_w - 1))
                    y1 = max(0, min(box[1] - pad_y, img_h - 1))
                    x2 = max(x1 + 1, min(box[2] + pad_x, img_w))
                    y2 = max(y1 + 1, min(box[3] + pad_y, img_h))

                    # Crop the isolated garment region
                    cropped = pil_img.crop((x1, y1, x2, y2))
                    buf = io.BytesIO()
                    cropped.save(buf, format="JPEG", quality=95)
                    crop_data = buf.getvalue()

                    crop_key = f"crops/{ctx.garment.tenant_id}/{ctx.garment.id}_{region.label}_{idx}.jpg"
                    crop_uri = await ctx.storage.put_object(crop_key, crop_data, content_type="image/jpeg")
                    crop_refs.append(crop_uri)

                    # Draw garment bounding box on annotated image in electric blue
                    draw.rectangle([x1, y1, x2, y2], outline=(59, 130, 246), width=4)
                    draw.rectangle([x1, max(0, y1 - 22), x1 + 120, y1], fill=(59, 130, 246))
                    draw.text((x1 + 5, max(0, y1 - 18)), region.label.upper(), fill=(255, 255, 255))
            else:
                crop_refs.append(source_image.object_uri)

            # Save annotated image for frontend visual feedback
            ann_buf = io.BytesIO()
            annotated_img.save(ann_buf, format="JPEG", quality=90)
            ann_key = f"crops/{ctx.garment.tenant_id}/{ctx.garment.id}_annotated.jpg"
            annotated_uri = await ctx.storage.put_object(ann_key, ann_buf.getvalue(), content_type="image/jpeg")

        # The first region is this garment's own crop (unchanged single-garment behavior).
        # Every additional detected garment becomes its own sibling Garment row — each is a
        # genuinely distinct item that needs its own attributes/embedding, not a sub-part of
        # this one. Not enqueued here: the orchestrator only commits once, at the very end of
        # the whole run — enqueuing before that commit would let the worker's separate DB
        # session try to load a row that isn't durable yet. See PipelineOrchestrator.run().
        ctx.garment.garment_crop_refs = crop_refs[:1]
        spawned_garment_ids = []

        # Idempotency: re-running this stage (e.g. a retry/force re-run, or the live-demo
        # /step endpoint being invoked twice) must not spawn a second batch of duplicate
        # siblings for regions already split out. Crop URIs are deterministic (derived from
        # this garment's own id + label + index), so an existing sibling with the same crop
        # ref is the same detected region, not a new one.
        existing_siblings_stmt = select(Garment.garment_crop_refs).where(
            Garment.source_image_id == ctx.garment.source_image_id,
            Garment.id != ctx.garment.id,
        )
        existing_refs = {ref for (refs,) in (await ctx.session.execute(existing_siblings_stmt)).all() if refs for ref in refs}

        for sibling_crop_uri in crop_refs[1:]:
            if sibling_crop_uri in existing_refs:
                continue
            sibling = Garment(
                tenant_id=ctx.garment.tenant_id,
                member_id=ctx.garment.member_id,
                source_image_id=ctx.garment.source_image_id,
                image_type=ImageType.CROP.value,
                garment_crop_refs=[sibling_crop_uri],
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
                "garment_crop_refs": crop_refs,
                "annotated_overlay_uri": annotated_uri,
                "spawned_garment_ids": spawned_garment_ids,
            },
            input_hash=input_hash,
            model=detection.model,
            model_version=detection.model_version,
            algorithm_version="crop_v1",
        )
