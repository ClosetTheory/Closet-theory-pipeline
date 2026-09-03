"""Stage 2: Facial/Person Detection and Garment Crop (RetinaFace + SAM / OpenCV).

Localizes faces, isolates garment regions, generates clean crops, creates visual overlays, and stores crop refs.
"""

import io
from PIL import Image, ImageDraw
from app.models.image_asset import ImageAsset
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.detection import get_detection_provider


class Stage02Crop(BaseStage):
    stage_name = PipelineStage.STAGE_02_CROP.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        source_image = ctx.garment.source_image
        image_bytes = await ctx.storage.get_object(source_image.object_uri)
        input_hash = compute_stage_input_hash(image_bytes)

        provider = get_detection_provider()
        detection = await provider.detect_and_crop(image_bytes)

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
                    x1 = max(0, min(box[0], img_w - 1))
                    y1 = max(0, min(box[1], img_h - 1))
                    x2 = max(x1 + 1, min(box[2], img_w))
                    y2 = max(y1 + 1, min(box[3], img_h))

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

        ctx.garment.garment_crop_refs = crop_refs

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"source_image_id": source_image.id},
            output_refs={
                "person_detected": detection.person_detected,
                "face_box": detection.face_box,
                "garment_crop_refs": crop_refs,
                "annotated_overlay_uri": annotated_uri,
            },
            input_hash=input_hash,
            model=detection.model,
            model_version=detection.model_version,
            algorithm_version="crop_v1",
        )
