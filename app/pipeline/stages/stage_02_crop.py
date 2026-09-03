"""Stage 2: Facial/Person Detection and Garment Crop (RetinaFace + SAM).

Localizes faces, isolates garment regions, generates clean crops, and stores crop refs.
"""

import io
from PIL import Image
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
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            img_w, img_h = pil_img.size

            if detection.garment_regions:
                for idx, region in enumerate(detection.garment_regions):
                    box = region.box  # [x1, y1, x2, y2]
                    # Clamp box to valid boundaries
                    x1 = max(0, min(box[0], img_w - 1))
                    y1 = max(0, min(box[1], img_h - 1))
                    x2 = max(x1 + 1, min(box[2], img_w))
                    y2 = max(y1 + 1, min(box[3], img_h))

                    cropped = pil_img.crop((x1, y1, x2, y2))
                    buf = io.BytesIO()
                    cropped.save(buf, format="JPEG", quality=95)
                    crop_data = buf.getvalue()

                    crop_key = f"crops/{ctx.garment.tenant_id}/{ctx.garment.id}_{region.label}_{idx}.jpg"
                    crop_uri = await ctx.storage.put_object(crop_key, crop_data, content_type="image/jpeg")
                    crop_refs.append(crop_uri)
            else:
                # Flat-lay or catalog fallback: use source image ref
                crop_refs.append(source_image.object_uri)

        ctx.garment.garment_crop_refs = crop_refs

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"source_image_id": source_image.id},
            output_refs={
                "person_detected": detection.person_detected,
                "face_box": detection.face_box,
                "garment_crop_refs": crop_refs,
            },
            input_hash=input_hash,
            model=detection.model,
            model_version=detection.model_version,
            algorithm_version="crop_v1",
        )
