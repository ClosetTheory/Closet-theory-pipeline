"""Pipeline state machine definitions and transition order."""

from enum import Enum
from typing import Dict, List


class GarmentState(str, Enum):
    RECEIVED = "RECEIVED"
    CLASSIFIED = "CLASSIFIED"
    CROPPED = "CROPPED"
    ATTRIBUTES_EXTRACTED = "ATTRIBUTES_EXTRACTED"
    DIGITIZED = "DIGITIZED"
    EMBEDDED = "EMBEDDED"
    CATEGORY_BUNDLED = "CATEGORY_BUNDLED"
    LAYERING_ANALYZED = "LAYERING_ANALYZED"
    STRUCTURE_ANALYZED = "STRUCTURE_ANALYZED"
    VISUAL_ANALYZED = "VISUAL_ANALYZED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PipelineStage(str, Enum):
    STAGE_01_CLASSIFY = "STAGE_01_CLASSIFY"
    STAGE_02_CROP = "STAGE_02_CROP"
    STAGE_03_ATTRIBUTES = "STAGE_03_ATTRIBUTES"
    STAGE_04_DIGITISE = "STAGE_04_DIGITISE"
    STAGE_05_EMBED = "STAGE_05_EMBED"
    STAGE_06_CATEGORY = "STAGE_06_CATEGORY"
    STAGE_07_LAYERING = "STAGE_07_LAYERING"
    STAGE_08_STRUCTURE = "STAGE_08_STRUCTURE"
    STAGE_09_VISUAL = "STAGE_09_VISUAL"


# Sequential pipeline execution order
PIPELINE_STAGES_ORDER: List[PipelineStage] = [
    PipelineStage.STAGE_01_CLASSIFY,
    PipelineStage.STAGE_02_CROP,
    PipelineStage.STAGE_03_ATTRIBUTES,
    PipelineStage.STAGE_04_DIGITISE,
    PipelineStage.STAGE_05_EMBED,
    PipelineStage.STAGE_06_CATEGORY,
    PipelineStage.STAGE_07_LAYERING,
    PipelineStage.STAGE_08_STRUCTURE,
    PipelineStage.STAGE_09_VISUAL,
]

# Mapping from completed stage to garment state
STAGE_TO_GARMENT_STATE: Dict[PipelineStage, GarmentState] = {
    PipelineStage.STAGE_01_CLASSIFY: GarmentState.CLASSIFIED,
    PipelineStage.STAGE_02_CROP: GarmentState.CROPPED,
    PipelineStage.STAGE_03_ATTRIBUTES: GarmentState.ATTRIBUTES_EXTRACTED,
    PipelineStage.STAGE_04_DIGITISE: GarmentState.DIGITIZED,
    PipelineStage.STAGE_05_EMBED: GarmentState.EMBEDDED,
    PipelineStage.STAGE_06_CATEGORY: GarmentState.CATEGORY_BUNDLED,
    PipelineStage.STAGE_07_LAYERING: GarmentState.LAYERING_ANALYZED,
    PipelineStage.STAGE_08_STRUCTURE: GarmentState.STRUCTURE_ANALYZED,
    PipelineStage.STAGE_09_VISUAL: GarmentState.COMPLETED,  # All stages finished
}
