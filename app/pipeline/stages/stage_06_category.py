"""Stage 6: Category Bundling.

Maps subcategory to canonical category via deterministic lookup table.
No LLMs, strictly versioned, unknown subcategories route to human review.
"""

from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.rules.garment_class import bundle_garment_class, infer_garment_class_from_subcategory


class Stage06Category(BaseStage):
    stage_name = PipelineStage.STAGE_06_CATEGORY.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        subcategory = ctx.garment.subcategory or ""
        garment_class = (ctx.garment.attributes_json or {}).get("garment_class") or infer_garment_class_from_subcategory(subcategory)
        input_hash = compute_stage_input_hash({"subcategory": subcategory, "garment_class": garment_class})

        canonical_category, taxonomy_version, requires_review = bundle_garment_class(garment_class)

        if requires_review or canonical_category is None:
            # SPEC.md Section 37: taxonomy mapping failure -> explicit fallback/review, never silent discard
            return StageExecutionResult(
                status="REVIEW_REQUIRED",
                input_refs={"subcategory": subcategory, "garment_class": garment_class},
                output_refs={},
                input_hash=input_hash,
                algorithm_version=taxonomy_version,
                error=f"Unrecognized garment_class '{garment_class}' requires taxonomy review.",
                quality_status="REVIEW_REQUIRED",
            )

        # Update garment category + garment_class
        ctx.garment.category = canonical_category
        ctx.garment.garment_class = garment_class

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"subcategory": subcategory, "garment_class": garment_class},
            output_refs={
                "category": canonical_category,
                "garment_class": garment_class,
                "subcategory": subcategory,
                "taxonomy_version": taxonomy_version,
            },
            input_hash=input_hash,
            algorithm_version=taxonomy_version,
        )
