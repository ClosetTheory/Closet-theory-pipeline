"""Stage 6: Category Bundling.

Maps subcategory to canonical category via deterministic lookup table.
No LLMs, strictly versioned, unknown subcategories route to human review.
"""

from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.rules.taxonomy import bundle_category


class Stage06Category(BaseStage):
    stage_name = PipelineStage.STAGE_06_CATEGORY.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        subcategory = ctx.garment.subcategory or ""
        input_hash = compute_stage_input_hash({"subcategory": subcategory})

        canonical_category, taxonomy_version, requires_review = bundle_category(subcategory)

        if requires_review or canonical_category is None:
            # PRD Section 13: Unknown category goes to explicit fallback/review
            return StageExecutionResult(
                status="REVIEW_REQUIRED",
                input_refs={"subcategory": subcategory},
                output_refs={},
                input_hash=input_hash,
                algorithm_version=taxonomy_version,
                error=f"Unrecognized subcategory '{subcategory}' requires taxonomy review.",
                quality_status="REVIEW_REQUIRED",
            )

        # Update garment category
        ctx.garment.category = canonical_category

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"subcategory": subcategory},
            output_refs={
                "category": canonical_category,
                "subcategory": subcategory,
                "taxonomy_version": taxonomy_version,
            },
            input_hash=input_hash,
            algorithm_version=taxonomy_version,
        )
