"""Stage 8: Structural Compatibility Analysis.

Extracts slot assignment, silhouette, and fit parameters for outfit construction.
"""

from app.models.compatibility import CompatibilityResult
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.rules.structural import evaluate_structural_compatibility, STRUCTURAL_RULE_VERSION


class Stage08Structure(BaseStage):
    stage_name = PipelineStage.STAGE_08_STRUCTURE.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        attrs = ctx.garment.attributes_json or {}
        category = ctx.garment.category or ""
        fit = attrs.get("fit", "regular")
        silhouette = attrs.get("silhouette", "straight")

        input_hash = compute_stage_input_hash({
            "category": category,
            "fit": fit,
            "silhouette": silhouette,
        })

        features = dict(ctx.garment.compatibility_features or {})
        features["structure"] = {
            "slot": category,
            "fit": fit,
            "silhouette": silhouette,
            "sleeve_length": attrs.get("sleeve_length", "not_applicable"),
        }
        ctx.garment.compatibility_features = features

        # Pairwise structural evaluation if compare_with_garment is present
        compare_garment = ctx.context_data.get("compare_garment")
        if compare_garment:
            decision, score, reason, ver = evaluate_structural_compatibility(
                attrs, compare_garment.attributes_json
            )
            comp_rec = CompatibilityResult(
                garment_a=ctx.garment.id,
                garment_b=compare_garment.id,
                compatibility_type="STRUCTURAL",
                decision=decision,
                score=score,
                reason=reason,
                algorithm_version=ver,
            )
            ctx.session.add(comp_rec)
            await ctx.session.flush()

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"category": category, "fit": fit},
            output_refs=features["structure"],
            input_hash=input_hash,
            algorithm_version=STRUCTURAL_RULE_VERSION,
        )
