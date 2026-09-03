"""Stage 7: Layering Compatibility Analysis.

Extracts layering stack capabilities and evaluates compatibility rules.
"""

from app.models.compatibility import CompatibilityResult
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.rules.layering import evaluate_layering_compatibility, LAYERING_RULE_VERSION


class Stage07Layering(BaseStage):
    stage_name = PipelineStage.STAGE_07_LAYERING.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        attrs = ctx.garment.attributes_json or {}
        role = attrs.get("layering_role", "standalone")
        warmth = attrs.get("warmth", 0.5)

        input_hash = compute_stage_input_hash({"layering_role": role, "warmth": warmth})

        # Precompute garment's intrinsic layering features
        features = dict(ctx.garment.compatibility_features or {})
        features["layering"] = {
            "role": role,
            "warmth": warmth,
            "permissible_inner": ["base"] if role in ("mid", "outer") else [],
            "permissible_outer": ["outer"] if role in ("base", "mid") else [],
        }
        ctx.garment.compatibility_features = features

        # Pairwise compatibility evaluation if compare_with_garment is present in context
        compare_garment = ctx.context_data.get("compare_garment")
        if compare_garment:
            decision, score, reason, ver = evaluate_layering_compatibility(
                attrs, compare_garment.attributes_json
            )
            comp_rec = CompatibilityResult(
                garment_a=ctx.garment.id,
                garment_b=compare_garment.id,
                compatibility_type="LAYERING",
                decision=decision,
                score=score,
                reason=reason,
                algorithm_version=ver,
            )
            ctx.session.add(comp_rec)
            await ctx.session.flush()

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"layering_role": role, "warmth": warmth},
            output_refs=features["layering"],
            input_hash=input_hash,
            algorithm_version=LAYERING_RULE_VERSION,
        )
