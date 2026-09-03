"""Stage 9: Visual Compatibility Analysis.

Extracts visual harmony features; evaluates deterministic rules with VLM fallback.
Finalizes the canonical garment ingestion pipeline.
"""

from app.config import settings
from app.models.compatibility import CompatibilityResult
from app.pipeline.idempotency import compute_stage_input_hash
from app.pipeline.stages.base import BaseStage, StageExecutionContext, StageExecutionResult
from app.pipeline.state_machine import PipelineStage
from app.providers.vlm import get_vlm_provider
from app.rules.visual import evaluate_visual_rules, VISUAL_RULE_VERSION


class Stage09Visual(BaseStage):
    stage_name = PipelineStage.STAGE_09_VISUAL.value

    async def execute(self, ctx: StageExecutionContext) -> StageExecutionResult:
        attrs = ctx.garment.attributes_json or {}
        colors = attrs.get("colour", [])
        pattern = attrs.get("pattern", "solid")
        occasions = attrs.get("occasion", [])

        input_hash = compute_stage_input_hash({
            "colors": colors,
            "pattern": pattern,
            "occasions": occasions,
        })

        features = dict(ctx.garment.compatibility_features or {})
        features["visual"] = {
            "colors": colors,
            "pattern": pattern,
            "occasions": occasions,
            "versatility": attrs.get("versatility", 0.5),
        }
        ctx.garment.compatibility_features = features

        model_name = None
        model_version = None

        # Pairwise visual compatibility evaluation if compare_with_garment is present
        compare_garment = ctx.context_data.get("compare_garment")
        if compare_garment:
            confident, decision, score, reason, ver = evaluate_visual_rules(
                attrs, compare_garment.attributes_json
            )

            # PRD Section 16: If deterministic rules are not confident, invoke VLM fallback
            if not confident:
                vlm = get_vlm_provider()
                model_name = settings.VLM_MODEL_NAME
                model_version = settings.VLM_MODEL_VERSION
                decision, score, reason = await vlm.evaluate_visual_compatibility(
                    None, None, attrs, compare_garment.attributes_json
                )

            comp_rec = CompatibilityResult(
                garment_a=ctx.garment.id,
                garment_b=compare_garment.id,
                compatibility_type="VISUAL",
                decision=decision,
                score=score,
                reason=reason,
                algorithm_version=ver,
                model_version=f"{model_name}:{model_version}" if model_name else None,
            )
            ctx.session.add(comp_rec)
            await ctx.session.flush()

        # Update final garment status if pipeline completed cleanly
        if ctx.garment.quality_status != "REVIEW_REQUIRED":
            ctx.garment.quality_status = "APPROVED"
            ctx.garment.status = "COMPLETED"

        return StageExecutionResult(
            status="SUCCEEDED",
            input_refs={"colors": colors, "pattern": pattern},
            output_refs=features["visual"],
            input_hash=input_hash,
            algorithm_version=VISUAL_RULE_VERSION,
            model=model_name,
            model_version=model_version,
            quality_status=ctx.garment.quality_status,
        )
