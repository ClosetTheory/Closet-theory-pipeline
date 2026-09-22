"""Maps Hopit's /v1/outfits:rank response into the same RankingTrace/OutfitCandidate shape
our own Stages 5-6 (retrieval + compatibility) produce, so Stages 7-10 run completely
unaware of which one produced their input — only Stages 5 and 6 branch on use_hopit, per
the Hopit tab spec ("only Stage 5 and Stage 6 of the original Styling pipeline should be
changed").
"""

from typing import Any, Dict, List
from app.schemas.styling import OutfitCandidate, ScoreBreakdown
from app.styling.ranking import RankingTrace

# Hopit's factor names -> our ScoreBreakdown fields, for the ones with a reasonable direct
# match (see docs/HOPIT_EVALUATION.md's sample response for the full factor list). Anything
# unmapped (e.g. Hopit-specific axes) is simply not represented in our score breakdown.
_FACTOR_MAP = {
    "compatibility": "compatibility",
    "occasion_fit": "occasion_fit",
    "weather_fit": "weather_fit",
    "novelty": "novelty",
    "preference_fit": "user_preference",
    "colour_harmony": "visual_harmony",
    "text_fit": "request_match",
}


def hopit_response_to_ranking_trace(hopit_result: Dict[str, Any]) -> RankingTrace:
    outfits: List[OutfitCandidate] = []
    for o in hopit_result.get("outfits", []):
        slots = o.get("slots") or {}
        roles = {garment_id: role for role, garment_id in slots.items()}
        garment_ids = list(slots.values()) or list(o.get("garment_ids", []))
        factors = o.get("factors") or {}
        score_kwargs: Dict[str, float] = {
            our_key: factors[hopit_key] for hopit_key, our_key in _FACTOR_MAP.items() if hopit_key in factors
        }
        score_kwargs["final_score"] = o.get("score", 0.0)

        reason = "Ranked by Hopit /v1/outfits:rank"
        if o.get("warnings"):
            reason += f" ({', '.join(o['warnings'])})"

        outfits.append(OutfitCandidate(
            garment_ids=garment_ids,
            roles=roles,
            compatibility_reason=reason,
            scores=ScoreBreakdown(**score_kwargs),
        ))

    return RankingTrace(
        outfits=outfits,
        total_evaluated=len(hopit_result.get("outfits", [])),
        total_compatible=len(outfits),
        pairing_rejected=0,
    )
