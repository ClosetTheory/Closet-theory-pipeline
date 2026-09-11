"""Unit tests for the aesthetics orchestration module (Option A) — isolated from the live
ranking pipeline and from any real network call, using MockAestheticProvider (which derives
its score from the same pairwise colour-theory rules Option B added to app/rules/visual.py).
"""

import pytest
from app.models.garment import Garment
from app.providers.aesthetic.mock import MockAestheticProvider
from app.schemas.styling import OutfitCandidate, StylingContext, StylingIntent
import app.styling.aesthetics as aesthetics_module


def _garment(garment_id: str, colour: str, subcategory: str = "top") -> Garment:
    return Garment(
        id=garment_id,
        tenant_id="tenant-1",
        member_id="member-1",
        source_image_id="img-1",
        category="apparel",
        subcategory=subcategory,
        attributes_json={"pattern": "solid", "colour": [colour], "occasion": ["casual"]},
        status="COMPLETED",
        quality_status="PASS",
    )


def _context() -> StylingContext:
    return StylingContext(intent=StylingIntent(occasion="casual"))


@pytest.fixture(autouse=True)
def _use_mock_provider(monkeypatch):
    monkeypatch.setattr(aesthetics_module, "get_aesthetic_provider", lambda: MockAestheticProvider())


@pytest.mark.asyncio
async def test_scores_every_candidate_and_keys_by_sorted_garment_ids():
    top = _garment("garm_top", "navy blue", "top")
    bottom = _garment("garm_bottom", "mustard", "bottom")
    candidate = OutfitCandidate(garment_ids=["garm_top", "garm_bottom"], roles={"garm_top": "TOP", "garm_bottom": "BOTTOM"})
    garments_by_id = {"garm_top": top, "garm_bottom": bottom}

    results = await aesthetics_module.score_outfits_aesthetics([candidate], _context(), garments_by_id)

    key = aesthetics_module._candidate_key(candidate)
    assert key in results
    result = results[key]
    assert 0.0 <= result.score <= 1.0
    assert result.model == "mock-aesthetic-provider"


@pytest.mark.asyncio
async def test_analogous_colors_score_higher_than_arbitrary_mismatched_colors():
    good_top = _garment("garm_good_top", "teal", "top")
    good_bottom = _garment("garm_good_bottom", "turquoise", "bottom")
    good = OutfitCandidate(garment_ids=["garm_good_top", "garm_good_bottom"], roles={})

    bad_top = _garment("garm_bad_top", "mystery shade one", "top")
    bad_bottom = _garment("garm_bad_bottom", "mystery shade two", "bottom")
    bad = OutfitCandidate(garment_ids=["garm_bad_top", "garm_bad_bottom"], roles={})

    garments_by_id = {
        "garm_good_top": good_top, "garm_good_bottom": good_bottom,
        "garm_bad_top": bad_top, "garm_bad_bottom": bad_bottom,
    }

    results = await aesthetics_module.score_outfits_aesthetics([good, bad], _context(), garments_by_id)

    good_score = results[aesthetics_module._candidate_key(good)].score
    bad_score = results[aesthetics_module._candidate_key(bad)].score
    assert good_score > bad_score


@pytest.mark.asyncio
async def test_single_garment_outfit_gets_neutral_score_without_error():
    single = _garment("garm_solo", "black", "one_piece")
    candidate = OutfitCandidate(garment_ids=["garm_solo"], roles={"garm_solo": "ONE_PIECE"})

    results = await aesthetics_module.score_outfits_aesthetics([candidate], _context(), {"garm_solo": single})

    result = results[aesthetics_module._candidate_key(candidate)]
    assert result.score == 0.75
