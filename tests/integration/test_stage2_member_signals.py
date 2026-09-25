"""Stage 2 of the styling pipeline reads a character's stated profile, and the member-context
summary no longer learns from Outfit-of-the-Day's own synthesised prompts."""

from datetime import date

import pytest

from app.models.base import generate_uuid
from app.models.garment import Garment
from app.models.image_asset import ImageAsset
from app.models.ootd import OutfitOfTheDay
from app.models.persona import Persona
from app.models.styling import StylingRequest
from app.models.user import User
from app.styling.filtering import apply_hard_constraints
from app.styling.member_context import NO_HISTORY_NOTE, derive_member_context, load_recent_asks
from app.styling.member_signals import load_member_signals

WINTER = {
    "season": "Winter", "season_sub": "Deep Winter", "undertone": "cool", "depth": "deep",
    "temperature": "cool", "contrast": "high",
    "palette": ["#1b2a49", "#c8102e", "#00674f", "#f7f7f5"],
    "summary": "A cool-toned deep Winter best flattered by true red, ink navy and emerald.",
}


async def _character(session, slug="riya_test"):
    backing_id = generate_uuid("user")
    session.add(User(
        id=backing_id, email=f"{slug}@personas.closettheory.local", display_name=slug,
        password_hash="!locked-persona-account", gender="women", tenant_id=backing_id, member_id=backing_id,
    ))
    persona = Persona(
        slug=slug, user_id=backing_id, display_name="Riya Test", gender="women", city="kochi",
        body_shape="triangle", bio="test character",
        color_analysis=WINTER,
        preferences={"fits_loved": ["relaxed"], "colour_preferences": {"love": ["navy", "red"], "avoid": ["mustard"]}},
        hard_constraints=["no_heels", "quick_dry_only", "needs_pockets"],
        weekly_plan={"source": "panel_authored", "days": [{"day": "mon", "tags": ["work"], "note": "Ward rounds"}]},
        climate={"zone": "equatorial_humid_southwest"},
    )
    session.add(persona)
    await session.commit()
    return persona


async def _garment(session, owner_id, gid, **attrs):
    asset = ImageAsset(id=f"img_{gid}", tenant_id=owner_id, member_id=owner_id, object_uri=f"raw/{gid}.jpg",
                       mime_type="image/jpeg", width=10, height=10, sha256=f"sha_{gid}")
    session.add(asset)
    g = Garment(id=gid, tenant_id=owner_id, member_id=owner_id, source_image_id=asset.id,
                category=attrs.get("category"), subcategory=attrs.get("subcategory"),
                status="COMPLETED", quality_status="APPROVED", attributes_json=attrs)
    session.add(g)
    return g


@pytest.mark.asyncio
async def test_a_character_backed_account_loads_its_stated_profile(db_session):
    persona = await _character(db_session)
    signals = await load_member_signals(db_session, persona.user_id, persona.user_id)
    assert signals.source == "persona"
    assert signals.color_analysis["season_sub"] == "Deep Winter"
    assert signals.hard_constraints == ["no_heels", "quick_dry_only", "needs_pockets"]
    assert signals.weekly_plan["days"][0]["tags"] == ["work"]

    ordinary = await load_member_signals(db_session, "tenant_nobody", "tenant_nobody")
    assert ordinary.is_empty


@pytest.mark.asyncio
async def test_member_context_ignores_ootd_prompts_but_quotes_real_asks(db_session):
    persona = await _character(db_session)
    uid = persona.user_id

    # 1. An OOTD-generated request, linked from the OOTD row.
    ootd_req = StylingRequest(tenant_id=uid, member_id=uid, raw_text="Suggest today's outfit. This member most often…",
                              normalized_intent={"occasion": "general", "formality": "CASUAL"})
    db_session.add(ootd_req)
    await db_session.flush()
    db_session.add(OutfitOfTheDay(tenant_id=uid, member_id=uid, for_date=date(2026, 9, 25), location="Kochi",
                                  context_used="…", weather_snapshot={}, styling_request_id=ootd_req.id,
                                  generation_source="nightly_cron"))
    # 2. An OOTD prompt whose row was never written (crash between the two writes).
    db_session.add(StylingRequest(tenant_id=uid, member_id=uid, raw_text="Suggest today's outfit. No styling history yet…",
                                  normalized_intent={"occasion": "general", "formality": "CASUAL"}))
    # 3. What the member actually asked for, twice, plus an anchor-only request with no text.
    for _ in range(2):
        db_session.add(StylingRequest(tenant_id=uid, member_id=uid, raw_text="Something for a wedding reception, not too flashy",
                                      normalized_intent={"occasion": "wedding", "formality": "FORMAL"}))
    db_session.add(StylingRequest(tenant_id=uid, member_id=uid, raw_text=None, anchor_garment_ids=["g1"],
                                  normalized_intent={"occasion": "wedding"}))
    await db_session.commit()

    signals = await load_member_signals(db_session, uid, uid)
    summary = await derive_member_context(db_session, uid, uid, signals=signals, today=date(2026, 9, 28))  # a Monday

    assert "wedding occasions" in summary                      # learned from the member's own asks
    assert "general occasions" not in summary                  # not from OOTD's prompts
    assert '"Something for a wedding reception' in summary     # quoted back
    assert "Colour analysis: A cool-toned deep Winter" in summary and "loves navy, red" in summary
    assert "no heels" in summary and "needs pockets" in summary
    assert "Today's plan: work — Ward rounds" in summary

    asks = await load_recent_asks(db_session, uid, uid)
    assert asks and all(a.startswith("Something for a wedding") for a in asks)


@pytest.mark.asyncio
async def test_a_character_with_no_history_still_gets_a_profile_paragraph(db_session):
    persona = await _character(db_session, slug="quiet_test")
    signals = await load_member_signals(db_session, persona.user_id, persona.user_id)
    summary = await derive_member_context(db_session, persona.user_id, persona.user_id, signals=signals)
    assert summary != NO_HISTORY_NOTE and "Colour analysis" in summary
    # and an ordinary member with nothing at all keeps the old placeholder
    assert await derive_member_context(db_session, "t_none", "t_none") == NO_HISTORY_NOTE


@pytest.mark.asyncio
async def test_hard_constraints_drop_violating_candidates_but_never_to_zero(db_session):
    persona = await _character(db_session, slug="filter_test")
    uid = persona.user_id
    heels = await _garment(db_session, uid, "g_heels", category="FOOTWEAR", subcategory="heels", material="leather")
    flats = await _garment(db_session, uid, "g_flats", category="FOOTWEAR", subcategory="flats", material="canvas")
    wool = await _garment(db_session, uid, "g_wool", category="OUTERWEAR", subcategory="coat", material="wool")
    await db_session.commit()

    result = apply_hard_constraints([heels, flats, wool], ["no_heels", "quick_dry_only", "needs_pockets"])
    assert [g.id for g in result.kept] == ["g_flats"]
    assert result.dropped == 2 and not result.fell_back
    # leather heels break both rules; the trace names every one
    assert result.violations == {"g_heels": ["no_heels", "quick_dry_only"], "g_wool": ["quick_dry_only"]}

    # anchors survive with the violation named
    result = apply_hard_constraints([heels, flats], ["no_heels"], exempt_ids={"g_heels"})
    assert [g.id for g in result.kept] == ["g_heels", "g_flats"] and result.violations == {"g_heels": ["no_heels"]}

    # a wardrobe where everything violates is surfaced, not emptied
    result = apply_hard_constraints([heels, wool], ["no_heels", "no_wool"])
    assert [g.id for g in result.kept] == ["g_heels", "g_wool"] and result.fell_back
