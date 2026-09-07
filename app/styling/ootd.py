"""Outfit-of-the-Day: a daily, weather-aware outfit pick grounded in this specific member's
real data — not a hand-picked "persona" from a fixed list.

Deliberately NOT a new recommendation engine — it builds one synthesized request_text (real
weather + a context paragraph derived from this member's actual styling history and wardrobe,
see app/styling/member_context.py) and runs it through the exact same StylingOrchestrator every
/recommendations call already uses, then caches the result per (member, calendar date,
location) so repeated checks the same day don't re-run the pipeline (and re-bill the
underlying vision/LLM calls). This is the single function used by both the on-demand API
endpoints (app/api/v1/styling.py) and the daily scheduled generation loop
(app/worker/ootd_scheduler.py) — same code path either way, only `generation_source` differs.
"""

from datetime import date as date_type, datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.ootd import OutfitOfTheDay
from app.models.styling import StylingRequest
from app.providers.weather import get_weather_provider
from app.schemas.styling import OutfitOfTheDayResponse, StylingRecommendationRequest
from app.schemas.weather import WeatherSnapshot
from app.storage.base import StorageClient
from app.styling.member_context import derive_member_context
from app.styling.orchestrator import StylingOrchestrator
from app.styling.replay import replay_styling_request


def _build_request_text(weather: WeatherSnapshot, member_context: str, extra_hint: Optional[str]) -> str:
    place = weather.resolved_place_name or weather.location
    feels_like = f" (feels like {weather.feels_like_c:.0f}°C)" if weather.feels_like_c is not None else ""
    humidity = f", humidity {weather.humidity_pct:.0f}%" if weather.humidity_pct is not None else ""
    rain_note = " There's a real chance of rain today, so factor that in." if weather.is_rainy else ""
    extra = f" Additional context for today specifically: {extra_hint}." if extra_hint else ""
    return (
        f"Suggest today's outfit. {member_context}{extra} "
        f"Current weather in {place}: {weather.temp_c:.0f}°C{feels_like}, {weather.condition}{humidity}."
        f"{rain_note} Pick something genuinely appropriate for this member and these real "
        f"conditions, not just aesthetically nice — comfort and practicality for today's actual "
        f"weather matter as much as looking good."
    )


async def get_or_generate_ootd(
    session: AsyncSession,
    storage: StorageClient,
    tenant_id: str,
    member_id: str,
    location: str,
    extra_hint: Optional[str] = None,
    force: bool = False,
    generation_source: str = "on_demand",
) -> OutfitOfTheDayResponse:
    """
    `extra_hint`: optional free-text the caller can add on top of the auto-derived context
    (e.g. "there's a client meeting today") — not a fixed set of choices, since a member's
    real context can't be reduced to picking one of a handful of labels. Leave it out entirely
    and this still works, fully automatically, from whatever real data exists for the member.
    """
    today = datetime.now(timezone.utc).date()

    if not force:
        existing_stmt = select(OutfitOfTheDay).where(
            OutfitOfTheDay.tenant_id == tenant_id,
            OutfitOfTheDay.member_id == member_id,
            OutfitOfTheDay.for_date == today,
            OutfitOfTheDay.location == location,
        )
        existing = (await session.execute(existing_stmt)).scalars().first()
        if existing:
            styling_request = await session.get(StylingRequest, existing.styling_request_id)
            styling_result = await replay_styling_request(session, styling_request)
            return OutfitOfTheDayResponse(
                date=today.isoformat(),
                location=location,
                context_used=existing.context_used,
                weather=WeatherSnapshot.model_validate(existing.weather_snapshot),
                styling=styling_result,
                cached=True,
                generation_source=existing.generation_source,
            )

    weather_provider = get_weather_provider()
    weather = await weather_provider.get_weather(location)
    member_context = await derive_member_context(session, tenant_id, member_id)
    request_text = _build_request_text(weather, member_context, extra_hint)

    orchestrator = StylingOrchestrator(session, storage)
    rec_request = StylingRecommendationRequest(request_text=request_text, top_k=3)
    styling_result = await orchestrator.run(rec_request, tenant_id, member_id)

    context_used = member_context + (f" [extra: {extra_hint}]" if extra_hint else "")

    # Upsert: force_regenerate on an existing day replaces that day's cache entry rather than
    # violating the (tenant, member, date, location) uniqueness constraint.
    existing_stmt = select(OutfitOfTheDay).where(
        OutfitOfTheDay.tenant_id == tenant_id,
        OutfitOfTheDay.member_id == member_id,
        OutfitOfTheDay.for_date == today,
        OutfitOfTheDay.location == location,
    )
    existing_row = (await session.execute(existing_stmt)).scalars().first()
    if existing_row:
        existing_row.context_used = context_used
        existing_row.weather_snapshot = weather.model_dump(mode="json")
        existing_row.styling_request_id = styling_result.request_id
        existing_row.generation_source = generation_source
    else:
        session.add(OutfitOfTheDay(
            tenant_id=tenant_id,
            member_id=member_id,
            for_date=today,
            location=location,
            context_used=context_used,
            weather_snapshot=weather.model_dump(mode="json"),
            styling_request_id=styling_result.request_id,
            generation_source=generation_source,
        ))
    await session.commit()

    return OutfitOfTheDayResponse(
        date=today.isoformat(),
        location=location,
        context_used=context_used,
        weather=weather,
        styling=styling_result,
        cached=False,
        generation_source=generation_source,
    )
