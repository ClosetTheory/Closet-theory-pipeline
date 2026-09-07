"""Deterministic weather fallback — used when Open-Meteo can't be reached/geocode the location,
or when WEATHER_PROVIDER=mock. Unlike this session's earlier attribute-extraction bug (a fake
fallback that silently reported itself as the real model), this is always honestly labeled via
`model="mock-weather-fallback"` so a caller can tell the difference — check that field rather
than trusting the numbers as real conditions."""

from datetime import datetime, timezone
from app.schemas.weather import WeatherSnapshot
from app.providers.base import BaseWeatherProvider


class MockWeatherProvider(BaseWeatherProvider):
    model_name = "mock-weather-fallback"

    async def get_weather(self, location: str) -> WeatherSnapshot:
        return WeatherSnapshot(
            location=location,
            resolved_place_name=location,
            temp_c=27.0,
            feels_like_c=29.0,
            humidity_pct=60.0,
            wind_kph=10.0,
            precipitation_mm=0.0,
            condition="Partly cloudy (fallback estimate — real weather lookup unavailable)",
            is_rainy=False,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            model=self.model_name,
        )
