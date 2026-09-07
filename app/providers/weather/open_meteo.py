"""Real weather provider via Open-Meteo (https://open-meteo.com) — free, no API key required,
which matters here since this is one more real external dependency added purely to ground
outfit-of-the-day generation in actual conditions rather than a guess."""

from datetime import datetime, timezone
from typing import Optional
import httpx
from app.observability import logger
from app.providers.base import BaseWeatherProvider
from app.providers.weather.mock import MockWeatherProvider
from app.schemas.weather import WeatherSnapshot

# WMO weather interpretation codes (https://open-meteo.com/en/docs), collapsed to a short
# human-readable phrase — only the codes Open-Meteo's `current` block can actually return.
_WMO_CONDITIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}
_RAINY_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}


class OpenMeteoWeatherProvider(BaseWeatherProvider):
    """Geocodes a free-text place name (Open-Meteo's own geocoding API), then fetches current
    conditions for that coordinate (Open-Meteo's forecast API). Both are unauthenticated."""

    model_name = "open-meteo"

    def __init__(self):
        self._fallback = MockWeatherProvider()

    async def get_weather(self, location: str) -> WeatherSnapshot:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()
                results = geo_data.get("results") or []
                if not results:
                    raise ValueError(f"Could not geocode location '{location}'")
                place = results[0]
                lat, lon = place["latitude"], place["longitude"]
                resolved_name = ", ".join(
                    part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
                )

                weather_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                                   "precipitation,weather_code,wind_speed_10m",
                        "timezone": "auto",
                    },
                )
                weather_resp.raise_for_status()
                current = weather_resp.json().get("current", {})

            code = int(current.get("weather_code", 0))
            return WeatherSnapshot(
                location=location,
                resolved_place_name=resolved_name or None,
                temp_c=float(current.get("temperature_2m", 0.0)),
                feels_like_c=current.get("apparent_temperature"),
                humidity_pct=current.get("relative_humidity_2m"),
                wind_kph=current.get("wind_speed_10m"),
                precipitation_mm=current.get("precipitation"),
                condition=_WMO_CONDITIONS.get(code, "Unknown"),
                is_rainy=code in _RAINY_CODES,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                model=self.model_name,
            )
        except Exception as e:
            logger.warning(f"Open-Meteo weather fetch failed for '{location}': {e}. Falling back to mock weather.")
            return await self._fallback.get_weather(location)
