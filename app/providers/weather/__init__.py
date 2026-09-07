"""Weather provider factory."""

from app.config import settings
from app.providers.base import BaseWeatherProvider


def get_weather_provider() -> BaseWeatherProvider:
    if settings.WEATHER_PROVIDER.lower() == "mock":
        from app.providers.weather.mock import MockWeatherProvider

        return MockWeatherProvider()
    from app.providers.weather.open_meteo import OpenMeteoWeatherProvider

    return OpenMeteoWeatherProvider()
