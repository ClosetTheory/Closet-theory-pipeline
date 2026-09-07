"""Weather snapshot schema shared by every weather provider (real and mock)."""

from typing import Optional
from pydantic import BaseModel, Field


class WeatherSnapshot(BaseModel):
    """A point-in-time weather reading for one location, used to ground outfit-of-the-day
    generation in real conditions rather than a guess."""

    location: str = Field(..., description="The location string the caller asked for, e.g. 'Mumbai'")
    resolved_place_name: Optional[str] = Field(
        default=None, description="The geocoded place name actually matched, e.g. 'Mumbai, Maharashtra, India'"
    )
    temp_c: float
    feels_like_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_kph: Optional[float] = None
    precipitation_mm: Optional[float] = None
    condition: str = Field(..., description="Human-readable condition, e.g. 'Partly cloudy'")
    is_rainy: bool = False
    fetched_at: str
    model: str = "unknown"
