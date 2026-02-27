"""Weather data collection via NOAA API for outdoor game forecasts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from sports_predict.utils.cache import DataCache

logger = logging.getLogger(__name__)

# NFL stadium coordinates (lat, lon) for outdoor/retractable roof stadiums
NFL_OUTDOOR_STADIUMS: dict[str, tuple[float, float]] = {
    "BUF": (42.774, -78.787), "CHI": (41.862, -87.617), "CIN": (39.095, -84.516),
    "CLE": (41.506, -81.700), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
    "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "MIA": (25.958, -80.239),
    "NE": (42.091, -71.264), "NYG": (40.813, -74.074), "NYJ": (40.813, -74.074),
    "PHI": (39.901, -75.168), "PIT": (40.447, -80.016), "SF": (37.403, -121.970),
    "SEA": (47.595, -122.332), "TB": (27.976, -82.503), "TEN": (36.166, -86.771),
    "WAS": (38.908, -76.864), "BAL": (39.278, -76.623), "CAR": (35.226, -80.853),
}

# MLB stadiums (all are outdoor or retractable)
MLB_OUTDOOR_STADIUMS: dict[str, tuple[float, float]] = {
    "BOS": (42.346, -71.098), "NYY": (40.829, -73.926), "NYM": (40.757, -73.846),
    "PHI": (39.906, -75.167), "BAL": (39.284, -76.622), "WAS": (38.873, -77.007),
    "CHC": (41.948, -87.656), "CIN": (39.097, -84.507), "PIT": (40.447, -80.006),
    "STL": (38.623, -90.193), "CLE": (41.496, -81.685), "DET": (42.339, -83.049),
    "MIN": (44.982, -93.278), "KC": (39.051, -94.481), "CHW": (41.830, -87.634),
    "ATL": (33.891, -84.468), "COL": (39.756, -104.994), "LAD": (34.074, -118.240),
    "SF": (37.778, -122.389), "SD": (32.707, -117.157), "ARI": (33.445, -112.067),
    "OAK": (37.751, -122.201), "LAA": (33.800, -117.883), "SEA": (47.591, -122.332),
    "TEX": (32.751, -97.083),
}


@dataclass
class GameWeather:
    """Weather forecast for a game."""

    temperature_f: float
    wind_speed_mph: float
    wind_direction: str
    precipitation_pct: float
    humidity_pct: float
    conditions: str  # 'clear', 'rain', 'snow', etc.
    is_dome: bool = False


class WeatherCollector:
    """Fetches weather forecasts from NOAA Weather API."""

    def __init__(self, cache: DataCache | None = None):
        self.cache = cache
        self._client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "SportsPredict/0.1 (sports.predict@example.com)"},
        )

    def _get_forecast(self, lat: float, lon: float) -> dict | None:
        """Get forecast from NOAA Weather API."""
        try:
            # Step 1: Get the forecast grid endpoint for this location
            points_resp = self._client.get(
                f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            )
            points_resp.raise_for_status()
            points_data = points_resp.json()

            forecast_url = points_data["properties"]["forecast"]

            # Step 2: Get the actual forecast
            forecast_resp = self._client.get(forecast_url)
            forecast_resp.raise_for_status()
            return forecast_resp.json()

        except httpx.HTTPError as e:
            logger.warning("NOAA API error for (%.4f, %.4f): %s", lat, lon, e)
            return None

    def get_game_weather(
        self, team: str, sport: str, game_datetime: datetime
    ) -> GameWeather:
        """Get weather forecast for a specific game.

        Args:
            team: Home team abbreviation
            sport: 'nfl' or 'mlb'
            game_datetime: Scheduled game time
        """
        # Check if dome stadium
        stadiums = NFL_OUTDOOR_STADIUMS if sport == "nfl" else MLB_OUTDOOR_STADIUMS
        if team not in stadiums:
            return GameWeather(
                temperature_f=72.0,
                wind_speed_mph=0.0,
                wind_direction="N/A",
                precipitation_pct=0.0,
                humidity_pct=50.0,
                conditions="dome",
                is_dome=True,
            )

        lat, lon = stadiums[team]

        # Check cache
        cache_key = f"weather_{sport}_{team}_{game_datetime.strftime('%Y%m%d')}"
        if self.cache:
            cached = self.cache.get_json(cache_key)
            if cached:
                return GameWeather(**cached)

        forecast = self._get_forecast(lat, lon)
        if forecast is None:
            # Return neutral defaults if API fails
            return GameWeather(
                temperature_f=65.0,
                wind_speed_mph=5.0,
                wind_direction="N",
                precipitation_pct=10.0,
                humidity_pct=50.0,
                conditions="unknown",
            )

        # Find the forecast period closest to game time
        periods = forecast.get("properties", {}).get("periods", [])
        best_period = None
        best_diff = float("inf")

        for period in periods:
            start = datetime.fromisoformat(
                period["startTime"].replace("Z", "+00:00")
            )
            diff = abs((start - game_datetime.replace(tzinfo=start.tzinfo)).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_period = period

        if best_period is None:
            return GameWeather(
                temperature_f=65.0, wind_speed_mph=5.0, wind_direction="N",
                precipitation_pct=10.0, humidity_pct=50.0, conditions="unknown",
            )

        # Parse wind
        wind_str = best_period.get("windSpeed", "5 mph")
        try:
            wind_speed = float(wind_str.split()[0])
        except (ValueError, IndexError):
            wind_speed = 5.0

        # Determine conditions from short forecast
        short_forecast = best_period.get("shortForecast", "").lower()
        if "snow" in short_forecast:
            conditions = "snow"
        elif "rain" in short_forecast or "shower" in short_forecast:
            conditions = "rain"
        elif "cloud" in short_forecast or "overcast" in short_forecast:
            conditions = "cloudy"
        else:
            conditions = "clear"

        # Precipitation probability
        precip = best_period.get("probabilityOfPrecipitation", {})
        precip_pct = precip.get("value", 0) if precip else 0

        # Humidity
        humidity = best_period.get("relativeHumidity", {})
        humidity_pct = humidity.get("value", 50) if humidity else 50

        weather = GameWeather(
            temperature_f=float(best_period.get("temperature", 65)),
            wind_speed_mph=wind_speed,
            wind_direction=best_period.get("windDirection", "N"),
            precipitation_pct=float(precip_pct or 0),
            humidity_pct=float(humidity_pct or 50),
            conditions=conditions,
        )

        # Cache result
        if self.cache:
            from dataclasses import asdict
            self.cache.set_json(cache_key, asdict(weather))

        return weather

    def close(self):
        """Close HTTP client."""
        self._client.close()
