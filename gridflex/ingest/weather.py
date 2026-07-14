"""
Open-Meteo weather client. No API key required.

- fetch_historical(): ERA5 reanalysis via /v1/archive — for training data.
- fetch_forecast(): live forecast via /v1/forecast — for production inference.

Both request ALL 20 PJM zones in a single HTTP call using Open-Meteo's
comma-separated multi-location support, rather than 20 separate requests.
"""

from __future__ import annotations

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from gridflex.config import (
    HOURLY_WEATHER_VARS,
    OPEN_METEO_ARCHIVE,
    OPEN_METEO_FORECAST,
    ZONE_COORDS,
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def _get(url: str, params: dict) -> dict | list:
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _multi_location_params() -> dict:
    zones = list(ZONE_COORDS.keys())
    lats = ",".join(str(ZONE_COORDS[z][0]) for z in zones)
    lons = ",".join(str(ZONE_COORDS[z][1]) for z in zones)
    return {"latitude": lats, "longitude": lons}, zones


def _parse_multi(response: dict | list, zones: list[str]) -> pd.DataFrame:
    """Open-Meteo returns a single object if one location was requested, or a
    list of objects (one per location, same order as input) if multiple were.
    We always request multiple, so normalize defensively anyway."""
    entries = response if isinstance(response, list) else [response]
    frames = []
    for zone, entry in zip(zones, entries, strict=True):
        hourly = entry.get("hourly", {})
        df = pd.DataFrame(hourly)
        if df.empty:
            continue
        df["subba"] = zone
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.rename(columns={"time": "period"})
    return out


def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """ERA5 reanalysis weather for all PJM zones. Dates as YYYY-MM-DD.
    Used for training data — NOT for live inference (see fetch_forecast)."""
    loc_params, zones = _multi_location_params()
    params = {
        **loc_params,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_WEATHER_VARS),
        "timezone": "UTC",
    }
    resp = _get(OPEN_METEO_ARCHIVE, params)
    return _parse_multi(resp, zones)


def fetch_forecast(past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
    """Live forecast (+ recent past) for all PJM zones — used in production
    inference, where we need weather *ahead* of now, which /v1/archive can't
    provide (it's historical-only)."""
    loc_params, zones = _multi_location_params()
    params = {
        **loc_params,
        "hourly": ",".join(HOURLY_WEATHER_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = _get(OPEN_METEO_FORECAST, params)
    return _parse_multi(resp, zones)
