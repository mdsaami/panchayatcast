"""Weather forecast and reanalysis data fetching module with caching and retry logic.

Fetches:
- GFS seamless forecast baseline via Open-Meteo Historical Forecast API
- ERA5-Land reanalysis reference via Open-Meteo Archive API
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
GFS_DIR = RAW_DATA_DIR / "gfs"
ERA5_DIR = RAW_DATA_DIR / "era5land"

GFS_DIR.mkdir(parents=True, exist_ok=True)
ERA5_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "PanchayatCast-WeatherDownscaling-Research/1.0"


def _request_with_retries(url: str, max_retries: int = 5, backoff_factor: float = 2.0, timeout: int = 30) -> Dict[str, Any]:
    """Execute HTTP GET request with exponential backoff retries."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    raw_bytes = response.read()
                    return json.loads(raw_bytes.decode("utf-8"))
                else:
                    raise RuntimeError(f"HTTP Status {response.status}: {response.reason}")
        except Exception as e:
            last_exception = e
            wait_time = backoff_factor ** attempt
            # print retry notice
            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                break
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {last_exception}")


def fetch_gfs_forecast(
    latitude: float,
    longitude: float,
    start_date: str = "2024-08-01",
    end_date: str = "2024-08-31",
    force_refresh: bool = False,
    cache_tag: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch GFS seamless forecast from Open-Meteo Historical Forecast API.
    
    Variables: temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, precipitation.
    Caches results locally to data/raw/gfs/.
    """
    tag = cache_tag or f"lat{latitude:.4f}_lon{longitude:.4f}"
    cache_file = GFS_DIR / f"gfs_{tag}_{start_date}_{end_date}.json"

    if not force_refresh and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation",
        "models": "gfs_seamless"
    }
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
    data = _request_with_retries(url)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def fetch_era5_reference(
    latitude: float,
    longitude: float,
    start_date: str = "2024-08-01",
    end_date: str = "2024-08-31",
    force_refresh: bool = False,
    cache_tag: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch ERA5-Land reanalysis reference from Open-Meteo Archive API.
    
    Variables: temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, precipitation.
    Caches results locally to data/raw/era5land/.
    """
    tag = cache_tag or f"lat{latitude:.4f}_lon{longitude:.4f}"
    cache_file = ERA5_DIR / f"era5land_{tag}_{start_date}_{end_date}.json"

    if not force_refresh and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation"
    }
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(params)}"
    data = _request_with_retries(url)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data
