"""Fixed-run Open-Meteo ingestion for PanchayatCast.

Unlike the legacy Historical Forecast API fetcher, this module explicitly
records the model initialization time, valid time, and forecast lead time.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import validate_temporal_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_RUN_DIR = PROJECT_ROOT / "data" / "raw" / "forecast_runs"
RAW_RUN_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_RUN_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"

USER_AGENT = "PanchayatCast-Research/1.0"

DEFAULT_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]


def _request_json(
    url: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Fetch JSON from Open-Meteo."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Open-Meteo returned HTTP {response.status}"
            )

        return json.loads(response.read().decode("utf-8"))


def fetch_single_run(
    latitude: float,
    longitude: float,
    run: str,
    model: str = "gfs_global",
    forecast_hours: int = 120,
    hourly_variables: list[str] | None = None,
    timezone_name: str = "UTC",
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch one specific model run.

    Parameters
    ----------
    latitude, longitude:
        WGS84 coordinates.

    run:
        Model initialization time in UTC, e.g.
        ``2026-10-05T00:00``.

    model:
        Open-Meteo model identifier.

    forecast_hours:
        Number of forecast hours to retrieve.

    hourly_variables:
        Weather variables to retrieve.

    timezone_name:
        Keep UTC for research/backtesting to avoid ambiguity.

    force_refresh:
        Re-download even if cached.
    """

    if hourly_variables is None:
        hourly_variables = DEFAULT_HOURLY_VARIABLES.copy()

    run_dt = datetime.fromisoformat(run)

    if run_dt.tzinfo is None:
        run_dt = run_dt.replace(tzinfo=timezone.utc)

    run_dt = run_dt.astimezone(timezone.utc)

    run_string = run_dt.strftime("%Y-%m-%dT%H:%M")

    safe_run = run_string.replace(":", "")
    cache_file = (
        RAW_RUN_DIR
        / f"run_{model}_{latitude:.4f}_{longitude:.4f}_{safe_run}.json"
    )

    if cache_file.exists() and not force_refresh:
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "run": run_string,
        "models": model,
        "forecast_hours": forecast_hours,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone_name,
    }

    url = f"{SINGLE_RUN_ENDPOINT}?{urllib.parse.urlencode(params)}"

    data = _request_json(url)

    # Store metadata explicitly. Never infer forecast initialization
    # from the first valid_time later in the pipeline.
    data["_panchayatcast_metadata"] = {
        "forecast_init_time": run_string,
        "model": model,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
    }

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def normalize_single_run(
    raw: dict[str, Any],
) -> pd.DataFrame:
    """Convert an Open-Meteo single-run response into tabular records."""

    hourly = raw.get("hourly")
    metadata = raw.get("_panchayatcast_metadata", {})

    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response has no hourly time series.")

    forecast_init_time = metadata.get("forecast_init_time")

    if not forecast_init_time:
        raise ValueError(
            "forecast_init_time is missing. "
            "Never fabricate initialization time."
        )

    valid_times = pd.to_datetime(
        hourly["time"],
        utc=True,
    )

    init_time = pd.Timestamp(
        forecast_init_time,
        tz="UTC",
    )

    df = pd.DataFrame(
        {
            "forecast_init_time": init_time,
            "valid_time": valid_times,
        }
    )

    # Lead time is calculated only from explicit initialization
    # and valid times.
    df["lead_hours"] = (
        (df["valid_time"] - df["forecast_init_time"])
        .dt.total_seconds()
        / 3600.0
    )

    for variable, values in hourly.items():
        if variable == "time":
            continue

        if len(values) != len(df):
            raise ValueError(
                f"Length mismatch for variable '{variable}': "
                f"{len(values)} != {len(df)}"
            )

        df[variable] = values

    df["latitude"] = metadata.get("latitude")
    df["longitude"] = metadata.get("longitude")
    df["forecast_model"] = metadata.get("model")

    # Validate the temporal contract before data leaves the ingestion layer.
    # This catches initialization/valid-time/lead-time mismatches early.
    validate_temporal_columns(df)

    return df