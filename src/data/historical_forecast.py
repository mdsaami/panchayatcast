"""Historical forecast acquisition adapter."""

import pandas as pd
from typing import Dict, Any, Tuple
from src.data.fetch_weather import fetch_era5_reference
from src.data.canonical import to_canonical_forecast

def get_historical_reanalysis(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_tag: str = "benchmark"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch ERA5-Land reanalysis for historical benchmarking.
    
    PROVENANCE & SEMANTICS:
    - Source: Open-Meteo Archive API (ERA5-Land).
    - Type: Reanalysis, NOT initialized forecast runs. 
    - Limitations: Initialized runs for Nov 2013 are unavailable via single-runs API.
    - forecast_init_time: Fabricated as equal to valid_time purely for schema compatibility.
    - lead_hours: Fabricated as 0.0 purely for schema compatibility.
    - Grid resolution: 9km.
    - Usage: For station-level matching and pipeline validation only.
    """
    raw_data = fetch_era5_reference(
        latitude=latitude, 
        longitude=longitude, 
        start_date=start_date, 
        end_date=end_date,
        cache_tag=cache_tag
    )
    
    hourly = raw_data["hourly"]
    df = pd.DataFrame(hourly)
    df["valid_time"] = pd.to_datetime(df["time"], utc=True)
    df["forecast_init_time"] = df["valid_time"] # Explicitly documented as fabricated
    df["lead_hours"] = 0.0
    
    # Use ERA5 reanalysis proxy as station_id to avoid conflation
    df["station_id"] = "ERA5_REANALYSIS_PROXY" 
    
    df["latitude"] = raw_data.get("latitude", latitude)
    df["longitude"] = raw_data.get("longitude", longitude)
    
    # Rename variables to legacy gfs to reuse canonical adapter
    df = df.rename(columns={
        "temperature_2m": "temperature_2m_gfs",
        "relative_humidity_2m": "relative_humidity_2m_gfs",
        "surface_pressure": "surface_pressure_gfs",
        "wind_speed_10m": "wind_speed_10m_gfs",
        "precipitation": "precipitation_gfs",
    })
    
    df["panchayat_id"] = "CANDIDATE_PROXY" 
    df["block_id"] = "CANDIDATE_PROXY"
    
    canonical_df = to_canonical_forecast(df)
    
    metadata = {
        "source": "Open-Meteo ERA5-Land",
        "type": "reanalysis",
        "limitations": "Initialized historical runs unavailable for Nov 2013.",
        "disclaimer": "This is reanalysis data, not initialized forecasts. Init time and lead time are dummy values.",
        "start_date": start_date,
        "end_date": end_date
    }
    return canonical_df, metadata
