"""Dataset preparation and compilation module.

Fuses:
- GFS forecast baseline
- ERA5-Land reanalysis reference
- SRTM terrain derivatives
- ESA WorldCover surface fractions
- MODIS NDVI temporal alignment
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from src.data.fetch_weather import fetch_gfs_forecast, fetch_era5_reference
from src.terrain.srtm_extractor import extract_terrain_features
from src.surface.landcover import get_surface_features
from src.surface.ndvi_processor import get_aligned_ndvi
from src.features.pipeline import add_cyclical_time_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "locations.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_unified_dataset(
    start_date: str = "2024-08-01",
    end_date: str = "2024-08-31",
    force_refresh: bool = False
) -> pd.DataFrame:
    """Build or load the unified 15-location hourly dataset for August 2024."""
    parquet_path = PROCESSED_DIR / f"unified_dataset_{start_date}_{end_date}.parquet"
    csv_path = PROCESSED_DIR / f"unified_dataset_{start_date}_{end_date}.csv"

    if not force_refresh:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                pass

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        loc_cfg = json.load(f)

    all_dfs = []

    for loc in loc_cfg["locations"]:
        loc_id = loc["id"]
        loc_name = loc["name"]
        lat = loc["latitude"]
        lon = loc["longitude"]
        target_elev = loc["elevation_m"]

        # 1. Fetch GFS forecast
        gfs_raw = fetch_gfs_forecast(lat, lon, start_date=start_date, end_date=end_date, cache_tag=loc_id)
        gfs_hourly = gfs_raw["hourly"]
        gfs_elev = gfs_raw.get("elevation", target_elev)

        # 2. Fetch ERA5-Land reference
        era5_raw = fetch_era5_reference(lat, lon, start_date=start_date, end_date=end_date, cache_tag=loc_id)
        era5_hourly = era5_raw["hourly"]
        era5_elev = era5_raw.get("elevation", target_elev)

        # 3. Extract terrain derivatives
        terrain_feats = extract_terrain_features(lat, lon, loc_id=loc_id)

        # 4. Extract surface land cover
        surface_feats = get_surface_features(loc_id, lat, lon)

        # Base dataframe
        df = pd.DataFrame({
            "time": gfs_hourly["time"],
            "location_id": loc_id,
            "location_name": loc_name,
            "state": loc["state"],
            "zone": loc["zone"],
            "latitude": lat,
            "longitude": lon,
            "elevation_local": float(terrain_feats["elevation_local"]),
            "elevation_gfs": float(gfs_elev),
            "elevation_era5": float(era5_elev),
            "elevation_diff": float(terrain_feats["elevation_local"] - gfs_elev),
            # GFS Weather baseline
            "temperature_2m_gfs": gfs_hourly["temperature_2m"],
            "relative_humidity_2m_gfs": gfs_hourly["relative_humidity_2m"],
            "surface_pressure_gfs": gfs_hourly["surface_pressure"],
            "wind_speed_10m_gfs": gfs_hourly["wind_speed_10m"],
            "precipitation_gfs": gfs_hourly["precipitation"],
            # ERA5-Land reference
            "temperature_2m_era5": era5_hourly["temperature_2m"],
            "relative_humidity_2m_era5": era5_hourly["relative_humidity_2m"],
            "surface_pressure_era5": era5_hourly["surface_pressure"],
            "wind_speed_10m_era5": era5_hourly["wind_speed_10m"],
            "precipitation_era5": era5_hourly["precipitation"],
        })

        # Add terrain derivatives
        df["slope_deg"] = terrain_feats["slope_deg"]
        df["aspect_deg"] = terrain_feats["aspect_deg"]
        df["aspect_sin"] = terrain_feats["aspect_sin"]
        df["aspect_cos"] = terrain_feats["aspect_cos"]
        df["relief_500m"] = terrain_feats["relief_500m"]
        df["relief_1000m"] = terrain_feats["relief_1000m"]
        df["elev_std_1000m"] = terrain_feats["elev_std_1000m"]

        # Add surface features
        for k, v in surface_feats.items():
            df[k] = v

        # Add aligned NDVI
        df["ndvi_aligned"] = df["time"].apply(lambda t: get_aligned_ndvi(loc_id, t))

        # Target residual delta: T_refined = T_gfs + delta_T  ==> delta_T = T_era5 - T_gfs
        df["temp_residual"] = df["temperature_2m_era5"] - df["temperature_2m_gfs"]
        df["temp_error_gfs"] = df["temperature_2m_gfs"] - df["temperature_2m_era5"]

        all_dfs.append(df)

    unified = pd.concat(all_dfs, ignore_index=True)
    unified = add_cyclical_time_features(unified)

    # Save to disk
    unified.to_csv(csv_path, index=False)
    try:
        unified.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    return unified
