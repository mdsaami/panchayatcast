"""Feature engineering definitions and transformations for downscaling experiments."""

import math
import numpy as np
import pandas as pd
from typing import List, Dict

FEATURES_WEATHER = [
    "temperature_2m_gfs",
    "relative_humidity_2m_gfs",
    "surface_pressure_gfs",
    "wind_speed_10m_gfs",
    "precipitation_gfs"
]

FEATURES_TIME = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos"
]

# Historical basic elevation feature (Model B1) - causes spatial overfitting
FEATURES_TERRAIN_POINT_ELEV = [
    "elevation_local"
]

# Physically regularized metric terrain features
FEATURES_TERRAIN_FULL = [
    "elevation_local",
    "elevation_diff",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "relief_500m",
    "relief_1000m",
    "elev_std_1000m"
]

# Literature-backed physical terrain features (resolves Elevation Paradox)
# Features are strictly relative or dimensionless to avoid station memorization
FEATURES_PHYSICAL_TERRAIN = [
    "lapse_rate_correction",
    "tpi_ratio",
    "slope_rad",
    "aspect_sin",
    "aspect_cos",
    "elevation_gradient_mag"
]

FEATURES_SURFACE = [
    "cropland_fraction",
    "builtup_fraction",
    "tree_fraction",
    "water_fraction",
    "bare_fraction"
]

FEATURES_NDVI = [
    "ndvi_aligned"
]

# Ablation Model Feature Sets
FEATURE_SETS: Dict[str, List[str]] = {
    "Model_1_Weather_Time": FEATURES_WEATHER + FEATURES_TIME,
    "Model_B1_Weather_Time_Elevation": FEATURES_WEATHER + FEATURES_TIME + FEATURES_TERRAIN_POINT_ELEV,
    "Model_2_Weather_Time_Terrain": FEATURES_WEATHER + FEATURES_TIME + FEATURES_TERRAIN_FULL,
    "Model_3_Weather_Time_Surface": FEATURES_WEATHER + FEATURES_TIME + FEATURES_SURFACE,
    "Model_4_Weather_Time_Terrain_Surface": FEATURES_WEATHER + FEATURES_TIME + FEATURES_TERRAIN_FULL + FEATURES_SURFACE,
    "Model_5_Full_Physiographic_NDVI": FEATURES_WEATHER + FEATURES_TIME + FEATURES_TERRAIN_FULL + FEATURES_SURFACE + FEATURES_NDVI,
    "Model_6_Physical_Terrain": FEATURES_WEATHER + FEATURES_TIME + FEATURES_PHYSICAL_TERRAIN
}


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical time representations for hour of day and day of year."""
    df = df.copy()
    timestamps = pd.to_datetime(df["time"])
    
    hours = timestamps.dt.hour
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)

    doys = timestamps.dt.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * doys / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doys / 365.25)

    return df


def add_physical_terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly relative and physically grounded topographic features.
    
    Literature references:
    - Sha et al. (2020): Environmental lapse rate delta_T = -6.5 K/km * delta_z
    - Meyer et al. (2018): Relative topographic indices generalize across spatial folds
    """
    df = df.copy()
    
    # 1. Delta elevation relative to surrounding 1000m buffer
    elev_local = df["elevation_local"] if "elevation_local" in df.columns else df.get("elevation_m", 0.0)
    elev_std = df.get("elev_std_1000m", 1.0).clip(lower=0.1)
    relief_1000m = df.get("relief_1000m", 0.0)
    
    # Topographic Position Index ratio (normalized local prominence)
    df["tpi_ratio"] = relief_1000m / elev_std
    
    # 2. Environmental Lapse Rate Correction (-6.5 °C per 1000m)
    # Using local relief as proxy for topographic micro-climate temperature lapse
    df["lapse_rate_correction"] = -0.0065 * relief_1000m
    
    # 3. Slope in radians
    slope_deg = df.get("slope_deg", 0.0)
    df["slope_rad"] = np.radians(slope_deg)
    
    # 4. Aspect representation
    if "aspect_sin" not in df.columns or "aspect_cos" not in df.columns:
        aspect_deg = df.get("aspect_deg", 0.0)
        df["aspect_sin"] = np.sin(np.radians(aspect_deg))
        df["aspect_cos"] = np.cos(np.radians(aspect_deg))
        
    # 5. Dimensionless elevation gradient magnitude
    df["elevation_gradient_mag"] = np.tan(df["slope_rad"])
    
    return df
