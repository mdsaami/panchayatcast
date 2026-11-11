"""MOSDAC AWS observation ingestion adapter."""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MOSDAC_COLUMN_MAPPING = {
    "@STATION_ID": "station_id",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "ALTITUDE(m)": "elevation_m",
    "AIR_TEMP(°C)": "target_temperature",
    "HUMIDITY(%)": "observed_relative_humidity",
    "ATMO_PRESSURE(hpa)": "observed_surface_pressure",
    "WIND_SPEED(m/s)": "observed_wind_speed",
    "WIND_DIRECTION(deg)": "observed_wind_direction",
    "RAIN_FALL(mm)": "observed_precipitation",
}

def load_mosdac_aws_csv(path: str) -> pd.DataFrame:
    """Load MOSDAC AWS CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except Exception as e:
        logger.error(f"Failed to load MOSDAC CSV from {path}: {e}")
        raise

def normalize_mosdac_aws(df: pd.DataFrame, keep_raw: bool = False) -> pd.DataFrame:
    """Normalize MOSDAC columns into the canonical observation schema."""
    result = df.copy()
    
    # 1. Rename columns
    rename_map = {col: canonical for col, canonical in MOSDAC_COLUMN_MAPPING.items() if col in result.columns}
    
    if keep_raw:
        for col, canonical in rename_map.items():
            result[canonical] = result[col]
    else:
        result = result.rename(columns=rename_map)

    # Clean station_id by extracting just the station code before parentheses if needed
    if "station_id" in result.columns:
        result["station_name"] = result["station_id"].astype(str).str.extract(r'\((.*?)\)')
        result["station_id"] = result["station_id"].astype(str).str.replace(r'\(.*?\)', '', regex=True)
        result["station_id"] = result["station_id"].replace('nan', np.nan)

    # 2. Parse Timestamps
    # Verify timezone semantics: file has TIME(GMT) and DATE(GMT).
    if "DATE(GMT)" in result.columns and "TIME(GMT)" in result.columns:
        # Construct datetime string
        dt_str = result["DATE(GMT)"].astype(str) + " " + result["TIME(GMT)"].astype(str) + ":00:00"
        # Parse as UTC
        result["observation_time"] = pd.to_datetime(dt_str, format="%m/%d/%Y %H:%M:%S", errors="coerce").dt.tz_localize("UTC")
        if not keep_raw:
            result = result.drop(columns=["DATE(GMT)", "TIME(GMT)", "DATE(IST)", "TIME(IST)"], errors="ignore")
    
    # 3. Handle missing values
    # In some columns, 'null' might be read as string or NaN. Pandas handles 'null' as NaN usually, but let's make sure.
    result = result.replace({'null': np.nan, 'None': np.nan})

    # Ensure numeric columns
    numeric_cols = ["latitude", "longitude", "elevation_m", "target_temperature", "observed_relative_humidity",
                    "observed_surface_pressure", "observed_wind_speed", "observed_wind_direction", "observed_precipitation"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Drop columns not mapped if we are not keeping raw
    if not keep_raw:
        # Keep only canonical columns that exist
        canonical_cols = list(rename_map.values()) + ["observation_time", "station_name"]
        cols_to_keep = [col for col in canonical_cols if col in result.columns]
        # Allow keeping some unmapped like BATTERY_VOLTAGE if requested? The user said to flag it but not include it in model features.
        # So it's fine to keep it in the dataframe or drop it if keep_raw=False. Let's keep it if we want to flag it in quality report.
        if "BATTERY_VOLTAGE(V)" in result.columns:
            cols_to_keep.append("BATTERY_VOLTAGE(V)")
        result = result[cols_to_keep]
    
    # Sort chronologically
    if "observation_time" in result.columns:
        result = result.sort_values(by="observation_time").reset_index(drop=True)

    return result

def validate_mosdac_aws(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a quality report for the normalized MOSDAC AWS dataframe."""
    report = {
        "input_rows": len(df),
        "output_rows": len(df),
        "missing_values": {},
        "invalid_timestamps": 0,
        "duplicate_timestamps": 0,
        "conflicting_duplicates": 0,
        "missing_hourly_intervals": 0,
        "missing_coordinates": 0,
        "constant_columns": [],
        "flags": []
    }

    if "observation_time" in df.columns:
        invalid_ts = df["observation_time"].isna().sum()
        report["invalid_timestamps"] = int(invalid_ts)

        valid_df = df.dropna(subset=["observation_time"])
        duplicates = valid_df.duplicated(subset=["station_id", "observation_time"], keep=False)
        report["duplicate_timestamps"] = int(duplicates.sum())

        # Check for conflicting duplicates
        if duplicates.any():
            dup_df = valid_df[duplicates]
            # Group by station and time, check if there's >1 unique row
            conflicts = dup_df.groupby(["station_id", "observation_time"]).apply(lambda x: len(x.drop_duplicates()) > 1).sum()
            report["conflicting_duplicates"] = int(conflicts)
            if conflicts > 0:
                report["flags"].append(f"Found {conflicts} conflicting duplicate timestamps.")

        # Missing hourly intervals
        if len(valid_df) > 1:
            time_diffs = valid_df["observation_time"].diff().dt.total_seconds() / 3600
            # Any diff > 1 means missing intervals (assuming continuous timeseries per station)
            # Grouping by station
            for station, group in valid_df.groupby("station_id"):
                if len(group) > 1:
                    diffs = group["observation_time"].sort_values().diff().dt.total_seconds() / 3600
                    missing = diffs[diffs > 1]
                    report["missing_hourly_intervals"] += int((missing - 1).sum())

    missing_coords = 0
    if "latitude" in df.columns:
        missing_coords += df["latitude"].isna().sum()
    if "longitude" in df.columns:
        missing_coords += df["longitude"].isna().sum()
    report["missing_coordinates"] = int(missing_coords)

    # Missing values by column
    for col in df.columns:
        missing_count = df[col].isna().sum()
        report["missing_values"][col] = int(missing_count)
        
        # Check for constant columns
        if not df[col].isna().all():
            if df[col].nunique(dropna=True) == 1:
                report["constant_columns"].append(col)

    # Specific sensor checks (suspicious values)
    # E.g. temperature out of -50 to +60
    if "target_temperature" in df.columns:
        suspicious_temp = df[(df["target_temperature"] < -50) | (df["target_temperature"] > 60)]
        if len(suspicious_temp) > 0:
            report["flags"].append(f"Found {len(suspicious_temp)} suspicious temperature readings.")

    # Humidity
    if "observed_relative_humidity" in df.columns:
        suspicious_rh = df[(df["observed_relative_humidity"] < 0) | (df["observed_relative_humidity"] > 100)]
        if len(suspicious_rh) > 0:
            report["flags"].append(f"Found {len(suspicious_rh)} suspicious relative humidity readings.")

    # Battery Voltage
    if "BATTERY_VOLTAGE(V)" in df.columns:
        suspicious_volts = df[df["BATTERY_VOLTAGE(V)"] == 9953.0]
        if len(suspicious_volts) > 0:
            report["flags"].append(f"Flagged {len(suspicious_volts)} rows with battery voltage 9953 (likely error code).")

    return report

def save_mosdac_aws(df: pd.DataFrame, output_path: str) -> None:
    """Save the normalized dataframe to a CSV or Parquet file."""
    if output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def align_forecast_observations(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    tolerance: str = '30min'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align AWS station observations to forecast data.
    
    Joins on valid_time and station/location identity.
    Returns the joined dataframe and a report of unmatched rows.
    """
    # Ensure both dataframes have the necessary temporal columns
    if "valid_time" not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'valid_time'")
    if "observation_time" not in obs_df.columns:
        raise ValueError("obs_df must contain 'observation_time'")

    forecast_df = forecast_df.copy()
    obs_df = obs_df.copy()

    forecast_df["valid_time"] = pd.to_datetime(forecast_df["valid_time"], utc=True)
    obs_df["observation_time"] = pd.to_datetime(obs_df["observation_time"], utc=True)

    # Sort
    forecast_df = forecast_df.sort_values("valid_time")
    obs_df = obs_df.sort_values("observation_time")
    
    # We require explicit mapping of station to Panchayat.
    # Since this module only aligns, we assume the user has either passed forecast_df with station_id,
    # or we merge on a shared location key. If station_id is in forecast_df, we merge on that.
    # If not, we fail, to avoid accidental many-to-many geographic joins.
    if "station_id" not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'station_id' for explicit mapping.")
    if "station_id" not in obs_df.columns:
        raise ValueError("obs_df must contain 'station_id'.")

    # Reject duplicate observation timestamps for a given station to prevent silent many-to-one mapping
    if obs_df.duplicated(subset=["station_id", "observation_time"]).any():
        raise ValueError("obs_df contains duplicate observations for the same station and time.")


    # We keep a copy of observation_time to track exactly which obs were matched
    obs_df_merge = obs_df.rename(columns={"observation_time": "valid_time"})
    obs_df_merge["matched_obs_time"] = obs_df["observation_time"]

    # Perform a merge_asof on time for each station
    # merge_asof requires exact match on 'by' and nearest on 'on'
    aligned = pd.merge_asof(
        forecast_df,
        obs_df_merge,
        on="valid_time",
        by="station_id",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance)
    )

    unmatched_forecast = aligned["target_temperature"].isna().sum() if "target_temperature" in aligned.columns else 0
    # Calculate unmatched obs
    # An observation is unmatched if its (station_id, time) is not in the aligned dataset
    aligned_obs = aligned[["station_id", "matched_obs_time"]].dropna()
    obs_keys = set(zip(obs_df["station_id"], obs_df["observation_time"]))
    aligned_keys = set(zip(aligned_obs["station_id"], aligned_obs["matched_obs_time"]))
    unmatched_obs = len(obs_keys - aligned_keys)

    # Drop the temporary tracking column
    if "matched_obs_time" in aligned.columns:
        aligned = aligned.drop(columns=["matched_obs_time"])

    report = {
        "unmatched_forecast_rows": int(unmatched_forecast),
        "unmatched_observation_rows": int(unmatched_obs)
    }

    return aligned, report
