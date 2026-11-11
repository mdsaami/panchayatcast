"""Historical dataset pipeline for PanchayatCast."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.data.mosdac_aws import (
    load_mosdac_aws_csv,
    normalize_mosdac_aws,
    align_forecast_observations
)
from src.data.station_mapping import load_and_validate_mapping, join_station_to_panchayat
from src.data.historical_forecast import get_historical_reanalysis

logger = logging.getLogger(__name__)

def process_historical_aws(
    file_paths: List[str],
    output_dir: str,
    reports_dir: str
) -> pd.DataFrame:
    """Process multiple MOSDAC AWS CSV files, deduplicate, and store results."""
    all_dfs = []
    metadata_records = []
    
    processing_timestamp = datetime.now(timezone.utc).isoformat()
    
    for path in file_paths:
        try:
            raw_df = load_mosdac_aws_csv(path)
            norm_df = normalize_mosdac_aws(raw_df)
            
            # Extract basic info for metadata
            stations = norm_df["station_id"].unique().tolist() if "station_id" in norm_df.columns else []
            min_time = norm_df["observation_time"].min().isoformat() if "observation_time" in norm_df.columns and not norm_df["observation_time"].empty else None
            max_time = norm_df["observation_time"].max().isoformat() if "observation_time" in norm_df.columns and not norm_df["observation_time"].empty else None
            
            all_dfs.append(norm_df)
            metadata_records.append({
                "source_filename": os.path.basename(path),
                "stations": stations,
                "collection_period_start": min_time,
                "collection_period_end": max_time,
            })
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            raise

    if not all_dfs:
        raise ValueError("No valid data files provided.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if "observation_time" not in combined_df.columns or "station_id" not in combined_df.columns:
        raise ValueError("Data missing required columns for deduplication.")
        
    combined_df = combined_df.sort_values(by=["station_id", "observation_time"]).reset_index(drop=True)
    
    # Deduplication Policy
    # 1. Exact duplicates
    exact_duplicates_count = combined_df.duplicated().sum()
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    
    # 2. Conflicting duplicates (same station and time, different values)
    time_station_dups = combined_df.duplicated(subset=["station_id", "observation_time"], keep=False)
    conflicting_count = 0
    if time_station_dups.any():
        conflicting_df = combined_df[time_station_dups]
        conflicting_count = len(conflicting_df.groupby(["station_id", "observation_time"]))
        logger.warning(f"Found {conflicting_count} conflicting observations. Rejecting them.")
        # Reject completely
        combined_df = combined_df[~time_station_dups].reset_index(drop=True)

    # Missing hourly intervals
    missing_intervals = 0
    for station, group in combined_df.groupby("station_id"):
        if len(group) > 1:
            diffs = group["observation_time"].sort_values().diff().dt.total_seconds() / 3600
            missing = diffs[diffs > 1]
            missing_intervals += int((missing - 1).sum())

    report = {
        "processing_timestamp": processing_timestamp,
        "adapter_version": "1.0",
        "assumptions": "Rejected all conflicting duplicate records. Exact duplicates dropped.",
        "input_files": metadata_records,
        "total_rows_after_dedup": len(combined_df),
        "exact_duplicates_removed": int(exact_duplicates_count),
        "conflicting_records_rejected": int(conflicting_count),
        "missing_hourly_intervals": int(missing_intervals),
        "stations_processed": combined_df["station_id"].unique().tolist()
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, "historical_aws_normalized.parquet")
    combined_df.to_parquet(output_path, index=False)
    
    report_path = os.path.join(reports_dir, "historical_aws_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return combined_df

def run_benchmark(
    obs_df: pd.DataFrame, 
    forecast_df: pd.DataFrame, 
    mapping_path: str,
    level: str = "station",
    require_initialized: bool = False
) -> Dict[str, Any]:
    """Run baseline benchmark at station or panchayat level."""
    
    if require_initialized:
        from src.data.schema import require_initialized_forecast
        require_initialized_forecast(forecast_df)
    
    if level == "panchayat":
        mapping_df = load_and_validate_mapping(mapping_path)
        obs_df = join_station_to_panchayat(obs_df, mapping_df, aggregation_strategy="mean")
        if obs_df.empty:
            raise ValueError("No approved mappings available for Panchayat-level benchmark.")
        join_key = "panchayat_id"
        # Overwrite station_id in forecast with panchayat_id to allow alignment
        forecast_df = forecast_df.copy()
        if "panchayat_id" in forecast_df.columns:
            forecast_df["station_id"] = forecast_df["panchayat_id"]
        obs_df["station_id"] = obs_df["panchayat_id"]
    else:
        join_key = "station_id"
        # Allow station-level benchmark directly
        
    # Check temporal overlap
    min_obs = obs_df["observation_time"].min()
    max_obs = obs_df["observation_time"].max()
    min_fcst = forecast_df["valid_time"].min()
    max_fcst = forecast_df["valid_time"].max()
    
    if max_obs < min_fcst or min_obs > max_fcst:
        raise ValueError(f"Temporal disjoint: obs ({min_obs} to {max_obs}) vs forecast ({min_fcst} to {max_fcst})")

    # Align
    matched_df, match_report = align_forecast_observations(forecast_df, obs_df)
    
    if matched_df.empty or matched_df["target_temperature"].isna().all() or matched_df["baseline_temperature"].isna().all():
        raise ValueError("Matched dataset is empty or missing target/baseline temperatures.")

    # Calculate metrics
    matched_df = matched_df.dropna(subset=["target_temperature", "baseline_temperature"])
    
    mae = np.abs(matched_df["baseline_temperature"] - matched_df["target_temperature"]).mean()
    rmse = np.sqrt(np.mean((matched_df["baseline_temperature"] - matched_df["target_temperature"])**2))
    bias = (matched_df["baseline_temperature"] - matched_df["target_temperature"]).mean() # Positive bias = forecast > obs

    metrics = {
        "level": level,
        "sample_count": len(matched_df),
        "date_coverage_start": matched_df["valid_time"].min().isoformat(),
        "date_coverage_end": matched_df["valid_time"].max().isoformat(),
        "match_rate_obs": 1.0 - (match_report["unmatched_observation_rows"] / max(1, len(obs_df))),
        "baseline_mae": float(mae),
        "baseline_rmse": float(rmse),
        "mean_bias": float(bias),
        "mean_bias_convention": "forecast - observation",
        "unmatched_forecast_rows": match_report["unmatched_forecast_rows"],
        "unmatched_observation_rows": match_report["unmatched_observation_rows"]
    }

    return metrics
