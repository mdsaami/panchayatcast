"""CLI script to run the historical benchmark pipeline."""

import argparse
import json
import logging
import os
from pathlib import Path
import pandas as pd

from src.data.historical_pipeline import process_historical_aws, run_benchmark
from src.data.historical_forecast import get_historical_reanalysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="PanchayatCast Historical Benchmark Pipeline")
    parser.add_argument("--aws-files", nargs='+', required=True, help="List of MOSDAC AWS CSV files to process")
    parser.add_argument("--lat", type=float, default=25.98, help="Station latitude")
    parser.add_argument("--lon", type=float, default=85.66, help="Station longitude")
    parser.add_argument("--mapping-file", type=str, default="config/station_panchayat_mapping.csv", help="Path to station-Panchayat mapping")
    parser.add_argument("--level", choices=["station", "panchayat"], default="station", help="Benchmark aggregation level")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--require-initialized", action="store_true", help="Enforce initialized historical forecast runs (rejects reanalysis)")
    
    args = parser.parse_args()
    
    logger.info("1. Processing AWS Observations...")
    obs_df = process_historical_aws(args.aws_files, args.output_dir, args.reports_dir)
    logger.info(f"Processed {len(obs_df)} normalized observations.")
    
    # Extract date range from obs to fetch forecast
    min_obs_time = obs_df["observation_time"].min()
    max_obs_time = obs_df["observation_time"].max()
    
    start_date = min_obs_time.strftime("%Y-%m-%d")
    end_date = max_obs_time.strftime("%Y-%m-%d")
    
    logger.info(f"2. Fetching Forecast Data for {start_date} to {end_date}...")
    if args.require_initialized:
        logger.info("Require initialized flag set: fetching initialized historical single-runs instead of reanalysis.")
        # Note: In a full pipeline, we would iteratively fetch daily runs from openmeteo_runs.py
        # and merge them into a single historical forecast dataframe. Since this is an investigation
        # milestone and we know 2013 data fails, we raise an explicit error to signal the gap
        # rather than fabricating data, honoring the user's constraints.
        raise ValueError(
            f"Initialized forecast runs are required but historically unavailable for {start_date} to {end_date} via Open-Meteo free API. "
            "Please acquire authorized AWS observations overlapping the last 120 days to proceed with initialized benchmarking."
        )
    else:
        # Using reanalysis since 2013 single-runs are unavailable
        logger.warning("Using Reanalysis data as baseline proxy. Lead times and initialization are synthesized.")
        forecast_df, fcst_metadata = get_historical_reanalysis(
            latitude=args.lat,
            longitude=args.lon,
            start_date=start_date,
            end_date=end_date,
            cache_tag=f"lat{args.lat}_lon{args.lon}_historical"
        )
    
    # Save the forecast data and metadata
    forecast_out = os.path.join(args.output_dir, "historical_reanalysis_baseline.parquet")
    forecast_df.to_parquet(forecast_out, index=False)
    
    with open(os.path.join(args.reports_dir, "forecast_provenance.json"), "w") as f:
        json.dump(fcst_metadata, f, indent=2)
        
    logger.info("3. Running Benchmark...")
    
    # If station level, explicitly set forecast station_id to match obs for benchmark
    # to avoid merging issues since reanalysis adapter sets it to a proxy
    if args.level == "station":
        # Usually, forecast should have proper station mapping, but here we just align
        # by treating the reanalysis as co-located with the primary station in obs.
        # We find the primary station in obs_df and set it in forecast_df.
        primary_station = obs_df["station_id"].mode()[0]
        forecast_df["station_id"] = primary_station
    
    try:
        metrics = run_benchmark(
            obs_df=obs_df,
            forecast_df=forecast_df,
            mapping_path=args.mapping_file,
            level=args.level,
            require_initialized=args.require_initialized
        )
        
        metrics_out = os.path.join(args.reports_dir, f"benchmark_metrics_{args.level}.json")
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Benchmark completed successfully. Metrics saved to {metrics_out}")
        print("\n--- Benchmark Results ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
