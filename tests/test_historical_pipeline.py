import os
import json
import pytest
import pandas as pd
import numpy as np
from src.data.historical_pipeline import process_historical_aws, run_benchmark

def test_process_historical_aws_deduplication(tmp_path):
    # Create two overlapping CSV files with duplicates and conflicting duplicates
    csv1_content = """@STATION_ID,LATITUDE,LONGITUDE,ALTITUDE(m),DATE(GMT),TIME(GMT),AIR_TEMP(°C)
IMDE1401(PUSA),25.98,85.66,47.2,11/01/2013,0,20.0
IMDE1401(PUSA),25.98,85.66,47.2,11/01/2013,1,21.0
IMDE1402(OTHER),26.00,85.70,50.0,11/01/2013,0,22.0
"""
    # Overlapping, exact duplicate for 00:00, conflicting for 01:00
    csv2_content = """@STATION_ID,LATITUDE,LONGITUDE,ALTITUDE(m),DATE(GMT),TIME(GMT),AIR_TEMP(°C)
IMDE1401(PUSA),25.98,85.66,47.2,11/01/2013,0,20.0
IMDE1401(PUSA),25.98,85.66,47.2,11/01/2013,1,22.0
IMDE1401(PUSA),25.98,85.66,47.2,11/01/2013,3,23.0
"""
    file1 = tmp_path / "aws1.csv"
    file2 = tmp_path / "aws2.csv"
    file1.write_text(csv1_content, encoding="utf-8")
    file2.write_text(csv2_content, encoding="utf-8")
    
    out_dir = tmp_path / "processed"
    rep_dir = tmp_path / "reports"
    
    df = process_historical_aws([str(file1), str(file2)], str(out_dir), str(rep_dir))
    
    # IMDE1401 00:00 should be present (exact duplicate merged)
    # IMDE1401 01:00 should be rejected (conflicting duplicate)
    # IMDE1401 03:00 should be present
    # IMDE1402 00:00 should be present
    
    assert len(df) == 3
    
    report_file = rep_dir / "historical_aws_report.json"
    with open(report_file) as f:
        report = json.load(f)
        
    assert report["conflicting_records_rejected"] == 1
    assert report["exact_duplicates_removed"] == 1
    assert report["missing_hourly_intervals"] > 0
    assert "IMDE1401" in report["stations_processed"]
    assert "IMDE1402" in report["stations_processed"]

def test_run_benchmark_empty_overlap():
    obs_df = pd.DataFrame({
        "station_id": ["S1"],
        "observation_time": [pd.to_datetime("2026-01-01T00:00Z")],
        "target_temperature": [20.0]
    })
    fcst_df = pd.DataFrame({
        "station_id": ["S1"],
        "valid_time": [pd.to_datetime("2026-01-02T00:00Z")],
        "baseline_temperature": [21.0],
        "forecast_init_time": [pd.to_datetime("2026-01-02T00:00Z")],
        "lead_hours": [0.0]
    })
    
    with pytest.raises(ValueError, match="Temporal disjoint"):
        run_benchmark(obs_df, fcst_df, "dummy.csv", level="station")

def test_run_benchmark_metrics(tmp_path):
    obs_df = pd.DataFrame({
        "station_id": ["S1", "S1", "S1"],
        "observation_time": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T02:00Z"]),
        "target_temperature": [20.0, 21.0, 22.0]
    })
    fcst_df = pd.DataFrame({
        "station_id": ["S1", "S1"],
        "valid_time": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z"]),
        "baseline_temperature": [19.0, 21.0],
        "forecast_init_time": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:00Z"]),
        "lead_hours": [0.0, 1.0]
    })
    
    metrics = run_benchmark(obs_df, fcst_df, "dummy.csv", level="station")
    
    assert metrics["sample_count"] == 2
    # 2 matched out of 3 obs
    assert np.isclose(metrics["match_rate_obs"], 2/3)
    # Errors: |19-20|=1, |21-21|=0 -> MAE = 0.5
    assert metrics["baseline_mae"] == 0.5
    # RMSE: sqrt((1^2 + 0^2)/2) = sqrt(0.5)
    assert np.isclose(metrics["baseline_rmse"], np.sqrt(0.5))
    # Bias: forecast - obs = (19-20 + 21-21)/2 = -0.5
    assert metrics["mean_bias"] == -0.5
