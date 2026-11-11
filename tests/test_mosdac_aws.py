import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone
import json
import os

from src.data.mosdac_aws import (
    normalize_mosdac_aws,
    validate_mosdac_aws,
    align_forecast_observations,
    save_mosdac_aws,
    load_mosdac_aws_csv
)

@pytest.fixture
def sample_raw_df():
    data = {
        "@STATION_ID": ["IMDE1401_141579(PUSA_BH_AGRO)", "IMDE1401_141579(PUSA_BH_AGRO)"],
        "LATITUDE": [25.98, 25.98],
        "LONGITUDE": [85.66, 85.66],
        "ALTITUDE(m)": ["null", "null"],
        "TIME(GMT)": [0, 1],
        "DATE(GMT)": ["11/01/2013", "11/01/2013"],
        "AIR_TEMP(°C)": [29.5, 29.4],
        "HUMIDITY(%)": [99.0, 99.0],
        "ATMO_PRESSURE(hpa)": [956.7, 957.5],
        "WIND_SPEED(m/s)": [0.7, 0.3],
        "WIND_DIRECTION(deg)": [91.9, 122.2],
        "RAIN_FALL(mm)": [0.0, 0.0],
        "BATTERY_VOLTAGE(V)": [9953.0, 12.0]
    }
    return pd.DataFrame(data)

def test_column_normalization(sample_raw_df):
    df = normalize_mosdac_aws(sample_raw_df)
    
    assert "station_id" in df.columns
    assert "station_name" in df.columns
    assert df["station_id"].iloc[0] == "IMDE1401_141579"
    assert df["station_name"].iloc[0] == "PUSA_BH_AGRO"
    assert "target_temperature" in df.columns
    assert "elevation_m" in df.columns
    
    # Check no fabricated elevation
    assert pd.isna(df["elevation_m"].iloc[0])
    
    # Ensure correct columns kept
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "BATTERY_VOLTAGE(V)" in df.columns  # Kept for flagging
    
def test_utc_timestamp_conversion(sample_raw_df):
    df = normalize_mosdac_aws(sample_raw_df)
    
    assert "observation_time" in df.columns
    ts0 = df["observation_time"].iloc[0]
    assert ts0.tz == timezone.utc
    assert ts0.hour == 0
    assert ts0.month == 11
    assert ts0.day == 1
    assert ts0.year == 2013

def test_invalid_timestamps():
    data = {
        "@STATION_ID": ["ST1"],
        "TIME(GMT)": [25], # Invalid hour
        "DATE(GMT)": ["13/01/2013"] # Invalid month if format is MM/DD/YYYY
    }
    df = pd.DataFrame(data)
    norm_df = normalize_mosdac_aws(df)
    assert pd.isna(norm_df["observation_time"].iloc[0])
    
    report = validate_mosdac_aws(norm_df)
    assert report["invalid_timestamps"] == 1

def test_missing_values_and_constant_fields(sample_raw_df):
    sample_raw_df.loc[0, "AIR_TEMP(°C)"] = np.nan
    df = normalize_mosdac_aws(sample_raw_df)
    
    report = validate_mosdac_aws(df)
    assert report["missing_values"]["target_temperature"] == 1
    assert "observed_precipitation" in report["constant_columns"] # RAIN_FALL is 0.0 for both

def test_duplicate_timestamps():
    data = {
        "@STATION_ID": ["ST1", "ST1"],
        "TIME(GMT)": [0, 0],
        "DATE(GMT)": ["11/01/2013", "11/01/2013"],
        "AIR_TEMP(°C)": [20.0, 21.0] # conflicting duplicates
    }
    df = pd.DataFrame(data)
    norm_df = normalize_mosdac_aws(df)
    
    report = validate_mosdac_aws(norm_df)
    assert report["duplicate_timestamps"] == 2
    assert report["conflicting_duplicates"] == 1
    assert any("conflicting duplicate" in flag for flag in report["flags"])

def test_missing_hourly_intervals():
    data = {
        "@STATION_ID": ["ST1", "ST1", "ST1"],
        "TIME(GMT)": [0, 1, 3], # missing hour 2
        "DATE(GMT)": ["11/01/2013", "11/01/2013", "11/01/2013"],
    }
    df = pd.DataFrame(data)
    norm_df = normalize_mosdac_aws(df)
    report = validate_mosdac_aws(norm_df)
    assert report["missing_hourly_intervals"] == 1

def test_missing_station_identity_or_coords():
    data = {
        "@STATION_ID": [np.nan],
        "LATITUDE": [np.nan],
        "LONGITUDE": [np.nan],
        "TIME(GMT)": [0],
        "DATE(GMT)": ["11/01/2013"]
    }
    df = pd.DataFrame(data)
    norm_df = normalize_mosdac_aws(df)
    report = validate_mosdac_aws(norm_df)
    assert report["missing_coordinates"] == 2
    assert report["missing_values"]["station_id"] == 1

def test_suspicious_values_flagged(sample_raw_df):
    sample_raw_df.loc[0, "AIR_TEMP(°C)"] = 100.0 # Suspicious temp
    sample_raw_df.loc[1, "HUMIDITY(%)"] = -10.0 # Suspicious humidity
    
    df = normalize_mosdac_aws(sample_raw_df)
    report = validate_mosdac_aws(df)
    
    # Should flag but not destroy
    assert any("suspicious temperature" in flag for flag in report["flags"])
    assert any("suspicious relative humidity" in flag for flag in report["flags"])
    assert any("battery voltage 9953" in flag for flag in report["flags"])
    
    # Valid readings should remain
    assert not pd.isna(df["target_temperature"].iloc[0])
    assert not pd.isna(df["observed_relative_humidity"].iloc[1])

def test_forecast_observation_alignment():
    # Observations
    obs_data = {
        "station_id": ["ST1", "ST1", "ST2"],
        "observation_time": [
            pd.Timestamp("2013-11-01 00:00:00", tz="UTC"),
            pd.Timestamp("2013-11-01 01:00:00", tz="UTC"),
            pd.Timestamp("2013-11-01 00:00:00", tz="UTC")
        ],
        "target_temperature": [20.0, 21.0, 25.0]
    }
    obs_df = pd.DataFrame(obs_data)
    
    # Forecasts
    fcst_data = {
        "station_id": ["ST1", "ST1", "ST3"], # ST3 will not match
        "forecast_init_time": [
            pd.Timestamp("2013-10-31 00:00:00", tz="UTC"),
            pd.Timestamp("2013-10-31 00:00:00", tz="UTC"),
            pd.Timestamp("2013-10-31 00:00:00", tz="UTC")
        ],
        "valid_time": [
            pd.Timestamp("2013-11-01 00:00:00", tz="UTC"),
            pd.Timestamp("2013-11-01 01:05:00", tz="UTC"), # Within tolerance
            pd.Timestamp("2013-11-01 00:00:00", tz="UTC")
        ],
        "lead_hours": [24, 25, 24]
    }
    fcst_df = pd.DataFrame(fcst_data)
    
    aligned, report = align_forecast_observations(fcst_df, obs_df, tolerance='30min')
    
    assert len(aligned) == 3
    
    # ST1 at 00:00 matched
    match1 = aligned[(aligned["station_id"] == "ST1") & (aligned["lead_hours"] == 24)]
    assert match1["target_temperature"].iloc[0] == 20.0
    
    # ST1 at 01:05 -> With 30min tolerance, 01:00 is within tolerance
    match2 = aligned[(aligned["station_id"] == "ST1") & (aligned["lead_hours"] == 25)]
    assert match2["target_temperature"].iloc[0] == 21.0
    
    # ST3 unmatched
    match3 = aligned[aligned["station_id"] == "ST3"]
    assert pd.isna(match3["target_temperature"].iloc[0])
    
    # ST2 obs unmatched
    assert report["unmatched_observation_rows"] == 1
    assert report["unmatched_forecast_rows"] == 1

def test_accidental_many_to_many_prevented():
    obs_df = pd.DataFrame({"observation_time": [pd.Timestamp("2013-11-01 00:00:00", tz="UTC")]})
    fcst_df = pd.DataFrame({"valid_time": [pd.Timestamp("2013-11-01 00:00:00", tz="UTC")]})
    
    with pytest.raises(ValueError, match="station_id"):
        align_forecast_observations(fcst_df, obs_df)

def test_round_trip_output(tmp_path, sample_raw_df):
    df = normalize_mosdac_aws(sample_raw_df)
    
    # Save
    out_csv = tmp_path / "test.csv"
    save_mosdac_aws(df, str(out_csv))
    
    # Load
    loaded = pd.read_csv(str(out_csv))
    # Loaded timestamps are strings by default in CSV, but checking row counts
    assert len(loaded) == len(df)
    assert "target_temperature" in loaded.columns

def test_align_adversarial_near_hour_mismatches():
    obs_df = pd.DataFrame({
        "station_id": ["ST1", "ST1"],
        "observation_time": [pd.Timestamp("2023-01-01 12:00:00", tz="UTC"), pd.Timestamp("2023-01-01 14:00:00", tz="UTC")],
        "target_temperature": [20.0, 21.0]
    })
    
    # 13:00 is 60 mins away from 12:00 and 14:00. 30min tolerance should fail to match.
    fcst_df = pd.DataFrame({
        "station_id": ["ST1", "ST1"],
        "forecast_init_time": [pd.Timestamp("2023-01-01 00:00:00", tz="UTC")] * 2,
        "valid_time": [pd.Timestamp("2023-01-01 13:00:00", tz="UTC"), pd.Timestamp("2023-01-01 14:25:00", tz="UTC")],
        "lead_hours": [13, 14]
    })
    
    aligned, report = align_forecast_observations(fcst_df, obs_df, tolerance='30min')
    
    # 13:00 forecast matches nothing because diff is 60 min > 30min
    match1 = aligned[aligned["valid_time"] == pd.Timestamp("2023-01-01 13:00:00", tz="UTC")]
    assert pd.isna(match1["target_temperature"].iloc[0])
    
    # 14:25 forecast matches 14:00 obs because diff is 25 min <= 30min
    match2 = aligned[aligned["valid_time"] == pd.Timestamp("2023-01-01 14:25:00", tz="UTC")]
    assert match2["target_temperature"].iloc[0] == 21.0

def test_align_adversarial_duplicate_observation_timestamps():
    obs_df = pd.DataFrame({
        "station_id": ["ST1", "ST1"],
        "observation_time": [pd.Timestamp("2023-01-01 12:00:00", tz="UTC")] * 2,
        "target_temperature": [20.0, 21.0]
    })
    fcst_df = pd.DataFrame({
        "station_id": ["ST1"],
        "forecast_init_time": [pd.Timestamp("2023-01-01 00:00:00", tz="UTC")],
        "valid_time": [pd.Timestamp("2023-01-01 12:00:00", tz="UTC")],
        "lead_hours": [12]
    })
    
    with pytest.raises(ValueError, match="duplicate observations"):
        align_forecast_observations(fcst_df, obs_df)

def test_align_adversarial_multiple_forecast_initializations():
    obs_df = pd.DataFrame({
        "station_id": ["ST1"],
        "observation_time": [pd.Timestamp("2023-01-01 12:00:00", tz="UTC")],
        "target_temperature": [20.0]
    })
    # Same valid time, different init times
    fcst_df = pd.DataFrame({
        "station_id": ["ST1", "ST1"],
        "forecast_init_time": [pd.Timestamp("2023-01-01 00:00:00", tz="UTC"), pd.Timestamp("2023-01-01 06:00:00", tz="UTC")],
        "valid_time": [pd.Timestamp("2023-01-01 12:00:00", tz="UTC")] * 2,
        "lead_hours": [12, 6]
    })
    
    aligned, report = align_forecast_observations(fcst_df, obs_df)
    
    # Both forecast initializations should get the same observation
    assert len(aligned) == 2
    assert (aligned["target_temperature"] == 20.0).all()
    assert report["unmatched_forecast_rows"] == 0
    assert report["unmatched_observation_rows"] == 0

