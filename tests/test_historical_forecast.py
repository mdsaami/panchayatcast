import pandas as pd
import pytest
from src.data.historical_forecast import get_historical_reanalysis
from src.data.schema import ALL_CANONICAL_COLUMNS

def test_historical_reanalysis_semantics_documented(monkeypatch):
    # Mock the fetcher
    def mock_fetch(*args, **kwargs):
        return {
            "latitude": 25.98,
            "longitude": 85.66,
            "hourly": {
                "time": ["2013-11-01T00:00", "2013-11-01T01:00"],
                "temperature_2m": [25.0, 26.0],
                "relative_humidity_2m": [50.0, 51.0],
                "surface_pressure": [1000.0, 1001.0],
                "wind_speed_10m": [2.0, 3.0],
                "precipitation": [0.0, 0.0]
            }
        }
    monkeypatch.setattr("src.data.historical_forecast.fetch_era5_reference", mock_fetch)
    
    df, meta = get_historical_reanalysis(25.98, 85.66, "2013-11-01", "2013-11-07")
    
    # Verify semantics
    assert meta["type"] == "reanalysis"
    assert "limitations" in meta
    
    # Verify dummy values are created properly
    assert df["forecast_init_time"].equals(df["valid_time"])
    assert (df["lead_hours"] == 0.0).all()
    assert (df["station_id"] == "ERA5_REANALYSIS_PROXY").all()
    
    # Verify canonical schema
    assert "baseline_temperature" in df.columns
    assert "valid_time" in df.columns
