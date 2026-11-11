import pandas as pd
import pytest

from src.data.openmeteo_runs import normalize_single_run


def test_lead_hours_are_calculated_from_init_and_valid_time():
    raw = {
        "_panchayatcast_metadata": {
            "forecast_init_time": "2026-10-05T00:00",
            "model": "gfs_global",
            "latitude": 30.3165,
            "longitude": 78.0322,
        },
        "hourly": {
            "time": [
                "2026-10-05T00:00",
                "2026-10-05T01:00",
                "2026-10-05T06:00",
            ],
            "temperature_2m": [
                25.0,
                25.5,
                27.0,
            ],
        },
    }

    df = normalize_single_run(raw)

    assert df["lead_hours"].tolist() == [0.0, 1.0, 6.0]


def test_initialization_time_is_not_inferred():
    raw = {
        "_panchayatcast_metadata": {},
        "hourly": {
            "time": ["2026-10-05T00:00"],
            "temperature_2m": [25.0],
        },
    }

    with pytest.raises(ValueError, match="forecast_init_time"):
        normalize_single_run(raw)


def test_valid_time_is_utc():
    raw = {
        "_panchayatcast_metadata": {
            "forecast_init_time": "2026-10-05T00:00",
            "model": "gfs_global",
            "latitude": 30.3165,
            "longitude": 78.0322,
        },
        "hourly": {
            "time": ["2026-10-05T00:00"],
            "temperature_2m": [25.0],
        },
    }

    df = normalize_single_run(raw)

    assert isinstance(
        df["valid_time"].dtype,
        pd.DatetimeTZDtype,
    )


def test_normalized_run_validates_temporal_contract():
    raw = {
        "_panchayatcast_metadata": {
            "forecast_init_time": "2026-10-05T00:00",
            "model": "gfs_global",
            "latitude": 30.3165,
            "longitude": 78.0322,
        },
        "hourly": {
            "time": [
                "2026-10-05T00:00",
                "2026-10-05T03:00",
                "2026-10-05T06:00",
            ],
            "temperature_2m": [
                25.0,
                26.0,
                27.0,
            ],
        },
    }

    df = normalize_single_run(raw)

    assert df["lead_hours"].tolist() == [0.0, 3.0, 6.0]
    assert "forecast_init_time" in df.columns
    assert "valid_time" in df.columns
    assert "lead_hours" in df.columns
