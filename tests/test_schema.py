import pandas as pd
import pytest

from src.data.schema import (
    REQUIRED_FORECAST_COLUMNS,
    validate_canonical_columns,
    validate_temporal_columns,
)


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "forecast_init_time": [
                "2026-10-05T00:00:00Z",
                "2026-10-05T00:00:00Z",
            ],
            "valid_time": [
                "2026-10-05T00:00:00Z",
                "2026-10-05T06:00:00Z",
            ],
            "lead_hours": [0.0, 6.0],
        }
    )


def test_required_forecast_columns_are_defined():
    assert REQUIRED_FORECAST_COLUMNS == [
        "forecast_init_time",
        "valid_time",
        "lead_hours",
    ]


def test_valid_temporal_schema():
    df = make_valid_dataframe()

    validate_canonical_columns(
        df,
        REQUIRED_FORECAST_COLUMNS,
    )

    validate_temporal_columns(df)


def test_missing_column_is_rejected():
    df = make_valid_dataframe().drop(columns=["lead_hours"])

    with pytest.raises(
        ValueError,
        match="Missing required PanchayatCast columns",
    ):
        validate_temporal_columns(df)


def test_incorrect_lead_hours_are_rejected():
    df = make_valid_dataframe()
    df.loc[1, "lead_hours"] = 12.0

    with pytest.raises(
        ValueError,
        match="lead_hours does not match",
    ):
        validate_temporal_columns(df)

def test_require_initialized_forecast_accepts_valid():
    from src.data.schema import require_initialized_forecast
    df = make_valid_dataframe()
    require_initialized_forecast(df)

def test_require_initialized_forecast_rejects_reanalysis():
    from src.data.schema import require_initialized_forecast
    df = pd.DataFrame(
        {
            "forecast_init_time": [
                "2026-10-05T00:00:00Z",
                "2026-10-05T01:00:00Z",
            ],
            "valid_time": [
                "2026-10-05T00:00:00Z",
                "2026-10-05T01:00:00Z",
            ],
            "lead_hours": [0.0, 0.0],
        }
    )
    
    with pytest.raises(ValueError, match="reanalysis or seamless historical data"):
        require_initialized_forecast(df)