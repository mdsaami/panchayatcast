"""Canonical data contract for the PanchayatCast prediction pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


# Fields that identify a forecast record and its temporal context.
REQUIRED_FORECAST_COLUMNS = [
    "forecast_init_time",
    "valid_time",
    "lead_hours",
]

# Baseline forecast variables.
REQUIRED_BASELINE_COLUMNS = [
    "baseline_temperature",
]

# Geographic identity. Panchayat/block IDs are intentionally required
# by the canonical contract, but the legacy prototype does not yet provide them.
REQUIRED_GEOGRAPHIC_COLUMNS = [
    "panchayat_id",
    "block_id",
]

# Target is kept separate from the baseline. It may represent an observation
# source such as AWS/ARG when a valid observation dataset is available.
TARGET_COLUMNS = [
    "target_temperature",
]

# Core physiographic/context fields.
OPTIONAL_CONTEXT_COLUMNS = [
    "latitude",
    "longitude",
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "tri",
    "elevation_variability",
    "station_distance_km",
    "month",
    "day_of_year",
    "hour",
]

# Additional baseline variables supported by the final architecture.
OPTIONAL_BASELINE_COLUMNS = [
    "baseline_relative_humidity",
    "baseline_surface_pressure",
    "baseline_wind_speed",
    "baseline_precipitation",
]

ALL_CANONICAL_COLUMNS = (
    REQUIRED_FORECAST_COLUMNS
    + REQUIRED_BASELINE_COLUMNS
    + REQUIRED_GEOGRAPHIC_COLUMNS
    + TARGET_COLUMNS
    + OPTIONAL_BASELINE_COLUMNS
    + OPTIONAL_CONTEXT_COLUMNS
)


def validate_canonical_columns(
    df: pd.DataFrame,
    required: Iterable[str] | None = None,
) -> None:
    """Validate that a dataframe contains the requested canonical columns.

    This function checks schema only. It does not fabricate missing data.
    """
    required_columns = list(
        required if required is not None else REQUIRED_FORECAST_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required PanchayatCast columns: "
            + ", ".join(missing)
        )


def validate_temporal_columns(df: pd.DataFrame) -> None:
    """Validate forecast initialization, valid time, and lead time."""
    validate_canonical_columns(df, REQUIRED_FORECAST_COLUMNS)

    init_time = pd.to_datetime(
        df["forecast_init_time"],
        utc=True,
        errors="coerce",
    )
    valid_time = pd.to_datetime(
        df["valid_time"],
        utc=True,
        errors="coerce",
    )

    if init_time.isna().any():
        raise ValueError("Invalid forecast_init_time value detected.")

    if valid_time.isna().any():
        raise ValueError("Invalid valid_time value detected.")

    if pd.to_numeric(
        df["lead_hours"],
        errors="coerce",
    ).isna().any():
        raise ValueError("Invalid lead_hours value detected.")

    expected_lead = (
        (valid_time - init_time)
        .dt.total_seconds()
        / 3600.0
    )

    actual_lead = pd.to_numeric(
        df["lead_hours"],
        errors="coerce",
    )

    if not (actual_lead.round(6) == expected_lead.round(6)).all():
        raise ValueError(
            "lead_hours does not match "
            "(valid_time - forecast_init_time)."
        )

def require_initialized_forecast(df: pd.DataFrame) -> None:
    """Ensure the dataframe contains genuinely initialized forecasts.
    
    Rejects reanalysis or seamless historical series where initialization
    timestamps and lead times are fabricated (e.g., lead_hours all zero).
    """
    validate_temporal_columns(df)
    
    # If every single row has a lead time of 0.0, it's highly likely
    # this is a reanalysis or seamless product masked as an initialized run.
    # A true initialized run for a time series has varying lead times.
    if (df["lead_hours"] == 0.0).all():
        raise ValueError(
            "Dataset appears to be reanalysis or seamless historical data "
            "(lead_hours is uniformly 0.0). Genuinely initialized forecast "
            "runs are required."
        )