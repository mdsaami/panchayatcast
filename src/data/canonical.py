"""Canonical schema adapter for PanchayatCast.

Converts legacy forecast dataframes into the canonical PanchayatCast
baseline naming convention without changing the underlying values.
"""

from __future__ import annotations

import pandas as pd

from src.data.schema import validate_canonical_columns


LEGACY_TO_CANONICAL = {
    "temperature_2m_gfs": "baseline_temperature",
    "relative_humidity_2m_gfs": "baseline_relative_humidity",
    "surface_pressure_gfs": "baseline_surface_pressure",
    "wind_speed_10m_gfs": "baseline_wind_speed",
    "precipitation_gfs": "baseline_precipitation",
    "temperature_2m_era5": "target_temperature",
}


def to_canonical_forecast(
    df: pd.DataFrame,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Convert a legacy forecast dataframe to canonical PanchayatCast names.

    The original dataframe is not modified.

    Parameters
    ----------
    df:
        Input dataframe using either legacy GFS names or canonical names.
    require_target:
        Require target_temperature to be present. Use this for training/
        validation datasets, not inference-only forecast data.

    Returns
    -------
    pd.DataFrame
        Canonical dataframe.

    Raises
    ------
    ValueError
        If required source columns are missing or canonical validation fails.
    """
    result = df.copy()

    for legacy_name, canonical_name in LEGACY_TO_CANONICAL.items():
        if canonical_name in result.columns:
            continue

        if legacy_name in result.columns:
            result[canonical_name] = result[legacy_name]

    required = [
        "baseline_temperature",
        "panchayat_id",
        "block_id",
    ]

    if require_target:
        required.append("target_temperature")

    validate_canonical_columns(result, required=required)

    return result
