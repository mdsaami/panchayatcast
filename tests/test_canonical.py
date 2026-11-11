import numpy as np
import pandas as pd
import pytest

from src.data.canonical import to_canonical_forecast
from src.models.residual_model import DownscalingResidualModel


def test_legacy_forecast_maps_to_canonical_names():
    df = pd.DataFrame({
        "temperature_2m_gfs": [30.0, 31.0],
        "relative_humidity_2m_gfs": [70.0, 65.0],
        "surface_pressure_gfs": [1005.0, 1004.0],
        "wind_speed_10m_gfs": [10.0, 12.0],
        "precipitation_gfs": [0.0, 1.5],
        "panchayat_id": ["P001", "P001"],
        "block_id": ["B001", "B001"],
    })

    result = to_canonical_forecast(df)

    assert result["baseline_temperature"].tolist() == [30.0, 31.0]
    assert result["baseline_relative_humidity"].tolist() == [70.0, 65.0]
    assert result["baseline_surface_pressure"].tolist() == [1005.0, 1004.0]
    assert result["baseline_wind_speed"].tolist() == [10.0, 12.0]
    assert result["baseline_precipitation"].tolist() == [0.0, 1.5]


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({
        "temperature_2m_gfs": [30.0],
        "panchayat_id": ["P001"],
        "block_id": ["B001"],
    })

    original_columns = list(df.columns)

    to_canonical_forecast(df)

    assert list(df.columns) == original_columns
    assert "baseline_temperature" not in df.columns


def test_target_is_mapped_when_required():
    df = pd.DataFrame({
        "temperature_2m_gfs": [30.0],
        "temperature_2m_era5": [29.0],
        "panchayat_id": ["P001"],
        "block_id": ["B001"],
    })

    result = to_canonical_forecast(df, require_target=True)

    assert result["target_temperature"].tolist() == [29.0]


def test_canonical_missing_baseline_is_rejected():
    df = pd.DataFrame({
        "panchayat_id": ["P001"],
        "block_id": ["B001"],
    })

    with pytest.raises(ValueError):
        to_canonical_forecast(df)


def test_model_accepts_canonical_baseline():
    model = DownscalingResidualModel()

    df = pd.DataFrame({
        "baseline_temperature": [30.0, 31.0],
    })

    result = model._get_baseline_temperature(df)

    np.testing.assert_array_equal(result, [30.0, 31.0])


def test_model_accepts_legacy_gfs_baseline():
    model = DownscalingResidualModel()

    df = pd.DataFrame({
        "temperature_2m_gfs": [30.0, 31.0],
    })

    result = model._get_baseline_temperature(df)

    np.testing.assert_array_equal(result, [30.0, 31.0])


def test_canonical_baseline_takes_precedence():
    model = DownscalingResidualModel()

    df = pd.DataFrame({
        "baseline_temperature": [30.0],
        "temperature_2m_gfs": [99.0],
    })

    result = model._get_baseline_temperature(df)

    np.testing.assert_array_equal(result, [30.0])


def test_missing_baseline_is_rejected():
    model = DownscalingResidualModel()

    df = pd.DataFrame({
        "relative_humidity_2m_gfs": [70.0],
    })

    with pytest.raises(KeyError):
        model._get_baseline_temperature(df)
