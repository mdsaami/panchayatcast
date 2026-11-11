"""Unit and scientific integrity tests for the weather downscaling pipeline."""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain.srtm_extractor import _meters_to_lat_lon_offsets
from src.features.pipeline import add_cyclical_time_features
from src.advisory.engine import AgroAdvisoryEngine
from src.uncertainty.conformal import ConformalPredictor


def test_locations_configuration():
    cfg_file = PROJECT_ROOT / "config" / "locations.json"
    assert cfg_file.exists(), "locations.json missing"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    locs = cfg["locations"]
    assert len(locs) == 15, f"Expected 15 locations, found {len(locs)}"

    ids = [l["id"] for l in locs]
    assert len(set(ids)) == 15, "Duplicate location IDs found"

    folds = cfg["folds"]
    assert len(folds) == 5, "Expected 5 spatial folds"

    all_test_locs = []
    for f_idx, test_locs in folds.items():
        assert len(test_locs) == 3, f"Fold {f_idx} must have 3 test locations"
        all_test_locs.extend(test_locs)
    
    # Check that all 15 locations appear exactly once in test sets across 5 folds
    assert set(all_test_locs) == set(ids), "Spatial folds do not partition all 15 locations"


def test_metric_terrain_derivatives():
    d_lat, d_lon = _meters_to_lat_lon_offsets(28.6139, 500.0)
    # At latitude 28.6 deg: d_lat ~ 500 / 111139 ~ 0.0045 deg
    assert 0.004 < d_lat < 0.005, f"Unexpected d_lat: {d_lat}"
    assert 0.004 < d_lon < 0.006, f"Unexpected d_lon: {d_lon}"


def test_cyclical_time_features():
    df = pd.DataFrame({
        "time": ["2024-08-01T00:00:00", "2024-08-01T06:00:00", "2024-08-01T12:00:00", "2024-08-01T18:00:00"]
    })
    df = add_cyclical_time_features(df)
    
    # Hour sin^2 + cos^2 == 1.0
    for i in range(len(df)):
        val = df["hour_sin"].iloc[i]**2 + df["hour_cos"].iloc[i]**2
        assert np.isclose(val, 1.0), f"Hour cyclical normalization failed: {val}"


def test_advisory_engine_logic():
    engine = AgroAdvisoryEngine()
    
    # Test Rain wash-off alert
    adv_rain = engine.evaluate_advisory(
        crop="Rice (Paddy)",
        crop_stage="Flowering / Heading",
        temp_refined=29.0,
        temp_lower=27.5,
        temp_upper=30.5,
        relative_humidity=75.0,
        wind_speed_kmh=10.0,
        precipitation_mm=2.5
    )
    categories = {a["category"]: a for a in adv_rain["advisories"]}
    assert "Chemical Spraying" in categories
    assert categories["Chemical Spraying"]["status"] == "Hazard - Defer Spraying"

    # Test Thermal stress alert
    adv_heat = engine.evaluate_advisory(
        crop="Rice (Paddy)",
        crop_stage="Flowering / Heading",
        temp_refined=36.5,
        temp_lower=34.0,
        temp_upper=39.0,
        relative_humidity=45.0,
        wind_speed_kmh=8.0,
        precipitation_mm=0.0
    )
    heat_cat = {a["category"]: a for a in adv_heat["advisories"]}["Thermal Risk"]
    assert "Thermal Stress Alert" in heat_cat["status"]


def test_conformal_predictor():
    cp = ConformalPredictor(coverage=0.90)
    y_true = np.array([25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0])
    y_pred = np.array([25.2, 25.8, 27.1, 28.5, 28.8, 30.2, 30.9, 32.3, 33.1, 33.8])
    
    cp.calibrate(y_true, y_pred)
    res = cp.evaluate_coverage(y_true, y_pred)
    assert res["empirical_coverage_pct"] >= 90.0, f"Coverage lower than guarantee: {res}"


def test_downscale_api_inference(monkeypatch):
    import app.backend.main as backend_main
    from app.backend.main import downscale_forecast, DownscaleRequest

    def mock_fetch_gfs(latitude, longitude, start_date=None, end_date=None, **kwargs):
        return {
            "latitude": latitude,
            "longitude": longitude,
            "elevation": 216.0,
            "hourly": {
                "time": [
                    "2024-08-15T00:00",
                    "2024-08-15T06:00",
                    "2024-08-15T12:00",
                    "2024-08-15T18:00",
                ],
                "temperature_2m": [27.5, 29.0, 33.2, 30.1],
                "relative_humidity_2m": [82.0, 75.0, 62.0, 74.0],
                "surface_pressure": [982.5, 983.1, 980.4, 981.8],
                "wind_speed_10m": [8.5, 11.0, 14.2, 9.6],
                "precipitation": [0.0, 0.2, 1.4, 0.0],
            },
        }

    def mock_extract_terrain(latitude, longitude, **kwargs):
        return {
            "elevation_local": 218.0,
            "slope_deg": 1.2,
            "aspect_deg": 135.0,
            "aspect_sin": float(np.sin(np.radians(135.0))),
            "aspect_cos": float(np.cos(np.radians(135.0))),
            "relief_500m": 6.0,
            "relief_1000m": 12.0,
            "elev_std_1000m": 3.4,
        }

    monkeypatch.setattr(backend_main, "fetch_gfs_forecast", mock_fetch_gfs)
    monkeypatch.setattr(backend_main, "extract_terrain_features", mock_extract_terrain)

    req = DownscaleRequest(
        latitude=28.6139,
        longitude=77.2090,
        location_name="Delhi Panchayat",
        crop="Rice (Paddy)",
        crop_stage="Flowering / Heading",
        start_date="2024-08-15",
        end_date="2024-08-17"
    )
    res = downscale_forecast(req)
    assert "metadata" in res
    assert "advisory" in res
    assert "hourly_timeline" in res
    assert len(res["hourly_timeline"]) == 4
    assert "refined_temp_c" in res["hourly_timeline"][0]
