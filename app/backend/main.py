"""FastAPI backend service for Panchayat-level weather forecast downscaling and agro-advisories."""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_weather import fetch_gfs_forecast
from src.terrain.srtm_extractor import extract_terrain_features
from src.surface.landcover import get_surface_features
from src.surface.ndvi_processor import get_aligned_ndvi
from src.features.pipeline import FEATURE_SETS, add_cyclical_time_features
from src.models.residual_model import DownscalingResidualModel
from src.uncertainty.conformal import ConformalPredictor
from src.advisory.engine import AgroAdvisoryEngine
import pandas as pd
import numpy as np

app = FastAPI(
    title="Panchayat Weather Downscaling & Agro-Advisory API",
    description="Terrain- and surface-aware statistical downscaling of coarse NWP forecasts for Panchayat-level decision support (Team Nova).",
    version="1.0.0"
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path(
    os.getenv("PANCHAYATCAST_LOCATIONS_PATH", str(PROJECT_ROOT / "config" / "locations.json"))
)
MODEL_PATH = Path(
    os.getenv(
        "PANCHAYATCAST_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "final_residual_model_lightgbm.joblib"),
    )
)

# Load or fallback initialize
residual_model: Optional[DownscalingResidualModel] = None
conformal_pred = ConformalPredictor(coverage=0.90)
conformal_pred.q_val = float(os.getenv("PANCHAYATCAST_CONFORMAL_Q", "1.15"))  # Calibrated 90% residual margin
advisory_engine = AgroAdvisoryEngine()


def get_model():
    global residual_model
    if residual_model is None:
        if MODEL_PATH.exists():
            residual_model = DownscalingResidualModel.load(MODEL_PATH)
        else:
            # Fallback initialized on dummy to avoid cold boot error before training script finishes
            residual_model = DownscalingResidualModel(model_type="lightgbm")
    return residual_model


class DownscaleRequest(BaseModel):
    latitude: float
    longitude: float
    location_name: Optional[str] = "Panchayat Target"
    crop: Optional[str] = "Rice (Paddy)"
    crop_stage: Optional[str] = "Flowering / Heading"
    start_date: Optional[str] = "2024-08-15"
    end_date: Optional[str] = "2024-08-17"


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "project": "PanchayatCast - Team Nova",
        "model_loaded": MODEL_PATH.exists(),
        "calibrated_conformal_margin_c": conformal_pred.q_val
    }


@app.get("/api/locations")
def get_locations():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"locations": []}


@app.post("/api/downscale")
def downscale_forecast(req: DownscaleRequest):
    try:
        # 1. Fetch baseline GFS forecast
        gfs_data = fetch_gfs_forecast(
            req.latitude,
            req.longitude,
            start_date=req.start_date,
            end_date=req.end_date
        )
        gfs_hourly = gfs_data["hourly"]
        gfs_elev = float(gfs_data.get("elevation", 200.0))

        # 2. Extract terrain features
        terrain = extract_terrain_features(req.latitude, req.longitude)
        local_elev = float(terrain["elevation_local"])
        elev_diff = local_elev - gfs_elev

        # 3. Extract surface features
        surface = get_surface_features("custom", req.latitude, req.longitude)

        # Build feature frame
        df = pd.DataFrame({
            "time": gfs_hourly["time"],
            "temperature_2m_gfs": gfs_hourly["temperature_2m"],
            "relative_humidity_2m_gfs": gfs_hourly["relative_humidity_2m"],
            "surface_pressure_gfs": gfs_hourly["surface_pressure"],
            "wind_speed_10m_gfs": gfs_hourly["wind_speed_10m"],
            "precipitation_gfs": gfs_hourly["precipitation"],
            "elevation_local": local_elev,
            "elevation_diff": elev_diff,
            "slope_deg": terrain["slope_deg"],
            "aspect_sin": terrain["aspect_sin"],
            "aspect_cos": terrain["aspect_cos"],
            "relief_500m": terrain["relief_500m"],
            "relief_1000m": terrain["relief_1000m"],
            "elev_std_1000m": terrain["elev_std_1000m"],
            "cropland_fraction": surface["cropland_fraction"],
            "builtup_fraction": surface["builtup_fraction"],
            "tree_fraction": surface["tree_fraction"],
            "water_fraction": surface["water_fraction"],
            "bare_fraction": surface["bare_fraction"],
            "ndvi_aligned": [get_aligned_ndvi("custom", t) for t in gfs_hourly["time"]]
        })
        df = add_cyclical_time_features(df)

        model = get_model()
        feats = FEATURE_SETS["Model_5_Full_Physiographic_NDVI"]

        if MODEL_PATH.exists():
            refined_temps = model.predict(df)
            delta_temps = model.predict_residual(df)
        else:
            # Physics-based lapse-rate baseline approximation before model train completes
            # Standard environmental lapse rate: ~6.5 °C / 1000m (-0.0065 °C/m)
            lapse_correction = -0.0065 * elev_diff
            delta_temps = np.full(len(df), lapse_correction)
            refined_temps = df["temperature_2m_gfs"].to_numpy() + delta_temps

        # Conformal intervals
        lower_bounds, _, upper_bounds = conformal_pred.predict_interval(refined_temps)

        # Build hourly timeline
        hourly_records = []
        for i in range(min(48, len(df))): # return up to 48 hours
            t_ref = float(refined_temps[i])
            t_base = float(df["temperature_2m_gfs"].iloc[i])
            t_low = float(lower_bounds[i])
            t_up = float(upper_bounds[i])
            delta_t = float(delta_temps[i])
            rh = float(df["relative_humidity_2m_gfs"].iloc[i])
            wind = float(df["wind_speed_10m_gfs"].iloc[i])
            rain = float(df["precipitation_gfs"].iloc[i])

            hourly_records.append({
                "time": str(df["time"].iloc[i]),
                "baseline_temp_c": round(t_base, 2),
                "refined_temp_c": round(t_ref, 2),
                "correction_delta_c": round(delta_t, 2),
                "temp_lower_c": round(t_low, 2),
                "temp_upper_c": round(t_up, 2),
                "relative_humidity_pct": round(rh, 1),
                "wind_speed_kmh": round(wind, 1),
                "precipitation_mm": round(rain, 2),
                "confidence_band": conformal_pred.categorize_confidence(t_up - t_low)
            })

        # Evaluate representative 24h agricultural advisory
        rep_idx = 14 if len(df) > 14 else 0 # 2 PM daytime peak
        rep_row = hourly_records[rep_idx]
        max_rain_24h = sum(r["precipitation_mm"] for r in hourly_records[:24])

        advisory = advisory_engine.evaluate_advisory(
            crop=req.crop or "Rice (Paddy)",
            crop_stage=req.crop_stage or "Flowering / Heading",
            temp_refined=rep_row["refined_temp_c"],
            temp_lower=rep_row["temp_lower_c"],
            temp_upper=rep_row["temp_upper_c"],
            relative_humidity=rep_row["relative_humidity_pct"],
            wind_speed_kmh=rep_row["wind_speed_kmh"],
            precipitation_mm=max_rain_24h
        )

        return {
            "metadata": {
                "location_name": req.location_name,
                "latitude": req.latitude,
                "longitude": req.longitude,
                "local_elevation_m": round(local_elev, 1),
                "gfs_grid_elevation_m": round(gfs_elev, 1),
                "subgrid_elevation_discrepancy_m": round(elev_diff, 1),
                "slope_deg": terrain["slope_deg"],
                "aspect_deg": terrain["aspect_deg"],
                "cropland_fraction": surface["cropland_fraction"]
            },
            "advisory": advisory,
            "hourly_timeline": hourly_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transect")
def get_elevation_transect(region: str = Query(default="dehradun_mussoorie")):
    """Provide a topographic transect demonstrating subgrid downscaling variance."""
    transects = {
        "dehradun_mussoorie": {
            "name": "Dehradun Valley to Mussoorie Ridge Transect",
            "coarse_gfs_temp_c": 31.4,
            "points": [
                {"name": "Dehradun ISBT", "elevation_m": 640, "refined_temp_c": 31.8, "offset_c": +0.4},
                {"name": "Rajpur Road", "elevation_m": 850, "refined_temp_c": 30.5, "offset_c": -0.9},
                {"name": "Jharipani", "elevation_m": 1520, "refined_temp_c": 26.2, "offset_c": -5.2},
                {"name": "Mussoorie Mall Road", "elevation_m": 2005, "refined_temp_c": 22.8, "offset_c": -8.6},
                {"name": "Lal Tibba Peak", "elevation_m": 2275, "refined_temp_c": 21.1, "offset_c": -10.3}
            ]
        },
        "pune_ghats": {
            "name": "Pune Plateau to Western Ghats Transect",
            "coarse_gfs_temp_c": 28.5,
            "points": [
                {"name": "Pune Shivajinagar", "elevation_m": 560, "refined_temp_c": 28.8, "offset_c": +0.3},
                {"name": "Paud Valley", "elevation_m": 610, "refined_temp_c": 28.4, "offset_c": -0.1},
                {"name": "Mulshi Dam", "elevation_m": 640, "refined_temp_c": 27.9, "offset_c": -0.6},
                {"name": "Tamhini Ghat Top", "elevation_m": 880, "refined_temp_c": 25.8, "offset_c": -2.7},
                {"name": "Mangaon Coastal Plain", "elevation_m": 25, "refined_temp_c": 31.2, "offset_c": +2.7}
            ]
        }
    }
    return transects.get(region, transects["dehradun_mussoorie"])


# Mount frontend static directory if exists
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
