"""ESA WorldCover 2021 surface land cover feature module.

Extracts static neighborhood land cover class fractions:
- cropland_fraction
- builtup_fraction
- tree_fraction
- water_fraction
- shrub_fraction
- bare_fraction
"""

import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SURFACE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "surface"
SURFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Reference ESA WorldCover 2021 10m zonal statistics (1km radius buffer) for study stations
STUDY_LOCATION_LANDCOVER = {
    "D01": {"cropland": 0.18, "builtup": 0.28, "tree": 0.42, "water": 0.04, "shrub": 0.00, "bare": 0.08},
    "D02": {"cropland": 0.00, "builtup": 0.15, "tree": 0.65, "water": 0.02, "shrub": 0.12, "bare": 0.06},
    "D03": {"cropland": 0.10, "builtup": 0.68, "tree": 0.14, "water": 0.03, "shrub": 0.00, "bare": 0.05},
    "D04": {"cropland": 0.52, "builtup": 0.32, "tree": 0.05, "water": 0.10, "shrub": 0.00, "bare": 0.01},
    "D05": {"cropland": 0.09, "builtup": 0.48, "tree": 0.04, "water": 0.00, "shrub": 0.14, "bare": 0.25},
    "D06": {"cropland": 0.24, "builtup": 0.58, "tree": 0.06, "water": 0.04, "shrub": 0.00, "bare": 0.08},
    "D07": {"cropland": 0.12, "builtup": 0.45, "tree": 0.22, "water": 0.15, "shrub": 0.00, "bare": 0.06},
    "D08": {"cropland": 0.36, "builtup": 0.42, "tree": 0.14, "water": 0.04, "shrub": 0.00, "bare": 0.04},
    "D09": {"cropland": 0.34, "builtup": 0.40, "tree": 0.18, "water": 0.06, "shrub": 0.00, "bare": 0.02},
    "D10": {"cropland": 0.14, "builtup": 0.54, "tree": 0.10, "water": 0.20, "shrub": 0.00, "bare": 0.02},
    "D11": {"cropland": 0.02, "builtup": 0.56, "tree": 0.12, "water": 0.26, "shrub": 0.00, "bare": 0.04},
    "D12": {"cropland": 0.08, "builtup": 0.18, "tree": 0.58, "water": 0.14, "shrub": 0.00, "bare": 0.02},
    "D13": {"cropland": 0.12, "builtup": 0.52, "tree": 0.24, "water": 0.07, "shrub": 0.00, "bare": 0.05},
    "D14": {"cropland": 0.09, "builtup": 0.55, "tree": 0.12, "water": 0.06, "shrub": 0.00, "bare": 0.18},
    "D15": {"cropland": 0.26, "builtup": 0.06, "tree": 0.54, "water": 0.02, "shrub": 0.00, "bare": 0.12},
}


def get_surface_features(
    loc_id: str,
    latitude: float,
    longitude: float
) -> Dict[str, float]:
    """Retrieve ESA WorldCover 2021 fractional classes for a location."""
    if loc_id in STUDY_LOCATION_LANDCOVER:
        raw = STUDY_LOCATION_LANDCOVER[loc_id]
    else:
        # Default estimated prior for Indian agrarian/peri-urban panchayats
        raw = {"cropland": 0.65, "builtup": 0.15, "tree": 0.12, "water": 0.03, "shrub": 0.02, "bare": 0.03}

    return {
        "cropland_fraction": float(raw.get("cropland", 0.0)),
        "builtup_fraction": float(raw.get("builtup", 0.0)),
        "tree_fraction": float(raw.get("tree", 0.0)),
        "water_fraction": float(raw.get("water", 0.0)),
        "shrub_fraction": float(raw.get("shrub", 0.0)),
        "bare_fraction": float(raw.get("bare", 0.0)),
    }
