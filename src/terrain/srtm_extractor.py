"""Topographic and SRTM elevation feature extraction module.

Extracts:
- Point elevation
- Metric neighborhood sampling (250m, 500m, 1000m)
- Local slope (degrees) via metric gradient
- Local aspect (degrees), aspect_sin, aspect_cos
- Local relief (max - min) at 500m and 1000m scales
- Local elevation standard deviation
"""

import math
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TERRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "terrain"
TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "PanchayatCast-WeatherDownscaling-Research/1.0"


def _meters_to_lat_lon_offsets(lat: float, radius_m: float) -> tuple[float, float]:
    """Convert metric radius to latitude and longitude delta degrees using WGS84 ellipsoid approximation."""
    lat_rad = math.radians(lat)
    # 1 degree of latitude is approximately 111,139 meters
    d_lat = radius_m / 111139.0
    # 1 degree of longitude depends on latitude
    d_lon = radius_m / (111139.0 * max(0.01, math.cos(lat_rad)))
    return d_lat, d_lon


def query_elevation_batch(coords: List[tuple[float, float]]) -> List[float]:
    """Query Open-Meteo SRTM Elevation API for a batch of (lat, lon) coordinates."""
    lats_str = ",".join([f"{lat:.5f}" for lat, lon in coords])
    lons_str = ",".join([f"{lon:.5f}" for lat, lon in coords])
    
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats_str}&longitude={lons_str}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("elevation", [])


def extract_terrain_features(
    latitude: float,
    longitude: float,
    loc_id: str = "custom",
    force_refresh: bool = False
) -> Dict[str, float]:
    """Extract metric terrain derivatives around a target location.
    
    Samples center and 8-point cardinal/intercardinal compass directions at 250m, 500m, and 1000m.
    Caches features to data/raw/terrain/terrain_<loc_id>.json.
    """
    cache_file = TERRAIN_DIR / f"terrain_{loc_id}_{latitude:.4f}_{longitude:.4f}.json"
    if not force_refresh and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Neighborhood offsets at 250m, 500m, 1000m
    points_to_query = [(latitude, longitude)] # index 0 = center
    
    radii = [250.0, 500.0, 1000.0]
    offsets_meta = []
    
    for r in radii:
        d_lat, d_lon = _meters_to_lat_lon_offsets(latitude, r)
        # N, S, E, W, NE, NW, SE, SW
        cardinals = [
            (d_lat, 0.0, f"N_{int(r)}"),
            (-d_lat, 0.0, f"S_{int(r)}"),
            (0.0, d_lon, f"E_{int(r)}"),
            (0.0, -d_lon, f"W_{int(r)}"),
            (d_lat * 0.7071, d_lon * 0.7071, f"NE_{int(r)}"),
            (d_lat * 0.7071, -d_lon * 0.7071, f"NW_{int(r)}"),
            (-d_lat * 0.7071, d_lon * 0.7071, f"SE_{int(r)}"),
            (-d_lat * 0.7071, -d_lon * 0.7071, f"SW_{int(r)}")
        ]
        for dy, dx, name in cardinals:
            points_to_query.append((latitude + dy, longitude + dx))
            offsets_meta.append((name, r))

    elevations = query_elevation_batch(points_to_query)
    center_z = elevations[0]
    
    # Map offsets
    elev_dict = {"center": center_z}
    for (name, r), elev in zip(offsets_meta, elevations[1:]):
        elev_dict[name] = elev

    # Compute metric derivatives at 500m scale:
    # dz/dx: (E - W) / (2 * 500m)
    # dz/dy: (N - S) / (2 * 500m)
    e_500 = elev_dict["E_500"]
    w_500 = elev_dict["W_500"]
    n_500 = elev_dict["N_500"]
    s_500 = elev_dict["S_500"]

    dz_dx = (e_500 - w_500) / 1000.0 # 2 * 500m
    dz_dy = (n_500 - s_500) / 1000.0

    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    # Aspect: meteorological direction (0 = North, 90 = East, etc.)
    # In math: aspect = atan2(-dz/dx, dz/dy)
    aspect_rad = math.atan2(-dz_dx, dz_dy)
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    aspect_sin = math.sin(aspect_rad)
    aspect_cos = math.cos(aspect_rad)

    # Neighborhood relief and std at 500m and 1000m
    elevs_500 = [center_z] + [elev_dict[f"{d}_500"] for d in ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]
    elevs_1000 = [center_z] + [elev_dict[f"{d}_1000"] for d in ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]

    relief_500m = max(elevs_500) - min(elevs_500)
    relief_1000m = max(elevs_1000) - min(elevs_1000)

    mean_1000m = sum(elevs_1000) / len(elevs_1000)
    std_1000m = math.sqrt(sum((z - mean_1000m)**2 for z in elevs_1000) / len(elevs_1000))

    features = {
        "elevation_local": float(center_z),
        "slope_deg": float(round(slope_deg, 4)),
        "aspect_deg": float(round(aspect_deg, 2)),
        "aspect_sin": float(round(aspect_sin, 4)),
        "aspect_cos": float(round(aspect_cos, 4)),
        "relief_500m": float(round(relief_500m, 2)),
        "relief_1000m": float(round(relief_1000m, 2)),
        "elev_std_1000m": float(round(std_1000m, 2)),
        "dz_dx": float(round(dz_dx, 6)),
        "dz_dy": float(round(dz_dy, 6))
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    return features
