"""Pre-fetch and cache all raw GFS, ERA5-Land, and SRTM terrain data using Python standard library."""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_weather import fetch_gfs_forecast, fetch_era5_reference
from src.terrain.srtm_extractor import extract_terrain_features

CONFIG_PATH = PROJECT_ROOT / "config" / "locations.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    locations = cfg["locations"]
    print(f"Pre-fetching data for {len(locations)} locations across India (August 2024)...")

    for idx, loc in enumerate(locations, 1):
        loc_id = loc["id"]
        loc_name = loc["name"]
        lat = loc["latitude"]
        lon = loc["longitude"]
        print(f"[{idx}/15] Fetching {loc_id} ({loc_name}, {lat}, {lon})...", end="", flush=True)

        # 1. Fetch GFS forecast
        t0 = time.time()
        fetch_gfs_forecast(lat, lon, start_date="2024-08-01", end_date="2024-08-31", cache_tag=loc_id)
        # 2. Fetch ERA5-Land reference
        fetch_era5_reference(lat, lon, start_date="2024-08-01", end_date="2024-08-31", cache_tag=loc_id)
        # 3. Extract SRTM terrain
        extract_terrain_features(lat, lon, loc_id=loc_id)
        print(f" Done ({time.time() - t0:.2f}s)")

        # Friendly pause to respect API rate limits
        time.sleep(0.5)

    print("\nAll 15 locations successfully fetched and cached locally in data/raw/!")


if __name__ == "__main__":
    main()
