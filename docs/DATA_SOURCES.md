# Data Sources and Provenance

This document catalogs every dataset, API endpoint, and spatial asset used across PanchayatCast, including spatial/temporal resolutions, schema contracts, caching conventions, and known data limitations.

---

## 1. Summary of Data Sources

| Source | Role in System | Spatial Resolution | Temporal Resolution | Access / Storage |
| :--- | :--- | :--- | :--- | :--- |
| **Open-Meteo Historical Forecast (`gfs_seamless`)** | Coarse NWP baseline for 15-location benchmark & live app | ~25 km ($0.25^\circ$) | Hourly | Public API; cached in `data/raw/gfs_*.json` |
| **Open-Meteo Single-Runs API (`gfs_global`)** | Fixed-initialization forecast runs with explicit `lead_hours` | ~25 km ($0.25^\circ$) | Hourly per run | Public API (`single-runs-api.open-meteo.com`) |
| **Open-Meteo Archive (`era5_land`)** | Fine-scale reanalysis training & evaluation target | ~9–11 km ($0.1^\circ$) | Hourly | Public API; cached in `data/raw/era5_*.json` |
| **Open-Meteo Elevation API (SRTM DEM)** | 13-point local terrain stencil (`elevation`, `slope`, `aspect`, `relief`) | ~30 m–90 m | Static | Public API; cached in `data/raw/terrain_*.json` |
| **ESA WorldCover 2021 (v200)** | Land-cover class fractions (`cropland`, `builtup`, `tree`, `water`, `bare`) | 10 m aggregated to 1 km | Static (2021) | Pre-extracted for 15 study stations in `src/surface/landcover.py` |
| **MODIS Terra (`MOD13Q1`)** | 16-day NDVI vegetation phenology | 250 m aggregated to 1 km | 16-day composite (linearly interpolated to hourly) | Pre-extracted August 2024 profiles in `src/surface/ndvi_processor.py` |
| **LGD Gram Panchayat Boundaries** | Nationwide administrative polygon lookup & area normalization | Vector polygons (WGS84 / `EPSG:4326`) | 2024 snapshot | `data/raw/LGD_panchayats.parquet` (434.8 MB / 414.6 MiB, 319,287 rows) |
| **ISRO MOSDAC AWS Archives** | Ground-station verification target (`IMDE1401_141579` pilot) | Point station (2 decimal place coords) | Hourly | `data/raw/mosdac_aws/` & `data/interim/` |

---

## 2. Numerical Weather Prediction (NWP) & Reanalysis

### 2.1 Coarse Forecast Baseline (`fetch_weather.py`)
- **Endpoint**: `https://historical-forecast-api.open-meteo.com/v1/forecast`
- **Model**: `gfs_seamless`
- **Variables**: `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`
- **Study Window**: `2024-08-01` to `2024-08-31` (744 hourly steps $\times$ 15 locations = `11,160` observations in `data/processed/unified_dataset_2024-08-01_2024-08-31.csv`).
- **Important Technical Detail (`elevation_diff`)**: When querying `gfs_seamless` via Open-Meteo without disabling DEM downscaling, the `"elevation"` field returned in the JSON header is already interpolated to the ~90 m DEM at the query coordinate, matching the center elevation returned by the Open-Meteo Elevation API. Consequently, in `unified_dataset_2024-08-01_2024-08-31.csv`, `elevation_local == elevation_gfs` and `elevation_diff == 0.0` across all 15 stations. Sub-grid topographic variance is therefore captured through the multi-radius neighborhood derivatives (`slope_deg`, `aspect_sin`, `aspect_cos`, `relief_500m`, `relief_1000m`, `elev_std_1000m`, and `tpi_ratio`) rather than `elevation_diff`.

### 2.2 Fixed-Run Forecast Ingestion (`openmeteo_runs.py` & `historical_forecast.py`)
Seamless historical forecast concatenation blends short lead times across consecutive daily runs and does not preserve forecast initialization timestamps. To support strict lead-time verification (`lead_hours = (valid_time - forecast_init_time)`):
- `src/data/openmeteo_runs.py` queries `https://single-runs-api.open-meteo.com/v1/forecast` with an explicit `forecast_init_time`.
- `src/data/schema.py` (`require_initialized_forecast`) rejects any dataset where `lead_hours` is uniformly zero (e.g., reanalysis or uninitialized series masquerading as a forecast).
- `src/data/historical_forecast.py` records SHA-256 provenance hashes (`reports/forecast_provenance.json`) whenever historical forecast windows are fetched.

### 2.3 Reanalysis Reference Target (`ERA5-Land`)
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive` (`models=era5_land`)
- **Role**: Provides spatially continuous hourly `temperature_2m` targets for the 15-location benchmark where dense public sub-block AWS observations are unavailable.
- **Limitation**: At ~9–11 km native resolution, `ERA5-Land` resolves mesoscale valley/ridge gradients much better than 25 km GFS, but remains a gridded land-surface reanalysis rather than an in-situ thermometer measurement.

---

## 3. Terrain, Land Cover, and Vegetation Inputs

### 3.1 SRTM 30 m Metric Neighborhood Stencil (`src/terrain/srtm_extractor.py`)
For any target coordinate $(\phi, \lambda)$, `extract_terrain_features` converts metric offsets ($r \in \{250\text{ m}, 500\text{ m}, 1000\text{ m}\}$) into latitude/longitude offsets using:
$$\Delta \phi = \frac{d}{111320.0}, \quad \Delta \lambda = \frac{d}{111320.0 \cdot \cos(\phi)}$$
A single batched request of 13 points (center + N, E, S, W at 3 radii) is sent to `https://api.open-meteo.com/v1/elevation`. From the 250 m cardinal points, central finite differences (`dz/dx`, `dz/dy`) yield local slope (`slope_deg`) and aspect (`aspect_deg`, encoded cyclically as `aspect_sin` and `aspect_cos`), while the 500 m and 1000 m rings yield `relief_500m`, `relief_1000m`, and `elev_std_1000m`.

### 3.2 Land-Cover & NDVI Lookup Scope (`src/surface/`)
- **15 Benchmark Stations (`D01`–`D15`)**: `STUDY_LOCATION_LANDCOVER` (`src/surface/landcover.py`) and `MODIS_NDVI_SERIES` (`src/surface/ndvi_processor.py`) store pre-extracted 1 km ESA WorldCover 2021 class fractions and August 2024 MODIS 16-day NDVI values (`Day 1`, `Day 16`, `Day 31`, linearly interpolated to hourly timestamps).
- **Custom Map Coordinates (`loc_id="custom"`)**: When querying arbitrary coordinates via `POST /api/downscale`, the backend currently falls back to a representative rural agricultural surface profile (`cropland_fraction=0.65`, `builtup_fraction=0.08`, `tree_fraction=0.15`, `water_fraction=0.04`, `bare_fraction=0.08` and NDVI `0.48–0.58`) rather than querying a live raster tile server.

---

## 4. Nationwide Gram Panchayat Boundary Dataset

### 4.1 Source & Scale
- **Raw File**: `data/raw/LGD_panchayats.parquet` (`434,763,842` bytes / `434.8 MB` / `414.6 MiB`, excluded from Git via `.gitignore` due to GitHub's 100 MB file limit; sourced from the open community LGD Panchayat boundary mirror `datameet` / `ramSeraph/indian_admin_boundaries`).
- **Processed File**: `data/processed/panchayat_boundaries.parquet` (`775,758,738` bytes / `775.8 MB` / `739.8 MiB`, also excluded from Git via `.gitignore`; generated locally by running `python scripts/build_boundary_dataset.py`).
- **Coverage**: `319,287` polygon records across `28` States/UTs, representing `226,984` unique `panchayat_id` (Gram Panchayat LGD codes) and `6,629` unique `block_id` (Block/Mandal LGD codes).

### 4.2 Audit Findings & Normalization (`src/data/boundary_pipeline.py`)
Running `scripts/inspect_boundaries.py` (`reports/boundary_inspection.md`) identified three critical properties in the raw dataset:
1. **Multi-Polygon / Cadastral Splitting**: `27,624` unique LGD Panchayat codes appear across multiple rows (`94,276` rows total) because several states map individual revenue villages or non-contiguous enclaves as separate polygon records sharing the same Gram Panchayat LGD code. The pipeline preserves all rows without dissolving geometries (`row_id` primary key).
2. **Invalid Geometries**: `887` rows (`0.28%`) fail `shapely.is_valid` (primarily self-intersecting rings) and `7` rows have `panchayat_id == "0"`. All rows are preserved and flagged via `geometry_valid` (`bool`) and `missing_id_flag` (`bool`).
3. **Web Mercator `SHAPE_Area` Inflation**: The source `SHAPE_Area` column was computed in `EPSG:3857` (Web Mercator), which inflates planar area by $\sec^2(\phi)$ (~`+24.2%` at $26^\circ\text{N}$ in Bihar). `boundary_pipeline.py` computes true geodesic area (`area_sq_km`) on the `WGS84` ellipsoid using `pyproj.Geod(ellps="WGS84").geometry_area_perimeter`.

---

## 5. ISRO MOSDAC Automatic Weather Station (AWS) Data

### 5.1 Pilot Dataset (`IMDE1401_141579`)
- **Files**:
  - Raw export: `data/raw/mosdac_aws/IMDE1401_141579_20260409_105421.xls`
  - Normalized CSV: `data/processed/mosdac_aws_normalized.csv`
  - Quality Report: `reports/mosdac_aws_quality_report.json`
- **Station Identity**: `IMDE1401_141579(PUSA_BH_AGRO)` located at `latitude = 25.98`, `longitude = 85.67` (Pusa Block, Samastipur District, Bihar).
- **Temporal Window**: `2013-11-01T01:00:00Z` to `2013-11-08T00:00:00Z` (`168` hourly observations, 100% hourly completeness, 0 duplicate timestamps).

### 5.2 Quality Control & Spatial Validation Status
- **Sensor Anomaly Detection (`src/data/mosdac_aws.py`)**: All 168 temperature values (`14.7 °C` to `29.7 °C`) pass physical range checks, whereas all 168 `BATTERY_VOLTAGE(V)` values (`9953.0 V`) are automatically flagged as out-of-range sensor fill/error values.
- **Spatial Mapping Uncertainty (`reports/pilot_station_spatial_validation.json`)**: Point-in-polygon lookup places `(25.98, 85.67)` strictly inside Gram Panchayat **Mohmadpur Deopar** (`panchayat_id = 100318`, Block **Pusa** `block_id = 919`, Samastipur, Bihar). However, the reported coordinate is rounded to 2 decimal places ($\pm 0.005^\circ \approx 748.4\text{ m}$ uncertainty radius), while the point lies only `44.26 m` from the polygon boundary (`15` neighboring Panchayat polygons intersect the 748.4 m uncertainty circle). Therefore, `scripts/validate_station_mapping.py` assigns `validation_status = "rejected_coordinate_uncertainty"`, preventing unverified spatial joins until higher-precision station coordinates are supplied.

---

## 6. Dataset Licensing, Attribution, and Redistribution Terms

| Dataset / Source | License / Terms of Use | Required Attribution | Public Redistribution Status in This Repository |
| :--- | :--- | :--- | :--- |
| **ISRO MOSDAC AWS Observations** (`data/raw/mosdac_aws.csv`, `data/processed/mosdac_aws_normalized.csv`, `data/processed/historical_aws_normalized.parquet`) | **ISRO / SAC MOSDAC Data Policy** (Registration required; free redistribution of raw/derived MOSDAC datasets is **prohibited** without an explicit signed redistribution clause from SAC/ISRO). | *"Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC), Space Applications Centre (SAC), ISRO, Government of India."* | **Excluded via `.gitignore`**. Users must register at `https://www.mosdac.gov.in/`, download AWS exports directly under their own account, and run `python scripts/process_mosdac.py`. Only aggregate QC metadata (`reports/mosdac_aws_quality_report.json`) and synthetic unit-test fixtures (`tests/test_mosdac_aws.py`) are tracked. |
| **Open-Meteo APIs & Data** (`Historical Forecast`, `Single-Runs`, `Archive`, `Elevation` caches in `data/raw/`) | **1. API Terms of Use**: Hosted API access is offered free of charge for **non-commercial use only** (`https://open-meteo.com/en/terms`); commercial use of the hosted API requires a paid subscription.<br/>**2. Data License**: Weather data is licensed under **CC BY 4.0** (`https://open-meteo.com/en/license`).<br/>**3. Server Software**: Open-Meteo server source code is licensed under **AGPLv3** (distinct from API data). | *"Weather data by Open-Meteo.com (https://open-meteo.com/)"* (displayed in UI footer and README). | Cached responses for the 15 benchmark stations are included with `CC BY 4.0` attribution for non-commercial research/evaluation. Users must review Open-Meteo's official terms before any commercial use. |
| **NOAA NCEP GFS** (underlying forecast model in Open-Meteo) | **U.S. Public Domain** (`17 U.S.C. § 105`, NOAA Open Data Policy). | *"NOAA / NCEP Global Forecast System (GFS)."* | Unrestricted public domain. |
| **ECMWF Copernicus ERA5-Land** (underlying reanalysis in Open-Meteo Archive) | **Copernicus C3S License** (free and open access with attribution). | *"Contains modified Copernicus Climate Change Service information (2024); neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."* | Permitted with Copernicus attribution notice. |
| **NASA / USGS SRTM Digital Elevation Model** | **U.S. Public Domain** (NASA / USGS LP DAAC). | *"NASA / USGS Shuttle Radar Topography Mission (SRTM)."* | Unrestricted public domain. |
| **ESA WorldCover 10 m 2021 (`v200`)** | **Creative Commons Attribution 4.0 International (CC BY 4.0)**. | *"© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium."* | Permitted under `CC BY 4.0` with attribution (`15`-station summary fractions in `src/surface/landcover.py`). |
| **NASA MODIS Terra `MOD13Q1` v061** | **U.S. Public Domain / NASA EOSDIS Open Data Policy**. | *"NASA LP DAAC MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid (`MOD13Q1`)."* | Unrestricted with citation (`15`-station values in `src/surface/ndvi_processor.py`). |
| **LGD Gram Panchayat Boundaries** (`LGD_panchayats.parquet`, sourced from Government of India LGD / Bharat Maps via community mirror `ramSeraph/indian_admin_boundaries`) | **Government Open Data License – India (GODL-India) / Community Mirror (Unspecified Upstream Repo License)**. | *"Local Government Directory (LGD), Ministry of Panchayati Raj / NIC, Government of India; community GeoParquet mirror by `ramSeraph/indian_admin_boundaries` and DataMeet."* | **Excluded via `.gitignore`** (`434.8 MB` / `414.6 MiB` raw `LGD_panchayats.parquet`; `775.8 MB` / `739.8 MiB` processed `panchayat_boundaries.parquet`; `52.9 KB` `pusa_panchayats.geojson`; `2.8 KB` `pilot_mohmadpur_deopar.geojson`). Because the files exceed GitHub size limits and the upstream GitHub mirror does not attach a formal SPDX license file to the pre-packaged GeoParquet, no raw or derived boundary polygons are redistributed in this repository. |


