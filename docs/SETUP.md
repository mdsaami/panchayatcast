# Local Setup and Execution Guide

This guide covers environment setup, running the web application, executing the machine learning benchmarks, running the geospatial/AWS verification CLI utilities, and running the test suite.

---

## 1. Prerequisites

- **Python**: `3.11` or `3.12` (tested on Python `3.11.9` on Windows).
- **Disk Space**:
  - ~500 MB for Python dependencies and cached 15-location benchmark data.
  - ~1.3 GB additional space if using the full nationwide Gram Panchayat boundary GeoParquet files (`data/raw/LGD_panchayats.parquet` at 434.7 MB and `data/processed/panchayat_boundaries.parquet` at 775.7 MB).
- **Network Access**: Optional for default 15-location benchmarks and tests (all 15-location GFS, ERA5-Land, and SRTM files are pre-cached in `data/raw/`). Live internet access is only required when querying uncached custom coordinates on the map or fetching new historical forecast runs from Open-Meteo.

> [!TIP]
> **Windows + OneDrive Note**: If this repository is located inside a OneDrive folder with *Files On-Demand* enabled, ensure that `models/`, `data/`, and `.venv/` are marked **"Always keep on this device"** in File Explorer, or create your virtual environment outside OneDrive. Cloud-only placeholder files (`FILE_ATTRIBUTE_OFFLINE`) can cause Python imports or `pyarrow` Parquet reads to stall while waiting for background cloud hydration.

---

## 2. Installation

### Create and Activate a Virtual Environment

**PowerShell (Windows):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Bash (Linux / macOS):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional Environment Configuration

Copy `.env.example` to `.env` if you want to customize model/config paths or CORS settings:
```bash
cp .env.example .env
```
No external API keys are required for default operation.

---

## 3. Running the Web Application

Start the FastAPI backend (which also mounts the static frontend at `/`):

```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the application in your browser:
- **Interactive Dashboard**: `http://127.0.0.1:8000/`
- **Health Check Endpoint**: `http://127.0.0.1:8000/health`
- **Interactive OpenAPI Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`

### Example API Request (`POST /api/downscale`)

```bash
curl -X POST "http://127.0.0.1:8000/api/downscale" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 30.3165,
    "longitude": 78.0322,
    "location_name": "Dehradun",
    "crop": "Rice (Paddy)",
    "crop_stage": "Flowering / Heading",
    "start_date": "2024-08-15",
    "end_date": "2024-08-17"
  }'
```

---

## 4. Running the Experimental Benchmarks

All experiment scripts can be run from the repository root. They read the cached dataset in `data/processed/unified_dataset_2024-08-01_2024-08-31.csv` (or build it from `data/raw/` if missing) and write metrics to `results/metrics/` and figures to `results/figures/`.

| Command | Description | Output Files |
| :--- | :--- | :--- |
| `python scripts/prefetch_data.py` | Prefetches/verifies cached GFS, ERA5-Land, and SRTM files for the 15 study locations and builds the unified CSV. | `data/processed/unified_dataset_2024-08-01_2024-08-31.csv` |
| `python experiments/run_ablation_study.py` | Runs the 5-fold Leave-Location-Out (LLO) spatial CV ablation across `Model_0`–`Model_5`, chronological split evaluation, conformal calibration, and trains `models/final_residual_model_lightgbm.joblib`. | `results/metrics/ablation_study_summary.csv`, `ablation_spatial_folds.csv`, `feature_importances_model5.csv`, `models/final_residual_model_lightgbm.joblib` |
| `python experiments/run_benchmark.py` | Benchmarks `RawGFS`, `MeanBiasCorrection`, `RidgeRegression`, `RandomForest`, `HistGBDT`, `LightGBM`, and `BC-UNet` under spatial LLO CV. | `results/metrics/algorithm_comparison.csv` |
| `python experiments/run_reproduce_15loc.py` | Evaluates `Model_6_Physical_Terrain` (relative topographic position + `MultiQuantileResidualModel` + `StackedResidualEnsemble`) against `Model_1` and `Model_5`. | `results/metrics/historical_reproduction_folds.csv` |
| `python experiments/run_new_architecture_benchmark.py` | Compares single LightGBM (`Model_5`) against the stacked ensemble (`Model_6`) and checks 90% quantile interval empirical coverage. | `results/metrics/new_architecture_benchmark.csv` |
| `python experiments/generate_visualizations.py` | Generates all 13 diagnostic plots (ablation bars, diurnal biases, elevation lapse scatter, uncertainty bands, feature importances, spatial heatmaps). | `results/figures/01_*.png` – `13_*.png` |

---

## 5. Running the Geospatial & Station Verification CLI Scripts

The `scripts/` directory provides CLI tools for the nationwide Gram Panchayat boundary dataset and ISRO MOSDAC AWS station verification workflow:

### 5.1 Inspect and Build Nationwide Panchayat Boundaries
If `data/raw/LGD_panchayats.parquet` is present locally:

```bash
# Audit schema, geometry validity, LGD uniqueness, and area distortion
python scripts/inspect_boundaries.py

# Build normalized dataset with WGS84 geodesic areas and representative points
python scripts/build_boundary_dataset.py
```
Outputs `data/processed/panchayat_boundaries.parquet` and `data/processed/panchayat_lookup.csv`.

### 5.2 Point-in-Polygon Panchayat Lookup
Query which Gram Panchayat polygon contains a given coordinate (`lat`, `lon`):

```bash
python scripts/find_panchayat.py --lat 25.98 --lon 85.67
```

### 5.3 Validate Station-to-Panchayat Spatial Mapping
Verify candidate weather station mappings in `config/station_panchayat_mapping.csv` against actual Panchayat polygon boundaries and coordinate precision thresholds:

```bash
python scripts/validate_station_mapping.py
```
Writes `reports/pilot_station_spatial_validation.json`.

### 5.4 Normalize MOSDAC AWS Observations & Run Station Benchmark
```bash
# Normalize raw MOSDAC station export and generate QC report
python scripts/process_mosdac.py

# Align historical NWP forecasts with verified AWS observations
python scripts/run_historical_benchmark.py
```
Writes `reports/mosdac_aws_quality_report.json` and `reports/benchmark_metrics_station.json`.

---

## 6. Running the Test Suite

Execute the unit and integration test suite with `pytest`:

```bash
pytest tests/ -v
```

The test suite includes 10 test modules (`75` test cases) covering:
- `tests/test_boundary_pipeline.py`: Schema mapping, invalid geometry flagging, geodesic ellipsoid area calculation, and point-in-polygon lookup.
- `tests/test_canonical.py`: Legacy-to-canonical column mapping and model baseline precedence.
- `tests/test_historical_forecast.py`: Provenance metadata and SHA-256 dataset hashing.
- `tests/test_historical_pipeline.py`: End-to-end temporal forecast-to-observation alignment (`merge_asof`) and rejection of unapproved station mappings.
- `tests/test_mosdac_aws.py`: MOSDAC AWS parsing, duplicate timestamp removal, physical outlier QC flagging, and canonical target temperature mapping.
- `tests/test_openmeteo_runs.py`: Explicit `forecast_init_time` validation, UTC `valid_time` enforcement, and `lead_hours` calculation.
- `tests/test_pipeline.py`: Terrain stencil extraction, cyclical feature encoding, residual model training/inference, conformal interval calibration, advisory engine rules, and FastAPI endpoint integration.
- `tests/test_schema.py`: Rejection of reanalysis/seamless zero-lead series when initialized forecast runs are required.
- `tests/test_spatial_station_validation.py`: Synthetic polygon boundary tests (`contains`, `touches`, `disjoint`) and coordinate quantization uncertainty rejection.
- `tests/test_station_mapping.py`: CSV mapping schema and validation status enforcement (`approved` vs `pending_spatial_check` vs `rejected_coordinate_uncertainty`).
