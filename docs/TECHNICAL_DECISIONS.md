# Technical Decisions and Engineering Trade-Offs

This document records the principal architectural and methodological decisions made in PanchayatCast, including empirical findings from the ablation experiments that shaped the codebase.

---

## 1. Residual Learning Over Direct Temperature Prediction

### Decision
Instead of training a machine learning model to predict local 2 m temperature $T_{\text{local}}$ directly from geographic and meteorological inputs, `DownscalingResidualModel` (`src/models/residual_model.py`) predicts the sub-grid temperature correction residual:
$$\hat{r} = f_\theta(X_{\text{met}}, X_{\text{time}}, X_{\text{terrain}}, X_{\text{surface}})$$
$$\hat{T}_{\text{refined}} = T_{\text{baseline}} + \hat{r}$$
where $T_{\text{baseline}}$ is the coarse NWP forecast (`temperature_2m_gfs` / `baseline_temperature`).

### Rationale & Trade-Offs
- **Preserves Synoptic Skill**: Global NWP models already capture large-scale synoptic advection, frontal passages, and seasonal cycles. Predicting only the residual $\Delta T = T_{\text{target}} - T_{\text{baseline}}$ bounds the target variance (`mean = -0.35 °C`, `std = 1.48 °C` across the 15-location dataset) compared to raw temperatures (`11 °C` to `38 °C`).
- **Graceful Degradation**: If the ML residual approaches zero in flat, homogeneous terrain, the prediction naturally defaults to the physically consistent NWP baseline rather than extrapolating wildly.
- **Empirical Gain**: In 5-fold Leave-Location-Out (LLO) spatial cross-validation (`results/metrics/ablation_study_summary.csv` and `results/metrics/algorithm_comparison.csv`), the Raw GFS baseline exhibits `1.6614 ± 0.4529 °C` MAE (`2.1922 °C` RMSE). A residual LightGBM model using meteorological and cyclical temporal features (`Model_1_Weather_Time`) reduces spatial MAE to **`0.7907 ± 0.0352 °C`** (`52.41%` overall reduction; `47.25%` mean fold-wise reduction) and RMSE to **`1.0347 ± 0.0437 °C`**, while cutting inter-fold MAE standard deviation by over 12x.

---

## 2. Leave-Location-Out (LLO) Spatial Cross-Validation vs. Chronological Splits

### Decision
All model selection and feature ablation experiments (`src/validation/spatial_cv.py` and `src/validation/temporal_cv.py`) evaluate both:
1. **5-Fold Leave-Location-Out (LLO) Spatial CV**: `GroupKFold(n_splits=5)` grouped by `location_id` (`12` train locations, `3` completely unseen test locations per fold).
2. **Chronological Forecast Split**: Train on Days 1–22 of August 2024 (`528` hours/station), evaluate on Days 23–31 (`216` hours/station), alongside a combined spatiotemporal holdout (`3` unseen stations on Days 23–31).

### Rationale & Empirical Finding ("The Elevation Paradox")
Random row-wise train/test splits leak station-specific static predictors (elevation, slope, land-cover fractions) between adjacent hours. Evaluating temporal, spatial, and spatiotemporal splits exposed a critical divergence across feature sets (`results/metrics/ablation_study_summary.csv`, `results/metrics/historical_reproduction_folds.csv`, and `results/metrics/new_architecture_benchmark.csv`):

| Model Configuration | Features | Spatial LLO CV MAE (°C) | Spatial LLO CV RMSE (°C) | Temporal Split MAE (°C) | Spatiotemporal MAE (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Model_0_GFS_Baseline` | — | `1.6614 ± 0.4529` | `2.1922 ± 0.5625` | `1.5078` | `1.5078` |
| `Model_1_Weather_Time` | 9 | **`0.7907 ± 0.0352`** | **`1.0347 ± 0.0437`** | `0.7695` | **`0.8649`** |
| `Model_B1_Weather_Time_Elevation` | 10 | `0.8080 ± 0.0332` | `1.0527 ± 0.0454` | `0.7524` | `0.8760` |
| `Model_2_Weather_Time_Terrain` | 17 | `0.8483 ± 0.0843` | `1.1004 ± 0.0965` | `0.7489` | `0.8841` |
| `Model_3_Weather_Time_Surface` | 14 | `0.8426 ± 0.1034` | `1.1039 ± 0.1268` | **`0.7388`** | `0.8966` |
| `Model_4_Weather_Time_Terrain_Surface` | 22 | `0.8481 ± 0.1116` | `1.1110 ± 0.1397` | `0.7460` | `0.8948` |
| `Model_5_Full_Physiographic_NDVI` | 23 | `0.8511 ± 0.1149` | `1.1131 ± 0.1460` | `0.7609` | `0.9141` |
| `Model_6_Physical_Terrain` (LightGBM) | 13 | `0.8154 ± 0.0694` | `1.0674 ± 0.0798` | — | — |
| `NewArch_Stacked_Ensemble` (`Model_6` features) | 13 | `0.8113 ± 0.0567` | `1.0619 ± 0.0641` | — | — |

### Why Did Adding Static Elevation Degrade Spatial LLO CV?
1. **Small $N$ of Unique Static Locations**: With only 15 distinct stations (`12` per training fold), decision trees split on exact static values (`elevation_local`, `builtup_fraction`) to memorize station-level mean biases during training. On 3 held-out locations with unseen elevations (e.g., Ooty at `2,238 m` or Shimla at `2,204 m`), those static tree thresholds do not generalize as well as surface pressure (`surface_pressure_gfs`) and relative humidity, which already encode altitude and air-mass properties continuously. Across all 5 spatial folds (`results/metrics/historical_reproduction_folds.csv`), adding raw point elevation (`Model_B1`) slightly worsened MAE relative to `Model_1`.
2. **Open-Meteo `gfs_seamless` Pre-Downscaled Elevation**: In `data/processed/unified_dataset_2024-08-01_2024-08-31.csv`, Open-Meteo's `gfs_seamless` endpoint already reports DEM-interpolated elevation equal to `elevation_local`, making `elevation_diff = 0.0` across all rows (`feature_importances_model5.csv` confirms `importance = 0` for `elevation_diff`).
3. **Mitigation via Relative Topographic Position (`Model_6_Physical_Terrain`)**: Replacing raw elevation with scale-free relative indices (`tpi_ratio = elevation_diff / (relief_1000m + 10.0)` and `wind_exposure`) combined with a Ridge-regularized `StackedResidualEnsemble` (`src/models/advanced_residual.py`) reduced the spatial overfitting penalty from `0.8511 °C` (`Model_5`) down to `0.8154 ± 0.0694 °C` (`Model_6` LightGBM) and `0.8113 ± 0.0567 °C` (`NewArch_Stacked_Ensemble`).

---

## 3. Split-Conformal & Multi-Quantile Uncertainty Estimation

### Decision
Implement both split-conformal residual calibration (`src/uncertainty/conformal.py`) and multi-quantile GBDT (`MultiQuantileResidualModel` in `src/models/advanced_residual.py`).

### Rationale & Current Implementation Scope
- **Risk-Threshold Propagation**: Agricultural advisories (such as frost warnings or heat-stress alerts during flowering) should trigger on upper/lower risk bounds (`temp_upper_c`, `temp_lower_c`) rather than point estimates alone.
- **Current Prototype vs. Empirical Quantile Benchmarks**:
  - `ConformalPredictor` (`src/uncertainty/conformal.py`) implements split-conformal calibration using the finite-sample quantile $\lceil (n+1)(1-\alpha)\rceil / n$ on absolute calibration residuals, which guarantees marginal coverage under exchangeability. However, in the current FastAPI backend (`app/backend/main.py` line 58), `conformal_pred.q_val` is initialized to a static configured scalar of `1.15 °C` (`±1.15 °C`, width `2.30 °C`) rather than recalibrating dynamically per station or lead time, and spatial out-of-station predictions violate strict i.i.d. exchangeability.
  - For adaptive heteroscedastic intervals tested under 5-fold Leave-Location-Out spatial CV (`experiments/run_new_architecture_benchmark.py` and `results/metrics/new_architecture_benchmark.csv`), `MultiQuantileResidualModel` (`q10`, `q50`, `q90`, nominal 80% interval) achieves **`87.17%` empirical spatial out-of-fold coverage** with a `0.8477 ± 0.0754 °C` median (`q50`) MAE.

---

## 4. Preserving Multi-Row LGD Polygons and Computing Geodesic Area

### Decision
In `src/data/boundary_pipeline.py`:
1. Do **not** dissolve polygons sharing the same `panchayat_id` (`27,624` LGD codes span `94,276` rows). Assign a deterministic `row_id` while retaining `panchayat_id` for lookup.
2. Recompute polygon areas on the `WGS84` ellipsoid (`pyproj.Geod(ellps="WGS84")`) instead of trusting the source `SHAPE_Area` attribute.

### Rationale
- Dissolving 94,276 multi-part polygons nationwide is computationally expensive and merges non-contiguous administrative enclaves into complex `MultiPolygon` objects whose geometric centroid may fall outside all constituent polygons. Using `geometry.representative_point()` per polygon guarantees that every representative coordinate lies strictly inside its polygon boundary.
- The source `SHAPE_Area` field in `LGD_panchayats.parquet` was calculated in `EPSG:3857` (Web Mercator), inflating areas by `+24.2%` at `26°N` latitude (`reports/boundary_inspection.md`). Geodesic calculation eliminates projection distortion across India's $8^\circ\text{N}$–$37^\circ\text{N}$ latitude span.

---

## 5. Strict Guardrails on Station-to-Panchayat Spatial Joins

### Decision
In `src/data/station_mapping.py` and `src/data/historical_pipeline.py`, a weather station cannot be joined to a Panchayat for verification metrics unless its mapping passes both point-in-polygon containment and a coordinate precision check (`uncertainty_radius_m <= distance_to_boundary_m`).

### Rationale
Publicly archived AWS metadata (such as ISRO MOSDAC's `IMDE1401_141579` station at `25.98°N, 85.67°E`) frequently truncates coordinates to 2 decimal places ($\pm 0.005^\circ \approx 748.4\text{ m}$ uncertainty radius). Because a typical rural Gram Panchayat in the Gangetic plain has an area of only `1.5–4.0 km²`, a `748.4 m` uncertainty radius overlaps up to 15 adjacent Panchayats (`reports/pilot_station_spatial_validation.json`). Automatically claiming Panchayat-level verification accuracy when the ground truth thermometer's coordinate uncertainty exceeds its distance to the Panchayat border (`44.26 m`) would be scientifically invalid. Encoding this check directly into the pipeline (`validation_status = "rejected_coordinate_uncertainty"`) ensures data integrity.

