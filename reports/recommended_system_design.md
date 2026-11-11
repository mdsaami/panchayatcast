# Recommended System Design: PanchayatCast Weather Downscaling

**Team Nova**
**If you were building PanchayatCast from scratch today after studying the best global research, this is how you would structure the architecture.**

---

## 1. Design Philosophy

### Core Principles (Derived from Literature)

1. **Residual correction, not absolute prediction**: `T_panchayat = T_GFS + ΔT_ML`
2. **Prove spatial generalization first**: LLO CV before anything else
3. **Start simple, add complexity only when justified**: LightGBM → U-Net → Diffusion
4. **Dense proxy before sparse fine-tuning**: ERA5-Land → Station observations
5. **Uncertainty is not optional**: Advisory services need confidence intervals

### What the Literature Tells Us

| Question | Answer | Evidence |
|----------|--------|----------|
| Should we use residual correction? | **Yes** — unanimous consensus | CorrDiff, FuXi, EDSR, Rasp & Lerch, our Phase 2A |
| Which ML model for sparse stations? | **LightGBM/XGBoost** | Cao et al., Ahmad et al., our benchmark |
| Which DL model for gridded downscaling? | **U-Net + CBAM** → **CorrDiff-Lite** | NOAA BC-Unet, CorrDiff, Sha et al. |
| How to handle terrain? | **Relative features** (not raw elevation) | Meyer et al., our Elevation Paradox |
| How to validate? | **Leave-Location-Out CV** | Meyer et al., Ploton et al. |
| How to handle data scarcity? | **Transfer learning** from ERA5-Land | Transfer learning review (+15-25% MAE) |
| How to quantify uncertainty? | **Quantile regression + Conformal** | Cannon (2011), our implementation |

---

## 2. Three-Tier Architecture

We recommend a **progressive three-tier system** that can be built incrementally:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 3: GENERATIVE REFINEMENT                │
│              (CorrDiff-Lite Diffusion for uncertainty)          │
│         Input: Tier 2 grid + noise → Ensemble of 1km grids     │
│                    Status: Future / Stretch Goal                │
├─────────────────────────────────────────────────────────────────┤
│                    TIER 2: GRIDDED U-NET                        │
│            (BC-Unet with CBAM for spatial correction)           │
│    Input: GFS grid + DEM + LandCover → Corrected 1km grid      │
│                 Status: Next Phase (needs GPU)                  │
├─────────────────────────────────────────────────────────────────┤
│                    TIER 1: POINT RESIDUAL ML                    │
│          (LightGBM for station-level bias correction)           │
│        Input: GFS features + terrain → ΔT per station          │
│                 Status: OPERATIONAL (MAE 0.79°C)                │
└─────────────────────────────────────────────────────────────────┘
```

Each tier builds on the previous one. Tier 1 is already operational. Tiers 2 and 3 are extensions.

---

## 3. Tier 1: Point Residual ML (Current System)

### Architecture
```
Input Features (9):
├── GFS 2m Temperature
├── GFS Dewpoint (2m)
├── GFS Relative Humidity
├── GFS Wind Speed (10m)
├── GFS Wind Direction (10m)
├── GFS Surface Pressure
├── GFS Cloud Cover
├── Hour of Day (cyclical: sin/cos)
└── Day of Year (cyclical: sin/cos)

Model: LightGBM (gradient boosted decision trees)
├── num_leaves: 31
├── learning_rate: 0.05
├── n_estimators: 500
├── reg_alpha: 0.1
├── reg_lambda: 1.0
└── min_child_samples: 20

Target: ΔT = T_ERA5Land - T_GFS_interpolated

Output: T_corrected = T_GFS + ΔT_predicted
```

### Current Performance
| Metric | GFS Baseline | Tier 1 (LightGBM) | Improvement |
|--------|-------------|-------------------|-------------|
| MAE | 1.66°C | 0.79°C | -52.4% |
| RMSE | 2.19°C | ~1.1°C | -49.8% |
| Bias | +1.11°C | ~0.0°C | Eliminated |

### Improvements to Implement

#### 3.1 Fix Terrain Features
```python
# WRONG: Raw elevation (causes station memorization)
features['elevation'] = station_elevation  # ❌

# RIGHT: Relative elevation (transferable)
features['delta_z'] = station_elevation - gfs_grid_mean_elevation  # ✅
features['lapse_correction'] = -6.5 * (station_elevation - gfs_elevation) / 1000  # ✅
features['slope'] = derived_from_dem  # ✅
features['aspect_sin'] = np.sin(np.radians(aspect))  # ✅
features['aspect_cos'] = np.cos(np.radians(aspect))  # ✅
features['tpi'] = station_elev - mean_elev_in_radius  # ✅
```

#### 3.2 Forward Feature Selection with LLO CV
```python
# Meyer et al. (2018) approach
from sklearn.model_selection import GroupKFold

def forward_feature_selection(X, y, groups, max_features=15):
    """Add features one-by-one, keeping only those that improve LLO CV."""
    selected = []
    best_mae = float('inf')
    
    for _ in range(max_features):
        best_feature = None
        for f in remaining_features:
            trial = selected + [f]
            mae = llo_cv_score(X[trial], y, groups)
            if mae < best_mae:
                best_mae = mae
                best_feature = f
        if best_feature:
            selected.append(best_feature)
        else:
            break
    return selected
```

#### 3.3 Expand Station Network
```
Current:  15 stations (August 2024)
Target:   100+ stations across India
Duration: 12+ months for seasonal coverage
Sources:  Open-Meteo ERA5-Land (proxy), IMD AWS (if accessible)
```

#### 3.4 Add Multi-Quantile Output
```python
# Cannon (2011) approach
from lightgbm import LGBMRegressor

quantile_models = {}
for alpha in [0.10, 0.50, 0.90]:
    quantile_models[alpha] = LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        **base_params
    )
```

---

## 4. Tier 2: Gridded U-Net with CBAM

### When to Implement
When Tier 1 is mature AND:
- GPU access is available (Colab T4 minimum)
- ERA5-Land gridded data is fetched as dense training target
- 100+ stations are available for validation

### Architecture: BC-Unet (Based on NOAA, 2024)

```
Input Channels (C=8):
├── GFS 2m Temperature         [1 channel, 25km grid]
├── GFS Dewpoint                [1 channel, 25km grid]
├── GFS Wind Speed              [1 channel, 25km grid]
├── GFS Surface Pressure        [1 channel, 25km grid]
├── GFS Cloud Cover             [1 channel, 25km grid]
├── SRTM DEM (elevation)        [1 channel, 1km grid]
├── DEM Slope                   [1 channel, 1km grid]
└── ESA Land Cover (one-hot)    [1 channel, 1km grid]

Architecture:
Encoder                        Decoder
├── Conv2d(8,64) + CBAM        ├── UpConv(512,256) + Skip
├── MaxPool + Conv2d(64,128)   ├── UpConv(256,128) + Skip
├── MaxPool + Conv2d(128,256)  ├── UpConv(128,64) + Skip
├── MaxPool + Conv2d(256,512)  └── Conv2d(64,1) → ΔT grid
└── Bottleneck

CBAM at each encoder level:
├── Channel Attention: avg_pool + max_pool → MLP → σ → weight
└── Spatial Attention: avg_pool + max_pool → Conv7x7 → σ → weight

Loss: Huber Loss (δ=1.0)
Optimizer: AdamW, lr=3e-4, weight_decay=1e-5
Scheduler: CosineAnnealingLR

Output: ΔT grid at 1km resolution
Final: T_corrected_grid = T_GFS_upsampled + ΔT_grid
```

### Training Strategy
```
Phase 1: Pre-train on ERA5-Land (dense gridded target)
├── Input: ERA5 at 0.25° → bilinear upsample to 0.1°
├── Target: ERA5-Land at 0.1° (9 km)
├── Objective: Learn general downscaling patterns
├── Duration: ~6 hours on T4 GPU
└── Loss: Huber

Phase 2: Fine-tune on station observations
├── Freeze encoder layers 1-3
├── Unfreeze decoder + final layers
├── Loss: Station-masked MAE (only penalize at station locations)
├── Duration: ~2 hours on T4 GPU
└── Validation: LLO CV on station network
```

### CBAM Implementation
```python
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).flatten(1))
        max_out = self.fc(self.max_pool(x).flatten(1))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x
```

---

## 5. Tier 3: CorrDiff-Lite (Stretch Goal)

### Architecture
Following Mardani et al. (2023) with simplifications:

```
Stage 1: Deterministic Mean (Tier 2 U-Net output)
├── Predicts E[ΔT | GFS, terrain]
└── Already trained in Tier 2

Stage 2: Stochastic Residual (Small Diffusion Model)
├── Architecture: Small U-Net (4 levels, 64 base channels)
├── Input: Tier 2 mean prediction + noise (t steps)
├── Conditioning: GFS fields + DEM + Tier 2 output
├── Loss: Denoising Score Matching
├── Sampling: DDIM with 10-50 steps
└── Output: Ensemble of ΔT corrections

Final: T_panchayat = T_GFS + ΔT_mean + ΔT_stochastic
```

### Simplified Implementation (ClimateDiffuse-based)
Use [robbiewatt1/ClimateDiffuse](https://github.com/robbiewatt1/ClimateDiffuse) as starting point:
- Clean PyTorch implementation
- EDM (Elucidating the Design Space of Diffusion Models) framework
- Much simpler than NVIDIA Modulus ecosystem

### Compute Requirements
- Training: ~12 hours on V100 or A100
- Inference: ~1 second per ensemble member (with DDIM 10-step)
- 10-member ensemble: ~10 seconds per forecast grid

---

## 6. Data Pipeline

### Current Pipeline
```
Open-Meteo APIs
├── GFS Historical Forecast API → data/raw/gfs/
├── ERA5-Land Archive API → data/raw/era5land/
└── Elevation API → data/raw/terrain/

Preprocessing
├── Temporal alignment (GFS → ERA5-Land timestamps)
├── Spatial interpolation (GFS 0.25° → station points)
├── Feature engineering (cyclical time, derived terrain)
└── Residual computation (ΔT = ERA5Land - GFS_interp)

Datasets
├── data/processed/merged_15stations_aug2024.csv
└── data/processed/merged_Nstations_fullperiod.csv
```

### Enhanced Pipeline (Recommended)
```
Phase 1: Point Data (Tier 1)
├── Scale to 100+ stations
├── Fetch 12+ months of data (seasonal coverage)
├── Add derived terrain features (Δz, slope, aspect, TPI)
├── Forward feature selection with LLO CV
└── Train LightGBM with quantile outputs

Phase 2: Gridded Data (Tier 2)
├── Fetch ERA5-Land 0.1° grids for India domain
│   ├── Lat: 6°N to 38°N
│   ├── Lon: 68°E to 98°E
│   └── Variables: 2m temp, dewpoint, wind
├── Fetch GFS 0.25° grids (matching domain/time)
├── Fetch SRTM DEM at 90m → resample to 0.01° (~1km)
├── Fetch ESA WorldCover at 10m → resample to 0.01°
├── Create training patches (e.g., 64x64 at 0.01°)
└── Train U-Net + CBAM
```

---

## 7. Validation Strategy

### Three-Level Validation Framework

```
Level 1: Cross-Validation (Development)
├── 5-fold Leave-Location-Out (GroupKFold by station cluster)
├── Forward feature selection within LLO CV
├── Report: MAE ± std, RMSE ± std, Bias ± std per fold
└── Compare: Each model variant must beat Tier 1 baseline

Level 2: Temporal Holdout (Pre-deployment)
├── Train on months 1-10, test on months 11-12
├── Report: Monthly MAE by climate zone
└── Test: Seasonal generalization (monsoon → winter)

Level 3: Spatiotemporal Holdout (Deployment Simulation)
├── Hold out 20% of stations AND last 2 months
├── This simulates: deploying to NEW panchayats in the FUTURE
├── Report: MAE on unseen stations × unseen times
└── This is the primary generalization metric for PanchayatCast
```

### Benchmark Comparison Matrix
```
| System              | MAE (°C) | Spatial CV? | Stations | Notes            |
|---------------------|----------|-------------|----------|------------------|
| Raw GFS             | 1.66     | N/A         | 15       | Our baseline     |
| Mean Bias Correction| ~1.2     | Yes         | 15       | Simple but robust|
| Tier 1 (LightGBM)  | 0.79     | Yes (5-fold)| 15       | Current best     |
| Tier 1 (improved)   | TBD      | Yes         | 100+     | Fixed terrain    |
| Tier 2 (U-Net+CBAM) | TBD      | Yes         | 100+     | Needs GPU        |
| Tier 3 (CorrDiff)   | TBD      | Yes         | 100+     | Stretch goal     |
| BharatFS 6km        | ???      | ???         | N/A      | IMD operational  |
```

---

## 8. Feature Engineering Roadmap

### Current Features (9)
```
Weather: gfs_temp, gfs_dewpoint, gfs_rh, gfs_wind_speed,
         gfs_wind_direction, gfs_pressure, gfs_cloud_cover
Time:    hour_sin, hour_cos, day_sin, day_cos
```

### Recommended Features (15-20)
```
Weather (7):
├── gfs_temp_2m
├── gfs_dewpoint_2m
├── gfs_relative_humidity
├── gfs_wind_speed_10m
├── gfs_wind_direction_10m (sin/cos)
├── gfs_surface_pressure
└── gfs_cloud_cover

Time (4):
├── hour_sin, hour_cos
└── day_sin, day_cos

Terrain - CORRECTED (6):
├── delta_z (station_elev - gfs_cell_mean_elev)  # NOT raw elevation
├── lapse_correction (-6.5 * delta_z / 1000)
├── slope (from DEM)
├── aspect_sin, aspect_cos (from DEM)
└── tpi (topographic position index: relative to 5km neighborhood)

Land Surface (2):
├── distance_to_coast_km
└── dominant_landcover_class (categorical)

Derived (2):
├── gfs_temp_anomaly (GFS temp minus daily climatology)
└── forecast_lead_hour
```

### Feature Selection Protocol
Use Meyer et al. (2018) forward selection with LLO CV:
1. Start with weather-only features (7)
2. Add time features (improve? keep if yes)
3. Add terrain features one-by-one (keep only if LLO MAE improves)
4. Add land surface features one-by-one
5. Final model uses only features that survived LLO selection

---

## 9. Experiment Roadmap

### Phase 1: Strengthen Tier 1 (Weeks 1-2)
```
Experiment 1.1: Terrain Feature Fix
├── Replace raw elevation with Δz, lapse correction, slope, aspect, TPI
├── Measure: LLO MAE change
└── Expected: Break Elevation Paradox

Experiment 1.2: Forward Feature Selection
├── Apply Meyer et al. (2018) protocol
├── Identify optimal feature subset via LLO CV
└── Expected: Reduced variance across folds

Experiment 1.3: Station Network Expansion
├── Scale to 50-100 stations
├── Add 6+ months of data (seasonal diversity)
├── Re-run ablation study
└── Expected: More robust spatial generalization

Experiment 1.4: Multi-Quantile LightGBM
├── Train P10, P50, P90 quantile models
├── Evaluate calibration (coverage at nominal levels)
├── Compare with conformal prediction
└── Expected: Better-calibrated uncertainty

Experiment 1.5: Ensemble Stacking
├── Combine LightGBM, HistGBDT, Ridge, RF
├── Meta-learner: Ridge regression
├── LLO CV evaluation
└── Expected: 5-10% MAE improvement
```

### Phase 2: Build Tier 2 (Weeks 3-4)
```
Experiment 2.1: ERA5-Land Grid Fetching
├── Fetch gridded ERA5-Land for India domain
├── Create training patches (64x64 at ~0.01°)
└── Store as zarr/netCDF

Experiment 2.2: Baseline U-Net
├── Implement simple U-Net (no attention)
├── Train on ERA5-Land grids
├── Evaluate against Tier 1 at station locations
└── Expected: Competitive with LightGBM

Experiment 2.3: BC-Unet (U-Net + CBAM)
├── Add CBAM attention modules
├── Compare with baseline U-Net
├── Analyze attention maps (where does model focus?)
└── Expected: Improvement in complex terrain

Experiment 2.4: Transfer Learning
├── Pre-train on global ERA5 → ERA5-Land
├── Fine-tune on Indian domain
├── Compare with training from scratch
└── Expected: 15-25% MAE improvement

Experiment 2.5: Station Fine-Tuning
├── Fine-tune Tier 2 U-Net at station locations only
├── Station-masked loss function
├── LLO CV evaluation
└── Expected: Best overall performance
```

### Phase 3: Prototype Tier 3 (Weeks 5-6, if GPU available)
```
Experiment 3.1: ClimateDiffuse Prototype
├── Adapt ClimateDiffuse codebase for our data
├── Train small diffusion model on residuals
├── Evaluate ensemble spread and calibration
└── Expected: Probabilistic uncertainty improvement

Experiment 3.2: Consistency Model
├── Train consistency model for single-step inference
├── Compare inference speed vs quality
└── Expected: Real-time capable probabilistic output
```

---

## 10. Technology Stack

### Current (Tier 1)
```
Python 3.11
├── numpy, pandas, scipy
├── scikit-learn
├── lightgbm
├── matplotlib, seaborn
├── fastapi, uvicorn
└── pytest
```

### Extended (Tier 2+)
```
Python 3.11
├── [Current stack]
├── torch >= 2.0 (PyTorch)
├── torchvision
├── xarray + netCDF4 (gridded data)
├── rioxarray (geospatial rasters)
├── diffusers (for Tier 3 diffusion)
└── wandb or tensorboard (experiment tracking)
```

### Infrastructure
```
Development: Local CPU (Tier 1)
Training:    Google Colab T4 (Tier 2) / A100 (Tier 3)
Deployment:  FastAPI + LightGBM (CPU server)
             Optional: ONNX Runtime for U-Net inference
```

---

## 11. Deployment Architecture

```
┌───────────────────────────────────────────────┐
│                  Frontend                      │
│    Leaflet Map + D3.js Charts + Dashboard     │
├───────────────────────────────────────────────┤
│               FastAPI Backend                  │
│  /api/downscale  /api/transect  /api/advisory │
├──────────┬───────────┬────────────────────────┤
│ Tier 1   │ Tier 2    │ Advisory Engine        │
│ LightGBM │ U-Net     │ Crop-stage rules       │
│ (CPU)    │ (ONNX/CPU)│ Temperature thresholds  │
├──────────┴───────────┴────────────────────────┤
│              Data Layer                        │
│  GFS API → Preprocessor → Feature Pipeline    │
│  DEM Cache  │  Land Cover Cache  │  Station DB│
└───────────────────────────────────────────────┘
```

### Inference Flow
```
1. User requests forecast for panchayat (lat, lon)
2. Fetch latest GFS forecast (0.25° grid cell)
3. Extract terrain features (pre-cached DEM derivatives)
4. Extract land surface features (pre-cached)
5. Tier 1: LightGBM predicts ΔT + uncertainty (P10, P50, P90)
6. Tier 2 (if available): U-Net predicts ΔT grid for surrounding area
7. Advisory engine: Apply crop-stage rules to corrected temperature
8. Return: Corrected forecast + uncertainty + advisory
```

---

## 12. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Station data too sparse for DL | High | Tier 2 fails | Use ERA5-Land as dense proxy; transfer learning |
| Elevation Paradox persists | Medium | Spatial CV degrades | Relative features; forward selection; lapse rate |
| No GPU available for Tier 2 | Medium | Limited to Tier 1 | Colab free tier; kaggle notebooks; focus on Tier 1 |
| IMD data inaccessible | High | No real station validation | Use ERA5-Land as proxy; Open-Meteo stations |
| GFS resolution too coarse | Low | Poor base forecast | Already 0.25°; ERA5 alternative; BharatFS |
| Seasonal non-stationarity | Medium | Model degrades in new season | Rolling retraining; seasonal features |
| Overfitting to training region | Medium | Fails on new states/zones | Diverse station network; spatial CV |

---

## 13. Success Criteria

### Minimum Viable Product (Core Prototype Milestones)
- [ ] Tier 1 operational with 50+ stations
- [ ] MAE < 1.0°C on spatial CV across 5+ climate zones
- [ ] Uncertainty intervals with >80% coverage
- [ ] Interactive dashboard showing corrected vs GFS
- [ ] Advisory output for 3+ crop types

### Stretch Goals
- [ ] Tier 2 U-Net producing 1km gridded correction maps
- [ ] Transfer learning from ERA5-Land improves MAE by >15%
- [ ] CBAM attention maps showing physically meaningful focus
- [ ] Comparison with BharatFS/MausamGram performance
- [ ] Real-time inference <100ms per panchayat

### Research Contributions
- [ ] Quantification of Elevation Paradox and solution
- [ ] LLO CV benchmark for Indian weather downscaling
- [ ] Comparison of ML vs DL for sparse station networks
- [ ] Open-source implementation with documentation

---

## 14. Final Answer

**If you were building PanchayatCast from scratch today:**

1. **Keep LightGBM residual correction** as Tier 1. It is competitive with literature results and works without GPU.

2. **Fix terrain features immediately**: Replace raw elevation with relative elevation (Δz), lapse rate correction, and topographic position index. Use forward feature selection with LLO CV.

3. **Scale the station network** to 100+ diverse locations with 12+ months of data.

4. **Add ERA5-Land** as a dense proxy training target to enable gridded deep learning models.

5. **Implement BC-Unet (U-Net + CBAM)** as Tier 2 — it is the exact architecture NOAA uses for GFS 2m temperature bias correction.

6. **Use transfer learning**: Pre-train on global ERA5→ERA5-Land, fine-tune on Indian stations.

7. **Adopt MetNet-3's densification masking** to train the U-Net to handle sparse station observations.

8. **For technical evaluation**, emphasize:
   - Residual correction is the **global state-of-the-art** (CorrDiff, FuXi, NOAA BC-Unet)
   - Our LLO spatial validation proves **real-world generalization**
   - Our uncertainty quantification enables **actionable crop advisories**
   - The three-tier architecture provides a **clear scaling roadmap**

9. **Position against IMD**: We are an ML **enhancement layer** on top of BharatFS/GFS, not a replacement.

10. **The single most impactful next experiment**: Fix terrain features + expand stations + forward feature selection. This requires zero new infrastructure and could significantly improve spatial generalization.

---

*This design document is based on systematic analysis of 42 papers, 15 open-source repositories, and 7 operational weather systems worldwide.*
*Full literature database: `results/literature_database.csv`*
*Comprehensive review: `reports/global_weather_downscaling_review.md`*
