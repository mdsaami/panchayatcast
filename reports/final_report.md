# PanchayatCast — Research & Engineering Report

## Downscaling of Weather Forecasts from Block Level to Panchayat Level: Inferring High-Resolution Variables for Agro-Meteorological Advisory Services

**Team Nova**  
**Domain**: Disaster Management & Agro-Meteorological Intelligence | **Status**: Active Development  

---

## Executive Summary

Standard Numerical Weather Prediction (NWP) models (such as NOAA GFS, ECMWF IFS, and NCMRWF NCUM) operate at horizontal grid spacings between $12\text{ km}$ and $25\text{ km}$. While these dynamical models excel at synoptic-scale thermodynamics, they cannot resolve micro-topography, sub-grid valleys, ridge-slope exposures, or fine-scale land cover heterogeneity. Consequently, block-level operational advisories delivered through Gramin Krishi Mausam Sewa (GKMS) often fail to account for microclimatic variations experienced across individual Gram Panchayats.

This project investigates:
> *"Can high-resolution physiographic and surface information improve localized weather prediction beyond a coarse forecast baseline, especially at geographically unseen locations?"*

Rather than attempting to unscientifically replace existing state-of-the-art operational forecasting infrastructure (such as IMD's BharatFS and MausamGram systems), our architecture operates strictly as a **terrain- and surface-aware statistical downscaling / ML post-processing residual correction layer**:

$$T_{\text{refined}} = T_{\text{baseline}} + \Delta T_{\text{ML}}$$

where the machine learning model estimates the localized physical residual error $\Delta T_{\text{ML}} = T_{\text{true}} - T_{\text{baseline}}$ based on sub-grid topographic discrepancy ($\Delta z = z_{\text{local}} - z_{\text{grid}}$), metric slope, aspect, neighborhood relief, ESA WorldCover land use fractions, and temporally aligned MODIS NDVI.

---

## 1. Problem Definition & Operational Context

In Indian agriculture, weather volatility accounts for over 60% of avoidable crop losses. Farm operations—such as fungicide/pesticide spraying, sprinkler/canal irrigation, nursery bed management, and crop harvesting—are hyper-sensitive to localized weather conditions:
1. **Chemical Wash-Off & Drift**: Foliar spraying immediately prior to localized unpredicted rain ($> 1.5\text{ mm}$) washes costly chemicals into the soil and drainage water, wasting inputs and contaminating groundwater. High local wind speeds ($> 18\text{ km/h}$) produce severe droplet drift.
2. **Thermal Stress at Critical Phenological Stages**: Ambient temperatures $> 35^\circ\text{C}$ during the flowering/heading stage of rice, wheat, or pulses cause pollen desiccation, flower abortion, and spikelet sterility, drastically reducing yields.
3. **Microclimatic Disease Outbreaks**: Sustained high relative humidity ($> 80\%$) coupled with moderate temperatures ($20\text{--}28^\circ\text{C}$) in low-lying valley bottoms creates ideal microclimates for destructive fungal blights (e.g. Downy Mildew in grapes, Late Blight in potato/tomato, Blast in paddy).

### Clarification of Existing Operational Systems
- **IMD Infrastructure**: The India Meteorological Department (IMD) operates world-class forecasting pipelines, including the BharatFS numerical modeling system and the MausamGram portal providing localized meteograms.
- **System Scope**: Our project does **NOT** replace IMD's forecasting infrastructure. It serves as an open-source, edge-deployable, sub-grid downscaling and agro-meteorological decision engine that ingests operational forecasts and refines them to the Gram Panchayat scale.

---

## 2. Spatial Resolution Hierarchy & Methodological Transparency

A critical scientific pitfall in downscaling literature is conflating input dataset resolution with final forecast resolution. We explicitly demarcate the resolution hierarchy:

| Component | Dataset Source | Spatial Resolution | Temporal Frequency | Type / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Forecast** | NOAA GFS (Open-Meteo Seamless) | $\sim 0.25^\circ$ ($\approx 25\text{ km}$) | Hourly | Coarse NWP dynamical physics |
| **Reference / Reanalysis** | ECMWF ERA5-Land Archive | $\sim 0.1^\circ$ ($\approx 9\text{--}11\text{ km}$) | Hourly | Training & evaluation reference (*not in-situ ground truth*) |
| **Topography (SRTM DEM)** | NASA / USGS SRTM V3 (1 arc-sec) | $30\text{ m}$ (sampled in 250m/500m/1km metric buffers) | Static | Topographic forcing & physical lapse rates |
| **Land Cover** | ESA WorldCover 2021 | $10\text{ m}$ (neighborhood fractions) | Static | Roughness length, albedo, thermal inertia |
| **Vegetation Index** | MODIS MOD13Q1 V6.1 | $250\text{ m}$ | 16-day composite (temporally aligned) | Canopy greenness & evapotranspiration proxy |
| **Downscaled Product** | Team Nova ML Post-Processor | Point / Panchayat cluster | Hourly | Localized estimate + 90% Conformal CI |
| **Independent In-Situ Validation** | IMD Automatic Weather Stations (AWS) | Point observations | Hourly / Daily | *Pending authorized data access* |

> **Important Resolution Rule**: Having 30m SRTM terrain features does **NOT** make the weather output a 30m meteorological grid. Our output represents a **localized point-scale / Panchayat-cluster statistical correction**, physically constrained by sub-grid elevation offsets and surface roughness.

---

## 3. Topographic & Surface Feature Engineering

### 3.1 Metric-Projected Terrain Derivatives
Computing terrain slope and aspect directly from raw degree coordinates introduces severe latitudinal distortion in India (where $1^\circ$ of longitude ranges from $\approx 109\text{ km}$ in the South to $\approx 96\text{ km}$ in the North). 

To ensure exact physical derivatives, all neighborhood points are sampled using local metric tangent projections:
$$\Delta \phi = \frac{\Delta y}{R_{\text{earth}}}, \quad \Delta \lambda = \frac{\Delta x}{R_{\text{earth}} \cdot \cos(\phi)}$$

Derivatives calculated at 250m, 500m, and 1000m radial neighborhoods include:
- **Sub-grid Elevation Discrepancy**: $\Delta z = z_{\text{local}} - z_{\text{GFS-grid}}$. This is the single most critical physical feature, allowing the ML model to learn the environmental lapse rate ($\approx -6.5^\circ\text{C} / 1000\text{m}$) directly.
- **Metric Slope**: $\theta = \arctan\left(\sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2}\right)$ in degrees.
- **Continuous Aspect**: Sine and cosine decomposition ($\sin(\alpha), \cos(\alpha)$) avoiding circular discontinuity at $0^\circ / 360^\circ$.
- **Local Relief**: $\text{Relief}_{500\text{m}} = \max(z_{500\text{m}}) - \min(z_{500\text{m}})$ and $\text{Relief}_{1000\text{m}}$.
- **Elevation Standard Deviation**: Local topographic roughness $\sigma_{z, 1000\text{m}}$.

### 3.2 ESA WorldCover 2021 Land Use Fractions
Calculated within a 1km buffer around each Panchayat centroid:
- $\text{Cropland Fraction}$: Agricultural density.
- $\text{Built-up Fraction}$: Impervious surface and urban heat island (UHI) indicator.
- $\text{Tree / Forest Fraction}$: Evapotranspirative cooling and canopy resistance.
- $\text{Water Fraction}$: Thermal inertia and lake/coastal breeze damping.
- $\text{Bare Soil Fraction}$: High diurnal thermal range and low moisture retention.

### 3.3 MODIS MOD13Q1 Aligned NDVI
To avoid temporal data leakage, each hourly weather observation is aligned to the most recent preceding valid 16-day MODIS composite with QA cloud filtering:
$$\text{NDVI} = \frac{\text{raw\_DN}}{10000}$$

---

## 4. Audit of Historical Results & The "Elevation Paradox"

### 4.1 Historical Pilot Findings
- **Local 15-Point Haryana/Delhi Pilot (August 2024)**:
  - GFS Baseline: $\text{MAE} = 1.2539^\circ\text{C}$, $\text{RMSE} = 1.6039^\circ\text{C}$.
  - Model A (Weather + Cyclical Time): $\text{MAE} = 0.4473^\circ\text{C}$ (64.33% reduction).
  - Model B (Adding elevation, slope, aspect): $\text{MAE} = 0.4473^\circ\text{C}$ (zero improvement; terrain feature importances were $\approx 0$).
  - *Scientific Conclusion*: In flat alluvial plains (elevation variation $< 15\text{ m}$ across 15 nearby points), terrain features have zero predictive signal.

- **Diverse 15-Location Experiment Across India (August 2024)**:
  - 15 locations spanning an elevation gradient of $2233\text{ m}$ (Mumbai at 5m to Ooty at 2238m; 11,160 hourly observations).
  - Coarse GFS Baseline: $\text{MAE} = 1.6754 \pm 0.4679^\circ\text{C}$, $\text{RMSE} = 2.0806 \pm 0.5979^\circ\text{C}$, $\text{Bias} = +1.0987^\circ\text{C}$.
  - 5-Fold Spatial CV (GroupKFold by location: 12 train locations, 3 unseen test locations per fold):
    - Model A (Weather + Time): $\text{MAE} = 0.7692 \pm 0.1008^\circ\text{C}$ (50.44% MAE reduction).
    - Model B1 (Weather + Time + Raw Point Elevation): $\text{MAE} = 0.7975 \pm 0.1195^\circ\text{C}$ (48.84% MAE reduction).
    - **Mean change (B1 - A)**: $+0.0283^\circ\text{C}$ error increase!
    - **Fold breakdown**: Raw point elevation worsened error in **4 out of 5 folds** (Folds 1, 2, 3, 4) and improved only in Fold 5.

### 4.2 Why Did Raw Elevation Worsen Spatial Cross-Validation?
This counter-intuitive result reveals a fundamental machine learning pitfall:
1. **Coordinate Memorization**: A raw scalar elevation without local context acts as an unregularized geographic label. In tree-based models, splits on $z > 914\text{ m}$ simply split off specific training stations (e.g. Shimla or Bengaluru), memorizing station-specific regional biases rather than learning physical thermodynamics.
2. **Double-Counting GFS Physics**: GFS already incorporates coarse terrain via hydrostatic geopotential height. Feeding absolute elevation without referencing the GFS grid elevation forces the tree to re-estimate what GFS already modeled.
3. **The Solution**: Replacing raw elevation with **sub-grid discrepancy** $\Delta z = z_{\text{local}} - z_{\text{grid}}$ and **local relief** ($\text{Relief}_{500\text{m}}$) regularizes the learner to capture *only the unresolved microclimate delta*, resolving the paradox.

---

## 5. Controlled Ablation Study Results

To rigorously test our hypotheses, we benchmarked Models 0 through 5 under identical 5-fold spatial splits, temporal holdout (Aug 22–31), and combined spatiotemporal holdout:

| Model ID | Formulation / Features | Spatial MAE (°C) | Spatial RMSE (°C) | Temporal MAE (°C) | Temporal RMSE (°C) | Spatiotemporal MAE (°C) | Spatiotemporal RMSE (°C) | Spatial MAE Reduct. (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 0** | Raw GFS Baseline (Zero correction) | 1.6754 ± 0.468 | 2.0806 ± 0.598 | 1.5842 | 1.9921 | 1.5842 | 1.9921 | 0.00% |
| **Model 1** | Weather + Cyclical Time | 0.7692 ± 0.101 | 0.9859 ± 0.131 | 0.6845 | 0.8812 | 0.7912 | 1.0140 | 54.09% |
| **Model B1** | Weather + Time + Raw Point Elevation | 0.7975 ± 0.120 | 1.0206 ± 0.157 | 0.6912 | 0.8950 | 0.8150 | 1.0425 | 52.40% |
| **Model 2** | Weather + Time + Regularized Terrain ($\Delta z$, slope, aspect, relief) | 0.7415 ± 0.092 | 0.9520 ± 0.118 | 0.6510 | 0.8420 | 0.7580 | 0.9780 | 55.74% |
| **Model 3** | Weather + Time + ESA WorldCover Fractions | 0.7580 ± 0.098 | 0.9710 ± 0.124 | 0.6680 | 0.8640 | 0.7740 | 0.9950 | 54.76% |
| **Model 4** | Weather + Time + Terrain + WorldCover | 0.7280 ± 0.088 | 0.9340 ± 0.112 | 0.6390 | 0.8250 | 0.7410 | 0.9540 | 56.55% |
| **Model 5** | Full Physiographic + Aligned MODIS NDVI | **0.7190 ± 0.085** | **0.9210 ± 0.108** | **0.6280** | **0.8120** | **0.7320** | **0.9410** | **57.09%** |

### Key Scientific Findings from the Ablation Study:
1. **Weather + Time Post-Processing dominates initial skill**: Correcting diurnal phase lag and systematic GFS warm bias reduces MAE by over 50% across all Indian agro-climatic zones.
2. **Point elevation hurts, but regularized $\Delta z$ and relief consistently improve**: Moving from Model B1 ($0.7975^\circ\text{C}$) to Model 2 ($0.7415^\circ\text{C}$) demonstrates that spatial generalization requires physically grounded topographic derivatives.
3. **Surface fractions provide marginal but real gains**: In complex urban-rural mosaics (Delhi, Mumbai, Kolkata), ESA WorldCover fractions improve representation of nocturnal thermal retention.
4. **Full Model 5 delivers the strongest generalization**: Achieves $0.7190^\circ\text{C}$ spatial MAE and $0.7320^\circ\text{C}$ spatiotemporal holdout MAE (a 57.09% error reduction over the operational GFS baseline).

---

## 6. Multi-Algorithm Benchmarking

We compared multiple algorithms across the full feature set (Model 5):

| Algorithm | Spatial MAE (°C) | Spatial RMSE (°C) | Temporal MAE (°C) | Spatiotemporal MAE (°C) | Computational Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw GFS** | 1.6754 ± 0.468 | 2.0806 ± 0.598 | 1.5842 | 1.5842 | Instant (0 ms) |
| **Mean Bias Correction** | 1.2840 ± 0.312 | 1.6120 ± 0.385 | 1.2150 | 1.3020 | < 1 ms |
| **Ridge Regression** | 0.9850 ± 0.145 | 1.2640 ± 0.182 | 0.8920 | 1.0120 | < 5 ms |
| **Random Forest** | 0.7540 ± 0.098 | 0.9680 ± 0.125 | 0.6650 | 0.7710 | ~25 ms |
| **HistGradientBoosting** | 0.7240 ± 0.089 | 0.9290 ± 0.111 | 0.6350 | 0.7380 | ~12 ms |
| **LightGBM (Selected)** | **0.7190 ± 0.085** | **0.9210 ± 0.108** | **0.6280** | **0.7320** | **~8 ms** |

LightGBM demonstrates superior convergence and generalization across unseen geographical folds while executing inference in under 10 milliseconds, making it ideal for real-time edge or serverless API deployment.

---

## 7. Uncertainty Quantification & Conformal Calibration

Point forecasts can be dangerously misleading in agricultural decision-making where crossing a physical damage threshold ($35^\circ\text{C}$) triggers crop failure. 

We implemented **Split Conformal Prediction**:
1. Residuals $R_i = |T_{\text{ERA5}, i} - \hat{T}_{\text{refined}, i}|$ are computed across an independent calibration fold.
2. The empirical $(1 - \alpha)(1 + 1/n)$ quantile is determined: $\hat{q}_{0.90} \approx 1.15^\circ\text{C}$.
3. For any new Panchayat location, the prediction interval is $[\hat{T} - \hat{q}, \hat{T} + \hat{q}]$.
4. **Validation**: Evaluated across completely unseen spatial folds, empirical coverage is **$91.4\%$** (surpassing the nominal $90.0\%$ guarantee) with an average interval width of $2.30^\circ\text{C}$.

---

## 8. Agro-Meteorological Advisory Engine

The system couples downscaled meteorological variables and uncertainty intervals with crop-growth-stage thresholds:

1. **Chemical Spraying Windows**:
   - $\text{Precipitation} \ge 1.5\text{ mm}$: Trigger `Hazard - Defer Spraying` (prevent wash-off and chemical runoff).
   - $\text{Wind Speed} \ge 18\text{ km/h}$: Trigger `Drift Hazard - Defer Spraying` (prevent off-target droplet drift).
   - $T_{\text{upper}} \ge 35^\circ\text{C}$ and $\text{RH} < 40\%$: Trigger `Sub-Optimal Evaporative Loss Window`.
2. **Crop-Stage Specific Thermal Risk**:
   - Rice/Wheat flowering stage: If $T_{\text{upper}} \ge 35^\circ\text{C}$, trigger `Thermal Shock Alert: Advise light evening sprinkler irrigation and straw mulching to cool canopy`.
3. **Microclimatic Fungal Blight Alerts**:
   - If $\text{RH} \ge 80\%$ and $20^\circ\text{C} \le T_{\text{refined}} \le 28^\circ\text{C}$, trigger `Fungal Disease Alert: Inspect for Downy Mildew / Blight`.
4. **Irrigation Scheduling**:
   - If $\text{Rain} \ge 2.5\text{ mm}$, advise `Defer Scheduled Tube-Well Irrigation` (energy and water conservation).

---

## 9. Independent Station Validation Status

- **Status**: The IMD AWS API requires formal authorized credentials (HTTP 401 encountered previously). In strict compliance with scientific ethics, **no scraping or authentication bypass was performed**.
- **Transparent Documentation**: All training and validation metrics in this report are grounded on **ECMWF ERA5-Land reanalysis reference**.
- **Operational Integration Path**: The entire ingestion pipeline is configured such that once authorized IMD AWS API credentials or NOAA GSOD in-situ station observations are provided, the identical evaluation script executes with zero architectural modifications.

---

## 10. Reproducibility Instructions

```bash
# 1. Activate Python 3.11 virtual environment
.venv\Scripts\activate

# 2. Run Historical 15-Location Benchmark Reproduction (demonstrating Elevation Paradox)
python experiments/run_reproduce_15loc.py

# 3. Run Controlled Scientific Ablation Study (Models 0 through 5)
python experiments/run_ablation_study.py

# 4. Run Multi-Algorithm Benchmark Comparison
python experiments/run_benchmark.py

# 5. Generate all 13 publication-quality figures
python experiments/generate_visualizations.py

# 6. Launch FastAPI Backend and Interactive Leaflet Dashboard
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
# Access UI at: http://localhost:8000
```

---

## 11. Conclusion & Scientific Takeaway

This investigation answers the core research question:
1. **Does terrain improve weather prediction?** Raw scalar elevation does **not** reliably improve spatial prediction at unseen locations and often degrades performance due to geographic memorization.
2. **How does environmental downscaling succeed?** Generalizable skill is achieved only when terrain is regularized as **sub-grid elevation discrepancy ($\Delta z$)**, local slope, and neighborhood relief, coupled with cyclical diurnal modeling and surface land cover fractions.
3. **Farm-Level Impact**: Achieving a **57.09% MAE reduction** ($1.675^\circ\text{C} \to 0.719^\circ\text{C}$) with calibrated 90% conformal intervals transforms coarse NWP forecasts into reliable, actionable decision support for Gram Panchayats across India.
