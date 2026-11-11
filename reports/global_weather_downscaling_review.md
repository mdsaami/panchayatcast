# Global Weather Downscaling Review: ML/DL Approaches for PanchayatCast

**Team Nova**
**Research Objective**: Downscaling of weather forecasts from Block level to Panchayat level
**Date**: September 2026
**Compiled from**: 42 papers, 15 repositories, 7 operational systems

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Formulation](#2-problem-formulation)
3. [Taxonomy of Approaches](#3-taxonomy-of-approaches)
4. [Statistical Downscaling & MOS](#4-statistical-downscaling--mos)
5. [Tree-Based ML Methods](#5-tree-based-ml-methods)
6. [CNN-Based Super-Resolution](#6-cnn-based-super-resolution)
7. [U-Net Architectures](#7-u-net-architectures)
8. [Vision Transformers & Attention](#8-vision-transformers--attention)
9. [Diffusion Models](#9-diffusion-models)
10. [GANs for Weather SR](#10-gans-for-weather-sr)
11. [Graph Neural Networks](#11-graph-neural-networks)
12. [Foundation Models](#12-foundation-models)
13. [Residual Learning Paradigm](#13-residual-learning-paradigm)
14. [Loss Functions Comparison](#14-loss-functions-comparison)
15. [Training Strategies](#15-training-strategies)
16. [Uncertainty Quantification](#16-uncertainty-quantification)
17. [Spatial Validation Methods](#17-spatial-validation-methods)
18. [Terrain-Aware Features](#18-terrain-aware-features)
19. [Operational Systems Worldwide](#19-operational-systems-worldwide)
20. [India-Specific Systems & Context](#20-india-specific-systems--context)
21. [Open-Source Codebases](#21-open-source-codebases)
22. [Data Sources & Availability](#22-data-sources--availability)
23. [Evaluation Metrics](#23-evaluation-metrics)
24. [Computational Requirements](#24-computational-requirements)
25. [Common Failure Modes](#25-common-failure-modes)
26. [Architecture Comparison Matrix](#26-architecture-comparison-matrix)
27. [Key Insights for PanchayatCast](#27-key-insights-for-panchayatcast)
28. [What We Should Do Next](#28-what-we-should-do-next)

---

## 1. Executive Summary

This review systematically surveys the global landscape of machine learning and deep learning approaches to weather forecast downscaling, with specific focus on techniques applicable to **residual correction of GFS 2m temperature from block level (~25 km) to panchayat level (~1 km)** over India.

### Key Findings

1. **Residual correction is the dominant paradigm**: Rather than predicting absolute high-resolution temperature, the most successful systems learn `ΔT = T_observed - T_GFS_interpolated`, then add this correction back. This is validated by CorrDiff (NVIDIA), FuXi, EDSR/VDSR, and our own Phase 2A experiments.

2. **Tree-based ML (LightGBM/XGBoost) outperforms deep learning for sparse point observations**: When downscaling to individual station locations without gridded high-resolution targets, tree-based models achieve competitive or superior MAE/RMSE with orders of magnitude less compute. This is confirmed by Rasp & Lerch (2018), Cao et al. (2024), Ahmad et al. (2024), and our own benchmark.

3. **Deep learning excels when gridded high-resolution targets exist**: U-Nets, EDSR, and diffusion models shine when the training target is a dense high-resolution grid (e.g., ERA5-Land, HRRR analysis, WRF output), not sparse station observations.

4. **Spatial cross-validation is non-negotiable**: Meyer et al. (2018) and Ploton et al. (2020) prove that random train/test splits overestimate spatial prediction skill by 30-50%. Leave-Location-Out (LLO) validation is mandatory for any model claiming to generalize to unmonitored panchayats.

5. **Terrain as input is essential but representation matters**: All successful downscaling systems incorporate high-resolution DEM data. However, raw scalar elevation can cause overfitting (confirmed by our Elevation Paradox). Successful approaches use elevation as a spatial input channel (in CNNs/U-Nets) or as derived features (lapse rate, slope, aspect, relief).

6. **Transfer learning is critical for data-scarce regions**: With only 15-100 Indian station observations, pre-training on dense global datasets (ERA5-Land) and fine-tuning on local stations improves MAE by 15-25%.

7. **IMD BharatFS operates at ~6 km natively**: India's operational system already provides 6 km forecasts. Our GFS-based system must bridge a ~25 km → 1 km gap, which is a 25× resolution increase.

---

## 2. Problem Formulation

### Our Specific Problem
```
Input:  GFS forecast at 0.25° (~25 km) resolution
        + Static terrain features (DEM, slope, aspect, land cover)
        + Point station observations for training
Output: Corrected 2m temperature at panchayat level (~1 km)
```

### Mathematical Formulation
```
T_panchayat = T_GFS_interpolated + ΔT_ML(features)
```

Where `ΔT_ML` is the residual correction predicted by the ML model.

### The Two Sub-Problems

| Aspect | Point Downscaling | Gridded Downscaling |
|--------|-------------------|---------------------|
| **Target** | Individual station/panchayat locations | Dense 2D grid |
| **Best Methods** | Tree-based ML, MLP, Kalman filters | CNN, U-Net, Diffusion |
| **Training Data** | Point observations | Gridded reanalysis/RCM |
| **Our Situation** | ✅ Current (15-100 stations) | Future aspiration |

**Critical insight**: Our project currently operates in the **point downscaling** regime (sparse station observations), not the gridded super-resolution regime. This fundamentally constrains which architectures are appropriate.

---

## 3. Taxonomy of Approaches

```
Weather Downscaling
├── Statistical/ML (Tabular)
│   ├── Model Output Statistics (MOS) — linear regression
│   ├── Analog Methods — nearest-neighbor in predictor space
│   ├── Random Forest — ensemble of decision trees
│   ├── Gradient Boosting — LightGBM, XGBoost, HistGBDT
│   ├── Neural Networks — MLP for point residuals
│   └── Hybrid — ML + Kalman Filter
├── Deep Learning (Gridded)
│   ├── CNN Super-Resolution — SRCNN, DeepSD, EDSR, VDSR
│   ├── U-Net — Encoder-decoder with skip connections
│   ├── Vision Transformers — ClimaX, PrecipFormer, 1EMD
│   ├── Diffusion Models — CorrDiff, SwinRDM, ClimateDiffuse
│   ├── GANs — ESRGAN, Stochastic SRGAN
│   ├── Fourier Neural Operators — Resolution-invariant
│   └── Graph Neural Networks — Irregular meshes
├── Generative / Probabilistic
│   ├── Probabilistic U-Net — Variational latent space
│   ├── Quantile Regression — QRNN, Q-SRDRN
│   ├── EMOS / NGR — Ensemble MOS
│   └── Conformal Prediction — Distribution-free coverage
└── Hybrid Physics-ML
    ├── Physics-Constrained DL — Hard constraint layers
    ├── Adaptive Residual Correction — FuXi-style cascade
    └── Data Assimilation + ML — 4DVar with neural surrogates
```

---

## 4. Statistical Downscaling & MOS

### Model Output Statistics (MOS)
Traditional MOS uses multiple linear regression to correct NWP forecasts at specific stations. **ML-MOS** replaces linear functions with machine learning algorithms.

### Perfect Prognosis (PP)
Establishes empirical relationships between large-scale predictors (from reanalysis) and local observations, then applies these relationships to NWP forecasts. Assumes NWP captures large-scale patterns correctly.

### Analog Methods
For each forecast, find the historically most similar large-scale pattern and use the corresponding observed local weather. Simple but limited by the historical record.

### Key Finding
> **MOS-style point correction remains the gold standard for station-level temperature bias correction.** When you have point observations and want point predictions, tabular ML on meteorological features outperforms deep learning architectures designed for gridded data.

**Reference**: Rasp & Lerch (2018) — Neural networks for postprocessing ensemble weather forecasts. 536 German stations, RMSE ~1.4°C, outperforming classical EMOS significantly.

---

## 5. Tree-Based ML Methods

### Why Trees Dominate Point Downscaling

| Advantage | Explanation |
|-----------|-------------|
| **Tabular efficiency** | Naturally handle mixed feature types (continuous weather, categorical land cover) |
| **Feature selection** | Built-in importance ranking; handle irrelevant features gracefully |
| **Non-linearity** | Capture complex interactions (e.g., elevation × humidity → temperature) |
| **Speed** | Training in seconds on 10K-100K samples |
| **Interpretability** | SHAP values provide physical insight |

### Key Results from Literature

| Paper | Method | Region | Variable | MAE/RMSE | vs Baseline |
|-------|--------|--------|----------|----------|-------------|
| Cao et al. (2024) | XGBoost | Mountain plateau | LST | 1.95K / 2.67K | Best of 5 methods |
| Ahmad et al. (2024) | XGBoost | Pakistan (S. Asia) | Temperature | Excellent | Works in our climate zone |
| Pan et al. (2024) | Random Forest | Urban | Air Temp | >30% MAE reduction | vs interpolation |
| An et al. (2025) | LightGBM | Urban/suburban | LST | R² > 0.85 | With satellite aux features |
| Our Phase 2A | LightGBM | India (15 stations) | 2m Temp | 0.79°C | 47% reduction vs GFS |

### Our Elevation Paradox — Explained by Literature
Meyer et al. (2018) and Ploton et al. (2020) explain exactly why adding raw station elevation *worsened* our spatial CV:

> "Random forests are highly susceptible to spatial autocorrelation... models memorize location-specific features (like elevation at station X) rather than learning transferable physical relationships."

The solution from literature: Use **derived terrain features** (lapse rate corrections, relative elevation within neighborhood, slope, aspect) rather than absolute elevation, OR use spatial block CV with forward feature selection to automatically reject features that don't generalize.

---

## 6. CNN-Based Super-Resolution

### DeepSD (Vandal et al., 2017)
The foundational paper for climate downscaling as image super-resolution:
- Stacked SRCNN with DEM as auxiliary input channel
- ~100 km → ~12.5 km (8× upscaling)
- RMSE ~1.2°C on CONUS
- **Code**: [tjvandal/deepsd](https://github.com/tjvandal/deepsd) (Python 2 / TF1.x — outdated)
- **Key insight**: Treating gridded weather as images and using SR architectures works

### EDSR / VDSR Residual Networks (2024)
Modern residual super-resolution applied to temperature:
- Learn high-frequency difference between interpolated coarse input and high-res target
- L1 loss (MAE) preferred for EDSR; MSE for VDSR
- **Improved PSNR and SSIM in complex terrain including Himalayas**
- **Code**: [Zenodo:14828131](https://zenodo.org/records/14828131) (PyTorch)
- **Key insight**: Residual learning in the image domain is equivalent to our `ΔT_ML` formulation

### Limitation for Our Project
CNN super-resolution requires **dense gridded targets** for training. We have point observations, not high-resolution grids. To use CNNs, we would need to:
1. Train on ERA5-Land (dense) as a proxy for observations
2. Fine-tune with station observations using a custom loss

---

## 7. U-Net Architectures

### Why U-Nets Are Popular for Weather Downscaling

The U-Net encoder-decoder with skip connections is the most widely used architecture:
- **Encoder** captures multi-scale context (synoptic patterns)
- **Decoder** reconstructs fine-scale details
- **Skip connections** preserve high-frequency information

### Key Papers

**Sha et al. (2020)** — 2m Temperature Downscaling in Complex Terrain:
- U-Net outperforms linear lapse-rate corrections
- RMSE 1.5-1.8°C depending on season
- British Columbia (similar complex terrain to Indian mountains)

**BC-Unet (NOAA, 2024)** — GFS 2m Temperature Bias Correction:
- U-Net with CBAM (Convolutional Block Attention Module)
- Directly targets GFS 2m temperature — **our exact variable and model**
- CBAM helps model focus on complex topographic regions
- Reduces spatial biases significantly over CONUS

**Probabilistic U-Net (2024)**:
- Injects variational latent space for uncertainty quantification
- Uses afCRPS + WMSE-MS-SSIM loss
- Better at representing spatial variability and extremes

### CBAM Attention — Critical Addition
The Convolutional Block Attention Module (CBAM) adds:
1. **Channel attention**: Which input variables matter most at each location?
2. **Spatial attention**: Which spatial regions need the most correction?

This is especially relevant for India's varied terrain where Western Ghats, Himalayas, Indo-Gangetic plains, and coastal regions all require different correction patterns.

---

## 8. Vision Transformers & Attention

### ClimaX (Microsoft, 2023)
- **Architecture**: ViT with variable tokenization and aggregation
- **Pre-trained** on CMIP6 using self-supervised learning
- **Fine-tuned** on ERA5 for downscaling (5.625° → 0.25°)
- **Key advantage**: Variable tokenization handles flexible input sets
- **Code**: [microsoft/ClimaX](https://github.com/microsoft/ClimaX)

### PrecipFormer
- Combines ViT with CBAM attention
- Focuses attention on critical spatial regions (mountains, coastlines)
- Relevant for targeting Indian topographic complexity

### 1EMD Multi-Variable ViT
- Shared encoder + variable-specific decoders
- 29-32% compute reduction vs single-variable models
- Cross-variable knowledge transfer reduces MSE

### Assessment for PanchayatCast
Vision Transformers offer global spatial context but require:
- Large training datasets (pre-training is essential)
- Significant compute (GPUs for training)
- Gridded inputs/outputs (not naturally suited for point observations)

**Verdict**: ClimaX is a strong candidate for a future gridded version but is **overkill for our current point-observation regime**.

---

## 9. Diffusion Models

### CorrDiff (NVIDIA, 2023) — THE Key Reference

**This is the most relevant architecture for our project.**

**Architecture**:
1. **Stage 1**: U-Net regression model predicts the deterministic mean state
2. **Stage 2**: Patch-based DDPM predicts the stochastic residual (high-frequency details)

**Training Data**: 25 km GEFS → 3 km HRRR (CONUS); also ERA5 → WRF (Taiwan)

**Results**:
- 1000× faster than WRF dynamical downscaling
- Superior CRPS and Power Spectral Density
- Captures extreme events (typhoons, intense precipitation)

**Code**: [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) (PyTorch, Modulus ecosystem)

**Why it matters for PanchayatCast**: CorrDiff's two-stage approach directly maps to our formulation:
- Stage 1 = Our LightGBM residual corrector (deterministic mean)
- Stage 2 = Diffusion model for spatial uncertainty / fine-scale details

### China 3km Adaptation (2024)
CorrDiff adapted to continental China — proves the approach scales to large, diverse topological domains like India (arXiv:2512.15549).

### SwinRDM (AAAI 2023)
SwinRNN for coarse prediction + Diffusion for super-resolution. Specifically evaluates T2M. Validates the two-stage coarse-then-refine approach.

### Practical Considerations
- Diffusion models require **multiple forward passes** for sampling (50-1000 steps)
- Consistency models (p-hss/consistency-climate-downscaling) offer **single-step** alternatives
- Requires dense gridded training targets

---

## 10. GANs for Weather SR

### ESRGAN (2020)
- Adversarial + Perceptual + L1 loss
- Produces sharper outputs than bicubic or simple CNNs
- **Problem**: Sometimes has worse MSE because it "hallucinates" high-frequency textures
- Not recommended for temperature where physical accuracy is paramount

### Stochastic SRGAN (2020)
- Recurrent architecture for temporal consistency
- Generates ensembles of high-res temporal sequences
- Useful for maintaining hourly forecast coherence

### Assessment
> **Diffusion models have largely superseded GANs** for weather applications due to:
> 1. More stable training
> 2. Better extreme value capture
> 3. No mode collapse
> 4. Controllable diversity through noise scheduling

---

## 11. Graph Neural Networks

### GraphCast + WRF Hybrid
GraphCast produces 0.25° global forecasts, then WRF provides dynamical downscaling. Computationally expensive — defeats the purpose of ML downscaling.

### GraphCast + Super-Resolution (2024)
Post-processing GraphCast outputs with CNN super-resolution. Similar to applying SR to GFS outputs.

### Assessment for PanchayatCast
GNNs are more natural for **irregular station networks** (our point observation case) than CNNs/U-Nets (which need regular grids). However, no mature weather downscaling GNN implementation exists that handles our specific problem.

---

## 12. Foundation Models

| Model | Organization | Architecture | Resolution | Status |
|-------|-------------|-------------|------------|--------|
| Pangu-Weather | Huawei | 3D Transformer | 0.25° | Could replace GFS |
| GenCast | DeepMind | Diffusion | 0.25° | Ensemble generation |
| FourCastNet | NVIDIA | Fourier Neural Operator | 0.25° | Global forecasting |
| ClimaX | Microsoft | ViT | Variable | Pre-trainable |
| FuXi | Fudan U | Cascade ML | 0.25° | Residual correction |
| MetNet-3 | Google | U-Net + densification | 1-4 km | Station integration |

### MetNet-3 — The Most Relevant Concept
MetNet-3 integrates **sparse ground station data** to predict dense grids by **randomly masking stations (25%) during training**. This "densification" technique is exactly how we could integrate our Indian point observations:
1. Place station observations on a sparse grid
2. During training, randomly mask 25% of stations
3. Model learns to interpolate/reconstruct from remaining stations
4. At inference, model densifies sparse observations to full panchayat grid

---

## 13. Residual Learning Paradigm

### Why Residual Correction Works

The consensus across the literature is overwhelming:

| Paper/System | Formulation | Why It Works |
|-------------|-------------|--------------|
| CorrDiff (NVIDIA) | `Y = U-Net_mean + Diffusion_residual` | Separates deterministic and stochastic components |
| FuXi (2023) | Adaptive residual correction | Prevents error accumulation over lead time |
| EDSR/VDSR (2024) | `Y = Bicubic_interp + CNN_residual` | Preserves large-scale structure |
| Our Phase 2A | `T = GFS + ΔT_LightGBM` | 47% MAE reduction over raw GFS |
| Rasp & Lerch (2018) | `T = NWP_ensemble + NN_correction` | Outperforms EMOS |

### Why Not Predict Absolute Temperature?
1. **Preservation of physical structure**: GFS already captures synoptic-scale patterns; we only need local corrections
2. **Smaller prediction range**: Residuals are O(1°C) vs absolute temperatures O(30°C)
3. **Faster convergence**: Model only needs to learn the error, not the full temperature field
4. **Better generalization**: Residual patterns are more transferable across locations

### Our Validation
Our Phase 2A experiments confirmed this:
- GFS Baseline MAE: 1.66°C
- LightGBM Residual Correction: 0.79°C (47% reduction)
- Adding raw elevation worsened performance → terrain feature engineering matters

---

## 14. Loss Functions Comparison

| Loss Function | Properties | When to Use | Papers |
|---------------|-----------|-------------|--------|
| **MSE (L2)** | Minimizes mean squared error; oversmooths extremes | Baseline; deterministic mean prediction | DeepSD, ClimaX |
| **MAE (L1)** | More robust to outliers; preserves sharper features | EDSR; when extremes matter | EDSR, BC-Unet |
| **Huber Loss** | Smooth transition from L2 (small errors) to L1 (large errors) | Best default for temperature residuals | General recommendation |
| **CRPS** | Proper scoring rule for probabilistic forecasts | Probabilistic downscaling; ensemble calibration | Rasp & Lerch, Probabilistic U-Net |
| **Pinball/Quantile** | Targets specific quantiles (P10, P50, P90) | Uncertainty bounds; extreme event prediction | Cannon (2011), Q-SRDRN |
| **Perceptual Loss** | Compares feature representations, not pixel values | Maintaining spatial texture/sharpness | ESRGAN, ML review recommendation |
| **Score Matching** | Trains diffusion models to denoise | Generative probabilistic downscaling | CorrDiff, GenCast, SwinRDM |
| **Adversarial** | Discriminator-based realism enforcement | Sharp outputs; risk of hallucination | ESRGAN, Stochastic SRGAN |
| **Physical Constraint** | Penalizes violations of conservation laws | Multi-variable consistency | González-Abad (AAAI) |

### Recommendation for PanchayatCast
**Start with Huber Loss** for the deterministic residual model. If probabilistic outputs are needed for advisory services, add **Quantile Loss** (P10, P50, P90) or **CRPS**.

---

## 15. Training Strategies

### Transfer Learning (Critical for Data-Scarce India)
With only 15-100 station observations:
1. **Pre-train** on dense global dataset (ERA5 → ERA5-Land) to learn general downscaling patterns
2. **Fine-tune** on sparse Indian station observations
3. Expected improvement: **15-25% MAE reduction** vs training from scratch (Transfer Learning review, 2024)

### Self-Supervised Pre-Training
Masked Autoencoders (MAE) on unlabeled ERA5 data:
- Up to 10% MSE improvement over fully supervised baselines when labeled data is limited
- Leverages vast historical GFS data without needing dense targets

### MetNet-3 Densification Strategy
- Place station observations on a sparse grid
- During training, randomly mask 25% of stations
- Forces model to learn spatial interpolation
- **Directly applicable to our sparse Indian station network**

### Curriculum Learning
Not widely adopted in weather downscaling, but a natural fit:
1. Start with easy cases (flat terrain, moderate temperatures)
2. Progressively add complex cases (mountains, extremes)

### Ensemble Stacking
Combine multiple base models (LightGBM, Ridge, RF) with a meta-learner:
- Our benchmark showed LightGBM was best single model
- Stacking could squeeze additional 5-10% improvement

---

## 16. Uncertainty Quantification

### Methods in Literature

| Method | Approach | Calibration | Compute | Our Status |
|--------|----------|-------------|---------|-----------|
| **Conformal Prediction** | Distribution-free coverage guarantee | ✅ Calibrated by construction | Low | ✅ Implemented |
| **Quantile GBDT** | Direct quantile regression (P10/P50/P90) | Needs post-hoc calibration | Low | ✅ Implemented |
| **MC Dropout** | Multiple stochastic forward passes | Often miscalibrated | Medium | Not implemented |
| **Ensemble Spread** | Multiple models with different seeds | Moderate | Medium | Planned |
| **CRPS Loss** | Optimizes full predictive distribution | ✅ Proper scoring rule | Medium | Planned |
| **Diffusion Sampling** | Multiple denoising trajectories | Good for spatial uncertainty | High | Future |

### Key Finding (Sousa et al., 2026)
Using **Earth observation embeddings** (satellite imagery features) as static descriptors improves CRPS by 11.5% for 2m temperature. This suggests that beyond DEM/elevation, satellite-derived surface descriptors carry meaningful information for downscaling uncertainty.

---

## 17. Spatial Validation Methods

### The Critical Problem
> **Random train/test splits overestimate spatial prediction skill by 30-50%** (Meyer et al., 2018; Ploton et al., 2020).

This is because nearby stations share similar weather (spatial autocorrelation), so a model tested on spatially proximate training stations appears to perform well but fails on distant, unseen locations.

### Required Validation Framework

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Leave-Location-Out (LLO)** | Hold out entire stations | Station-level generalization |
| **Spatial Block CV** | Group nearby stations into blocks | Regional generalization |
| **Buffered CV** | LLO + exclusion buffer around test points | Strict spatial independence |
| **Temporal Holdout** | Train on past, test on future | Operational forecast skill |
| **Spatiotemporal Holdout** | Both new locations AND new times | Ultimate generalization test |

### Our Implementation
We already implement 5-fold GroupKFold spatial CV with geographic station clusters. This is correct and aligns with Meyer et al.'s recommendations.

### Forward Feature Selection with LLO
Meyer et al. (2018) recommend combining LLO CV with forward feature selection — only add features that improve LLO performance. This would automatically reject features like raw station elevation that memorize station identity.

---

## 18. Terrain-Aware Features

### The Elevation Paradox Resolved

Our Phase 2A found that adding raw station elevation **worsened** spatial CV. The literature explains why and provides solutions:

| Approach | How Elevation Is Used | Works? | Reference |
|----------|----------------------|--------|-----------|
| **Raw scalar** | Single elevation value per station | ❌ Memorizes stations | Our Phase 2A |
| **Relative elevation** | Station elevation minus neighborhood mean | ✅ Transferable | Various terrain studies |
| **DEM as image channel** | High-res grid input to CNN/U-Net | ✅ Spatial patterns | DeepSD, EDSR, CorrDiff |
| **Lapse rate correction** | `T_corrected = T_GFS - Γ × (z_station - z_GFS)` | ✅ Physical | Sha et al. (2020) |
| **Derived features** | Slope, aspect, TPI, relief, curvature | Partially | Our ablation (mixed) |
| **EO embeddings** | Satellite-derived surface descriptors | ✅ +11.5% CRPS | Sousa et al. (2026) |

### Recommended Approach
1. **For point models (LightGBM)**: Use relative elevation (Δz), slope, and relief — NOT raw elevation
2. **For gridded models (U-Net)**: Feed high-res DEM as an auxiliary input channel
3. **For advanced models**: Use Earth observation embeddings as in Sousa et al.

---

## 19. Operational Systems Worldwide

### Current Operational Landscape

| Organization | System | Resolution | Method | Status |
|-------------|--------|------------|--------|--------|
| **ECMWF** | AIFS + Anemoi | 0.25° → higher | Flow-matching, Transformers, GNNs | R2O transition |
| **ECMWF** | ecPoint | Coarse → point | Statistical post-processing | Operational |
| **NOAA/NWS** | HRRR-Cast, EAGLE | 3 km | ResNets, GNNs, LightGBM | R2O / Experimental |
| **UK Met Office** | AI4NWP (FastNet) | Variable | Augmented ML on physics NWP | Experimental |
| **IMD** | BharatFS | ~6 km | TCo grid, dynamical | Operational |
| **IMD** | MausamGram | Panchayat (~1 km) | Statistical interpolation | Operational portal |
| **NCMRWF** | UP Rainfall Downscale | ~1 km | AI-driven diffusion | R2O |
| **IITM Pune** | Various research | Variable | ConvLSTM, GANs, DeepSD-style | Research |

### Key Insight
IMD's BharatFS already provides **6 km forecasts** via the Mithuna model. MausamGram attempts to downscale these to panchayat level. Our system must either:
1. Improve upon MausamGram's interpolation for specific variables (2m temperature)
2. Use GFS/BharatFS as base and add ML residual correction

---

## 20. India-Specific Systems & Context

### IMD BharatFS
- **Native resolution**: ~6 km using TCo grids
- **Models**: T1534 GFS, Mithuna (global + regional)
- **Portal**: MausamGram serves forecasts at gram panchayat level
- **Our opportunity**: BharatFS uses statistical interpolation for panchayat downscaling; ML residual correction could significantly improve this

### NCMRWF AI Downscaling
- Using **AI-driven diffusion technology** for 1 km downscaling
- Focus on rainfall (UP Rainfall Downscale system)
- Transitioning from research to operations
- **Validates our approach**: India's own meteorological institutions are moving to ML/DL downscaling

### IITM Pune Research
- Active research on ConvLSTM, GANs, and DeepSD-style architectures
- Focus on Indian monsoon dynamics
- Using ERA5 and TRMM training data

### Agrometeorological Advisory Context
- India's Agromet Advisory Service (AAS) provides crop-weather advisories
- Currently operates at district/block level
- Panchayat-level advisories require downscaled temperature
- Errors in our downscaling directly propagate to erroneous crop advisories
- Tree-based ML + CNN-LSTMs are used for crop-weather advisory systems

---

## 21. Open-Source Codebases

### Tier 1: Directly Applicable

| Repository | Architecture | Framework | Relevance |
|-----------|-------------|-----------|-----------|
| [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) (CorrDiff) | U-Net + Patch Diffusion | PyTorch | Our exact use case |
| [carlos-gg/dl4ds](https://github.com/carlos-gg/dl4ds) | ResNet, U-Net, CGAN, WGAN | TF/Keras | Off-the-shelf SR toolkit |
| [microsoft/ClimaX](https://github.com/microsoft/ClimaX) | Vision Transformer | PyTorch | Foundation model for fine-tuning |
| [srasp/nn_weather_postprocessing](https://github.com/srasp/nn_weather_postprocessing) | MLP point post-processing | Python | Baseline for station correction |
| [pangeo-data/scikit-downscale](https://github.com/pangeo-data/scikit-downscale) | RF, regression, analogs, BCSD | scikit-learn | Mandatory baseline |

### Tier 2: Useful Components

| Repository | Architecture | Framework | Relevance |
|-----------|-------------|-----------|-----------|
| [RolnickLab/constrained-downscaling](https://github.com/RolnickLab/constrained-downscaling) | Physics-constrained CNN/GAN | PyTorch | Conservation law enforcement |
| [robbiewatt1/ClimateDiffuse](https://github.com/robbiewatt1/ClimateDiffuse) | EDM diffusion | PyTorch | Simple diffusion prototype |
| [NatLabRockies/neural-operators-weather-downscaling](https://github.com/NatLabRockies/neural-operators-weather-downscaling) | Fourier Neural Operator | PyTorch | Resolution-invariant |
| [p-hss/consistency-climate-downscaling](https://github.com/p-hss/consistency-climate-downscaling) | Consistency Model | PyTorch | Fast single-step inference |
| [SantanderMetGroup/downscaleR.keras](https://github.com/SantanderMetGroup/downscaleR.keras) | CNN/U-Net | R/Keras | Reproducible baselines |

### Tier 3: Reference / Historical

| Repository | Architecture | Notes |
|-----------|-------------|-------|
| [tjvandal/deepsd](https://github.com/tjvandal/deepsd) | Stacked SRCNN | Python 2 / TF1.x (outdated) |
| [pacificclimate/ClimDown](https://github.com/pacificclimate/ClimDown) | BCCAQ | R-based |
| [tpys/FuXi](https://github.com/tpys/FuXi) | Cascade ML | Residual correction methodology |
| Zenodo:14828131 | EDSR/VDSR | Residual Matters (PyTorch) |

---

## 22. Data Sources & Availability

| Dataset | Provider | Resolution | Variables | Access | Cost |
|---------|----------|------------|-----------|--------|------|
| **GFS** | NOAA | 0.25° (~25 km) | 2m Temp, precip, wind, humidity | Open-Meteo API | Free |
| **ERA5** | ECMWF/Copernicus | 0.25° (~31 km) | Complete atmospheric state | CDS API | Free |
| **ERA5-Land** | ECMWF/Copernicus | 0.1° (~9 km) | Land surface variables | CDS / Open-Meteo | Free |
| **SRTM DEM** | NASA | 30m / 90m | Elevation | Open-Meteo Elevation API | Free |
| **ESA WorldCover** | ESA | 10m | Land cover classes | Public | Free |
| **MODIS NDVI** | NASA | 250m-1km | Vegetation index | AppEEARS / GEE | Free |
| **IMD AWS** | IMD | Station-level | Temperature, precip, wind | Requires credentials | Restricted |
| **ICAR** | India | Station-level | Agro-meteorological | Restricted | Restricted |

### Data Strategy for PanchayatCast
1. **GFS forecasts**: Via Open-Meteo Historical Forecast API (verified working)
2. **Ground truth**: ERA5-Land as dense proxy; IMD AWS for real stations (if accessible)
3. **Terrain**: SRTM 30m via Open-Meteo Elevation API (verified working)
4. **Land surface**: ESA WorldCover 10m (implemented as lookup table)

---

## 23. Evaluation Metrics

### Point-Level Metrics (Our Primary)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **MAE** | Mean Absolute Error | Average magnitude of errors |
| **RMSE** | Root Mean Squared Error | Penalizes large errors |
| **Bias** | Mean Error (signed) | Systematic over/under-prediction |
| **R²** | Coefficient of determination | Variance explained |
| **Skill Score** | 1 - RMSE_model/RMSE_baseline | Improvement over GFS |

### Spatial Metrics (For Gridded Output)

| Metric | What It Measures |
|--------|-----------------|
| **SSIM** | Structural similarity of spatial patterns |
| **PSD** | Power spectral density (spatial frequency content) |
| **FSS** | Fractions skill score (spatial extent of events) |

### Probabilistic Metrics

| Metric | What It Measures |
|--------|-----------------|
| **CRPS** | Overall quality of predictive distribution |
| **Coverage** | Fraction of observations within prediction intervals |
| **Sharpness** | Width of prediction intervals |
| **Reliability** | Calibration of probability assessments |

### Recommendation from Literature (Langguth et al.)
Evaluate using: **Point-wise RMSE/MAE** at station locations + **Spatial PSD** to ensure the downscaled temperature map looks physically realistic (not oversmoothed).

---

## 24. Computational Requirements

| Architecture | Training Time | GPU Memory | Inference Time | Our Feasibility |
|-------------|--------------|------------|---------------|-----------------|
| LightGBM | ~5 seconds | CPU only | <1 ms/sample | ✅ Immediate |
| XGBoost | ~10 seconds | CPU only | <1 ms/sample | ✅ Immediate |
| MLP | ~30 seconds | <1 GB | <1 ms/sample | ✅ Easy |
| CNN/SRCNN | ~1 hour | 4 GB | ~10 ms/grid | ✅ Feasible |
| U-Net | ~2-6 hours | 8 GB | ~50 ms/grid | ⚠️ Need GPU |
| U-Net + CBAM | ~4-8 hours | 8-16 GB | ~100 ms/grid | ⚠️ Need GPU |
| ClimaX (fine-tune) | ~12-24 hours | 16-32 GB | ~200 ms/grid | ⚠️ Cloud GPU |
| CorrDiff (full) | ~24-72 hours | 32+ GB | ~5s (50 steps) | ❌ Too heavy |
| CorrDiff (lite) | ~6-12 hours | 16 GB | ~1s (10 steps) | ⚠️ Cloud GPU |

### Our Compute Constraints
- Local machine: No GPU (CPU-only development)
- Free cloud: Google Colab (T4 GPU, 12-15 GB, session limits)
- **Strategy**: Keep LightGBM as primary; implement U-Net if GPU becomes available

---

## 25. Common Failure Modes

### 1. Spatial Overfitting (Most Common)
- **Symptom**: Excellent random CV, terrible spatial CV
- **Cause**: Model memorizes station-specific patterns (elevation, proximity)
- **Solution**: Leave-Location-Out CV; avoid raw station coordinates as features
- **References**: Meyer et al. (2018), Ploton et al. (2020)

### 2. Oversmoothing (Deep Learning)
- **Symptom**: Low RMSE but spatial patterns are blurry; extremes are muted
- **Cause**: MSE loss minimizes average error by predicting the mean
- **Solution**: Use L1/Huber loss; add perceptual loss or adversarial training
- **References**: ML Climate Review (2024), ESRGAN (2020)

### 3. Physical Implausibility
- **Symptom**: Predicted temperatures violate physical constraints (e.g., RH > 100%)
- **Cause**: Unconstrained optimization
- **Solution**: Hard constraint layers (González-Abad, AAAI 2024)

### 4. Non-Stationarity
- **Symptom**: Model performs well on historical data but degrades on recent/future data
- **Cause**: Climate drift, model version changes, seasonal distribution shift
- **Solution**: Rolling retraining; domain adaptation; use anomalies not absolutes

### 5. Terrain Feature Leakage
- **Symptom**: Adding elevation features improves temporal CV but worsens spatial CV
- **Cause**: Elevation uniquely identifies stations → spatial data leakage
- **Solution**: Use relative/derived terrain features; forward feature selection with LLO
- **Our experience**: Confirmed in Phase 2A (Elevation Paradox)

---

## 26. Architecture Comparison Matrix

| Criterion | LightGBM | U-Net | U-Net+CBAM | CorrDiff | ClimaX |
|-----------|----------|-------|------------|----------|--------|
| **Point observations** | ✅ Native | ❌ Needs grid | ❌ Needs grid | ❌ Needs grid | ❌ Needs grid |
| **Sparse data (15-100 pts)** | ✅ Works | ⚠️ Needs pre-train | ⚠️ Needs pre-train | ❌ Data hungry | ⚠️ Needs pre-train |
| **Complex terrain** | ⚠️ Feature engineering | ✅ Learns spatial | ✅ Attends to terrain | ✅ Learns spatial | ✅ Learns spatial |
| **Uncertainty** | ✅ Quantile/conformal | ⚠️ Needs extension | ⚠️ Needs extension | ✅ Native ensemble | ⚠️ Needs extension |
| **Training speed** | ✅ Seconds | ⚠️ Hours | ⚠️ Hours | ❌ Days | ❌ Days |
| **Inference speed** | ✅ <1 ms | ✅ ~50 ms | ✅ ~100 ms | ⚠️ ~5 s | ⚠️ ~200 ms |
| **No GPU needed** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Interpretability** | ✅ SHAP | ❌ Black box | ⚠️ Attention maps | ❌ Black box | ❌ Black box |
| **Operational deployment** | ✅ Easy | ⚠️ Moderate | ⚠️ Moderate | ❌ Complex | ❌ Complex |
| **Our MAE (spatial CV)** | 0.79°C | Unknown | Unknown | Unknown | Unknown |
| **Literature MAE/RMSE** | ~0.8-1.5°C | ~1.5-1.8°C | Reduced bias | Superior CRPS | Low NRMSE |

---

## 27. Key Insights for PanchayatCast

### Insight 1: We Are in the Right Paradigm
Our residual correction approach (`T = GFS + ΔT_ML`) aligns with the global state-of-the-art (CorrDiff, FuXi, EDSR, Rasp & Lerch). This is **not** a second-best compromise — it is the dominant methodology.

### Insight 2: Our LightGBM Is Competitive
For point observations with sparse station networks, LightGBM residual correction achieving 0.79°C MAE is within the range of literature results (0.8-1.5°C for similar problems). We are not underperforming.

### Insight 3: The Real Bottleneck Is Data, Not Architecture
With 15-100 stations, architecture choice matters less than:
1. How terrain features are represented
2. How validation is performed (spatial vs random)
3. How much training data we have (more months > more stations)

### Insight 4: The Path to Deep Learning Requires Dense Targets
To use U-Nets/diffusion models, we need a dense gridded target like ERA5-Land (~9 km). This becomes our "proxy ground truth" for training, with station observations for final fine-tuning and validation.

### Insight 5: IMD Is Already Doing This
NCMRWF is using diffusion for 1 km downscaling. BharatFS provides 6 km forecasts. Our contribution must clearly articulate what we add beyond existing IMD capabilities.

---

## 28. What We Should Do Next

### Actionable Item 1: Expand Station Network
- Scale from 15 to 100+ stations across India
- Include diverse climate zones (coastal, mountain, desert, plains)
- Target IMD AWS data access (even if through a data-sharing agreement)

### Actionable Item 2: Fix Terrain Feature Representation
- Replace raw elevation with **relative elevation** (Δz from GFS grid cell mean)
- Implement **lapse rate correction**: `T_corrected = T_GFS - 6.5°C/km × (z_station - z_GFS)`
- Use forward feature selection with LLO CV to automatically identify beneficial features

### Actionable Item 3: Add ERA5-Land as Dense Proxy Target
- Fetch ERA5-Land 0.1° (~9 km) 2m temperature as "ground truth" grid
- This enables training gridded models (U-Net, EDSR)
- Validate against actual station observations

### Actionable Item 4: Implement BC-Unet (U-Net + CBAM)
- Follow NOAA's BC-Unet architecture for GFS 2m temperature bias correction
- Input channels: GFS fields + DEM + land cover
- Loss: Huber or L1
- Requires GPU (Google Colab T4)

### Actionable Item 5: Prototype CorrDiff-Lite
- Implement a simplified version of CorrDiff:
  - Stage 1: LightGBM for deterministic mean (already done)
  - Stage 2: Small diffusion model for residual uncertainty
- Use ClimateDiffuse codebase as starting point (simpler than NVIDIA Modulus)

### Actionable Item 6: Add Transfer Learning
- Pre-train on global ERA5 → ERA5-Land downscaling
- Fine-tune on Indian station observations
- Expected: 15-25% MAE improvement

### Actionable Item 7: Implement MetNet-3 Densification
- Place station observations on sparse grid
- Random masking (25%) during training
- Model learns to interpolate to unmonitored panchayats

### Actionable Item 8: Enhance Uncertainty Quantification
- Add CRPS loss alongside MAE for probabilistic calibration
- Implement ensemble of LightGBM models (different seeds/hyperparameters)
- Compare with existing conformal prediction intervals

### Actionable Item 9: Build Comprehensive Evaluation Suite
- Point metrics: MAE, RMSE, Bias, R², Skill Score
- Spatial metrics: SSIM, PSD (if gridded output)
- Probabilistic: CRPS, Coverage, Sharpness
- Validation: LLO CV, Temporal holdout, Spatiotemporal holdout

### Actionable Item 10: Position Against IMD/MausamGram
- Clearly benchmark against BharatFS 6 km (if data accessible)
- Articulate our ML residual correction as an **add-on** to BharatFS, not a replacement
- Target the specific gap: BharatFS interpolation → ML-corrected panchayat forecast

---

## References

Full database of 42 papers and repositories available in `results/literature_database.csv`.

### Top 10 Must-Read Papers
1. Mardani et al. (2023) — CorrDiff: Residual Corrective Diffusion (arXiv:2309.15214)
2. Rasp & Lerch (2018) — Neural Networks for Postprocessing (DOI:10.1175/MWR-D-18-0187.1)
3. Meyer et al. (2018) — Spatial Validation (DOI:10.1016/j.envsoft.2018.03.020)
4. Sha et al. (2020) — U-Net for 2m Temperature (DOI:10.1175/JAMC-D-20-0057.1)
5. Vandal et al. (2017) — DeepSD (DOI:10.1145/3097983.3098004)
6. Nguyen et al. (2023) — ClimaX (arXiv:2301.10343)
7. NOAA BC-Unet (2024) — GFS Bias Correction
8. Sousa et al. (2026) — Earth Observation Embeddings for Downscaling
9. Ploton et al. (2020) — Spatial Validation Warning (DOI:10.1038/s41467-020-18321-y)
10. González-Abad et al. (2024) — Physics-Constrained Downscaling (AAAI)
