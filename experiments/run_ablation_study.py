"""Controlled Scientific Ablation Study Script.

Evaluates Models 0 through 5 across:
- 5-Fold Spatial CV (unseen locations)
- Temporal Holdout (Aug 22-31 future dates)
- Spatiotemporal Holdout (unseen locations + future dates)

Records all metrics and feature importances to results/metrics/.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_unified_dataset
from src.features.pipeline import FEATURE_SETS
from src.models.residual_model import DownscalingResidualModel
from src.validation.spatial_cv import run_spatial_cv_experiment
from src.validation.temporal_cv import run_temporal_validation
from src.validation.spatiotemporal import run_spatiotemporal_validation

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STARTING CONTROLLED SCIENTIFIC ABLATION STUDY")
    print("=" * 80)

    df = build_unified_dataset(start_date="2024-08-01", end_date="2024-08-31")

    # Baseline Model 0 metrics across whole dataset
    y_all = df["temperature_2m_era5"].to_numpy()
    gfs_all = df["temperature_2m_gfs"].to_numpy()
    m0_mae = mean_absolute_error(y_all, gfs_all)
    m0_rmse = np.sqrt(mean_squared_error(y_all, gfs_all))

    # Baseline Model 0 temporal metrics
    mask_future = df["time"] >= "2024-08-22T00:00:00"
    m0_temp_mae = mean_absolute_error(y_all[mask_future], gfs_all[mask_future])
    m0_temp_rmse = np.sqrt(mean_squared_error(y_all[mask_future], gfs_all[mask_future]))

    models_to_evaluate = [
        ("Model_0_GFS_Baseline", []),
        ("Model_1_Weather_Time", FEATURE_SETS["Model_1_Weather_Time"]),
        ("Model_B1_Weather_Time_Elevation", FEATURE_SETS["Model_B1_Weather_Time_Elevation"]),
        ("Model_2_Weather_Time_Terrain", FEATURE_SETS["Model_2_Weather_Time_Terrain"]),
        ("Model_3_Weather_Time_Surface", FEATURE_SETS["Model_3_Weather_Time_Surface"]),
        ("Model_4_Weather_Time_Terrain_Surface", FEATURE_SETS["Model_4_Weather_Time_Terrain_Surface"]),
        ("Model_5_Full_Physiographic_NDVI", FEATURE_SETS["Model_5_Full_Physiographic_NDVI"]),
    ]

    summary_rows = []
    spatial_fold_details = []

    for name, feats in models_to_evaluate:
        print(f"\nEvaluating {name} ({len(feats)} features)...")

        if name == "Model_0_GFS_Baseline":
            summary_rows.append({
                "Model": name,
                "Features_Count": 0,
                "Spatial_MAE": round(m0_mae, 4),
                "Spatial_MAE_Std": 0.0,
                "Spatial_RMSE": round(m0_rmse, 4),
                "Spatial_RMSE_Std": 0.0,
                "Temporal_MAE": round(m0_temp_mae, 4),
                "Temporal_RMSE": round(m0_temp_rmse, 4),
                "Spatiotemporal_MAE": round(m0_temp_mae, 4),
                "Spatiotemporal_RMSE": round(m0_temp_rmse, 4),
                "Spatial_MAE_Reduction_Pct": 0.0,
                "Spatial_RMSE_Reduction_Pct": 0.0
            })
            continue

        # 1. Spatial CV
        spat_res = run_spatial_cv_experiment(df, feature_cols=feats, model_type="lightgbm", model_name=name)
        spat_sum = spat_res["summary"]

        for _, r in spat_res["fold_df"].iterrows():
            f_dict = r.to_dict()
            f_dict["model_name"] = name
            spatial_fold_details.append(f_dict)

        # 2. Temporal Holdout
        temp_res = run_temporal_validation(df, feature_cols=feats, split_date="2024-08-22T00:00:00", model_type="lightgbm", model_name=name)

        # 3. Spatiotemporal Holdout
        sptemp_res = run_spatiotemporal_validation(df, feature_cols=feats, split_date="2024-08-22T00:00:00", model_type="lightgbm", model_name=name)
        sptemp_sum = sptemp_res["summary"]

        summary_rows.append({
            "Model": name,
            "Features_Count": len(feats),
            "Spatial_MAE": spat_sum["mean_ref_mae"],
            "Spatial_MAE_Std": spat_sum["std_ref_mae"],
            "Spatial_RMSE": spat_sum["mean_ref_rmse"],
            "Spatial_RMSE_Std": spat_sum["std_ref_rmse"],
            "Temporal_MAE": temp_res["ref_mae"],
            "Temporal_RMSE": temp_res["ref_rmse"],
            "Spatiotemporal_MAE": sptemp_sum["mean_ref_mae"],
            "Spatiotemporal_RMSE": sptemp_sum["mean_ref_rmse"],
            "Spatial_MAE_Reduction_Pct": spat_sum["mean_mae_reduction_pct"],
            "Spatial_RMSE_Reduction_Pct": spat_sum["mean_rmse_reduction_pct"]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(METRICS_DIR / "ablation_study_summary.csv", index=False)
    pd.DataFrame(spatial_fold_details).to_csv(METRICS_DIR / "ablation_spatial_folds.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL ABLATION STUDY RESULTS TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Fit final production Model 5 on entire dataset and extract feature importances
    print("\nTraining Final Production Model 5 (Full Physiographic + NDVI) on full dataset...")
    full_model = DownscalingResidualModel(model_type="lightgbm")
    feats_m5 = FEATURE_SETS["Model_5_Full_Physiographic_NDVI"]
    full_model.fit(df, y_all, feats_m5)
    full_model.save(MODELS_DIR / "final_residual_model_lightgbm.joblib")

    feat_imp = full_model.get_feature_importances()
    feat_imp.to_csv(METRICS_DIR / "feature_importances_model5.csv", index=False)

    print("\nTop Feature Importances (Model 5):")
    print(feat_imp.head(10).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
