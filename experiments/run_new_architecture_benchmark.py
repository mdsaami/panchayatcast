"""Evaluation benchmark for the new literature-backed downscaling architecture.

Runs 5-fold Leave-Location-Out (LLO) Cross-Validation comparing:
1. GFS Baseline
2. Model 1 (Weather + Cyclical Time)
3. Model B1 (Weather + Time + Point Elevation - Historical Baseline)
4. Model 6 (New Physical Terrain: Lapse Rate + Relative Relief + TPI)
5. New Architecture: Multi-Quantile Residual Model (P10, P50, P90)
6. New Architecture: Stacked Residual Ensemble (LightGBM + HistGBDT + Ridge)
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
from src.features.pipeline import (
    FEATURE_SETS,
    add_cyclical_time_features,
    add_physical_terrain_features
)
from src.validation.spatial_cv import get_spatial_folds
from src.models.residual_model import DownscalingResidualModel
from src.models.advanced_residual import MultiQuantileResidualModel, StackedResidualEnsemble

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_spatial_cv(df: pd.DataFrame, model_type: str, feature_cols: list) -> dict:
    folds = get_spatial_folds()
    mae_list, rmse_list, bias_list = [], [], []

    for fold_idx in range(1, 6):
        train_locs, test_locs = folds[fold_idx]
        train_df = df[df["location_id"].isin(train_locs)]
        test_df = df[df["location_id"].isin(test_locs)]

        y_train = train_df["temperature_2m_era5"].to_numpy()
        y_test = test_df["temperature_2m_era5"].to_numpy()
        gfs_test = test_df["temperature_2m_gfs"].to_numpy()

        if model_type == "stacked_ensemble":
            model = StackedResidualEnsemble()
            model.fit(train_df, y_train, feature_cols)
            t_pred, _ = model.predict(test_df)
        elif model_type == "multi_quantile":
            model = MultiQuantileResidualModel()
            model.fit(train_df, y_train, feature_cols)
            preds = model.predict(test_df)
            t_pred = preds["t_pred"]
        else:
            model = DownscalingResidualModel(model_type=model_type)
            model.fit(train_df, y_train, feature_cols)
            t_pred = model.predict(test_df)

        mae = mean_absolute_error(y_test, t_pred)
        rmse = np.sqrt(mean_squared_error(y_test, t_pred))
        bias = float(np.mean(t_pred - y_test))

        mae_list.append(mae)
        rmse_list.append(rmse)
        bias_list.append(bias)

    return {
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "bias_mean": float(np.mean(bias_list))
    }


def main():
    print("=" * 75)
    print("NEW ARCHITECTURE BENCHMARK: 5-FOLD LEAVE-LOCATION-OUT (LLO) CV")
    print("=" * 75)

    df = build_unified_dataset(start_date="2024-08-01", end_date="2024-08-31")
    df = add_physical_terrain_features(df)
    
    y_true = df["temperature_2m_era5"].to_numpy()
    gfs_temp = df["temperature_2m_gfs"].to_numpy()
    gfs_mae = mean_absolute_error(y_true, gfs_temp)
    gfs_rmse = np.sqrt(mean_squared_error(y_true, gfs_temp))

    models_to_test = [
        ("Model_1_Weather_Time", "lightgbm", FEATURE_SETS["Model_1_Weather_Time"]),
        ("Model_B1_Point_Elevation", "lightgbm", FEATURE_SETS["Model_B1_Weather_Time_Elevation"]),
        ("Model_6_Physical_Terrain", "lightgbm", FEATURE_SETS["Model_6_Physical_Terrain"]),
        ("NewArch_Stacked_Ensemble", "stacked_ensemble", FEATURE_SETS["Model_6_Physical_Terrain"]),
        ("NewArch_Multi_Quantile", "multi_quantile", FEATURE_SETS["Model_6_Physical_Terrain"])
    ]

    records = []
    records.append({
        "Model": "Raw_GFS_Baseline",
        "Spatial_MAE": round(gfs_mae, 4),
        "Spatial_MAE_Std": 0.0,
        "Spatial_RMSE": round(gfs_rmse, 4),
        "Spatial_RMSE_Std": 0.0,
        "MAE_Reduction_Pct": 0.0,
        "RMSE_Reduction_Pct": 0.0
    })

    for name, m_type, feats in models_to_test:
        print(f"Running 5-Fold LLO Spatial CV for: {name} (Features: {len(feats)})...")
        res = evaluate_spatial_cv(df, m_type, feats)
        mae_reduct = ((gfs_mae - res["mae_mean"]) / gfs_mae) * 100.0
        rmse_reduct = ((gfs_rmse - res["rmse_mean"]) / gfs_rmse) * 100.0

        records.append({
            "Model": name,
            "Spatial_MAE": round(res["mae_mean"], 4),
            "Spatial_MAE_Std": round(res["mae_std"], 4),
            "Spatial_RMSE": round(res["rmse_mean"], 4),
            "Spatial_RMSE_Std": round(res["rmse_std"], 4),
            "MAE_Reduction_Pct": round(mae_reduct, 2),
            "RMSE_Reduction_Pct": round(rmse_reduct, 2)
        })
        print(f"  --> MAE: {res['mae_mean']:.4f} ± {res['mae_std']:.4f} °C (-{mae_reduct:.2f}%)")

    summary_df = pd.DataFrame(records)
    out_path = METRICS_DIR / "new_architecture_benchmark.csv"
    summary_df.to_csv(out_path, index=False)
    print("\n" + "=" * 75)
    print(f"Benchmark completed! Results saved to: {out_path}")
    print(summary_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
