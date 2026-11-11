"""Multi-Algorithm Benchmark Comparison Script.

Compares:
1. Raw GFS
2. Mean Bias Corrector
3. Ridge Linear Regression
4. Random Forest
5. HistGradientBoosting
6. LightGBM

Across Spatial, Temporal, and Spatiotemporal validation schemes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_unified_dataset
from src.features.pipeline import FEATURE_SETS
from src.models.baseline import RawGFSBaseline, MeanBiasCorrector, RidgeResidualModel
from src.models.residual_model import DownscalingResidualModel
from src.validation.spatial_cv import get_spatial_folds

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_algorithm(df: pd.DataFrame, algo_name: str, feature_cols: list[str]) -> dict:
    folds = get_spatial_folds()
    spatial_maes = []
    spatial_rmses = []

    # 1. 5-Fold Spatial CV
    for fold_idx in range(1, 6):
        train_locs, test_locs = folds[fold_idx]
        train_df = df[df["location_id"].isin(train_locs)].copy()
        test_df = df[df["location_id"].isin(test_locs)].copy()

        y_train = train_df["temperature_2m_era5"].to_numpy()
        y_test = test_df["temperature_2m_era5"].to_numpy()

        if algo_name == "Raw GFS":
            model = RawGFSBaseline()
            model.fit(train_df, y_train)
            pred = model.predict(test_df)
        elif algo_name == "Mean Bias Correction":
            model = MeanBiasCorrector()
            model.fit(train_df, y_train)
            pred = model.predict(test_df)
        elif algo_name == "Ridge Regression":
            model = RidgeResidualModel(alpha=10.0)
            model.fit(train_df, y_train, feature_cols)
            pred = model.predict(test_df)
        elif algo_name == "Random Forest":
            model = DownscalingResidualModel(model_type="random_forest", n_estimators=60, max_depth=8)
            model.fit(train_df, y_train, feature_cols)
            pred = model.predict(test_df)
        elif algo_name == "HistGradientBoosting":
            model = DownscalingResidualModel(model_type="hist_gbdt", n_estimators=100)
            model.fit(train_df, y_train, feature_cols)
            pred = model.predict(test_df)
        elif algo_name == "LightGBM":
            model = DownscalingResidualModel(model_type="lightgbm", n_estimators=120, max_depth=6)
            model.fit(train_df, y_train, feature_cols)
            pred = model.predict(test_df)
        else:
            raise ValueError(algo_name)

        spatial_maes.append(mean_absolute_error(y_test, pred))
        spatial_rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

    # 2. Temporal Holdout (Aug 1-21 train, Aug 22-31 test)
    train_temp = df[df["time"] < "2024-08-22T00:00:00"].copy()
    test_temp = df[df["time"] >= "2024-08-22T00:00:00"].copy()
    y_tr_temp = train_temp["temperature_2m_era5"].to_numpy()
    y_te_temp = test_temp["temperature_2m_era5"].to_numpy()

    if algo_name == "Raw GFS":
        t_pred = test_temp["temperature_2m_gfs"].to_numpy()
    elif algo_name == "Mean Bias Correction":
        mb = MeanBiasCorrector()
        mb.fit(train_temp, y_tr_temp)
        t_pred = mb.predict(test_temp)
    elif algo_name == "Ridge Regression":
        rg = RidgeResidualModel(alpha=10.0)
        rg.fit(train_temp, y_tr_temp, feature_cols)
        t_pred = rg.predict(test_temp)
    elif algo_name == "Random Forest":
        rf = DownscalingResidualModel(model_type="random_forest", n_estimators=60, max_depth=8)
        rf.fit(train_temp, y_tr_temp, feature_cols)
        t_pred = rf.predict(test_temp)
    elif algo_name == "HistGradientBoosting":
        hgb = DownscalingResidualModel(model_type="hist_gbdt", n_estimators=100)
        hgb.fit(train_temp, y_tr_temp, feature_cols)
        t_pred = hgb.predict(test_temp)
    elif algo_name == "LightGBM":
        lgb_m = DownscalingResidualModel(model_type="lightgbm", n_estimators=120, max_depth=6)
        lgb_m.fit(train_temp, y_tr_temp, feature_cols)
        t_pred = lgb_m.predict(test_temp)

    temp_mae = mean_absolute_error(y_te_temp, t_pred)
    temp_rmse = np.sqrt(mean_squared_error(y_te_temp, t_pred))

    # 3. Spatiotemporal Holdout (mean across folds)
    sptemp_maes = []
    sptemp_rmses = []
    for fold_idx in range(1, 6):
        train_locs, test_locs = folds[fold_idx]
        tr_df = df[(df["location_id"].isin(train_locs)) & (df["time"] < "2024-08-22T00:00:00")].copy()
        te_df = df[(df["location_id"].isin(test_locs)) & (df["time"] >= "2024-08-22T00:00:00")].copy()
        y_tr = tr_df["temperature_2m_era5"].to_numpy()
        y_te = te_df["temperature_2m_era5"].to_numpy()

        if algo_name == "Raw GFS":
            st_pred = te_df["temperature_2m_gfs"].to_numpy()
        elif algo_name == "Mean Bias Correction":
            m = MeanBiasCorrector()
            m.fit(tr_df, y_tr)
            st_pred = m.predict(te_df)
        elif algo_name == "Ridge Regression":
            m = RidgeResidualModel(alpha=10.0)
            m.fit(tr_df, y_tr, feature_cols)
            st_pred = m.predict(te_df)
        elif algo_name == "Random Forest":
            m = DownscalingResidualModel(model_type="random_forest", n_estimators=60, max_depth=8)
            m.fit(tr_df, y_tr, feature_cols)
            st_pred = m.predict(te_df)
        elif algo_name == "HistGradientBoosting":
            m = DownscalingResidualModel(model_type="hist_gbdt", n_estimators=100)
            m.fit(tr_df, y_tr, feature_cols)
            st_pred = m.predict(te_df)
        elif algo_name == "LightGBM":
            m = DownscalingResidualModel(model_type="lightgbm", n_estimators=120, max_depth=6)
            m.fit(tr_df, y_tr, feature_cols)
            st_pred = m.predict(te_df)

        sptemp_maes.append(mean_absolute_error(y_te, st_pred))
        sptemp_rmses.append(np.sqrt(mean_squared_error(y_te, st_pred)))

    return {
        "Algorithm": algo_name,
        "Spatial_MAE_Mean": round(float(np.mean(spatial_maes)), 4),
        "Spatial_MAE_Std": round(float(np.std(spatial_maes)), 4),
        "Spatial_RMSE_Mean": round(float(np.mean(spatial_rmses)), 4),
        "Spatial_RMSE_Std": round(float(np.std(spatial_rmses)), 4),
        "Temporal_MAE": round(temp_mae, 4),
        "Temporal_RMSE": round(temp_rmse, 4),
        "Spatiotemporal_MAE_Mean": round(float(np.mean(sptemp_maes)), 4),
        "Spatiotemporal_RMSE_Mean": round(float(np.mean(sptemp_rmses)), 4)
    }


def main():
    print("=" * 80)
    print("STARTING MULTI-ALGORITHM BENCHMARK COMPARISON")
    print("=" * 80)

    df = build_unified_dataset(start_date="2024-08-01", end_date="2024-08-31")
    feature_cols = FEATURE_SETS["Model_5_Full_Physiographic_NDVI"]

    algorithms = [
        "Raw GFS",
        "Mean Bias Correction",
        "Ridge Regression",
        "Random Forest",
        "HistGradientBoosting",
        "LightGBM"
    ]

    records = []
    for algo in algorithms:
        print(f"Evaluating {algo}...")
        rec = evaluate_algorithm(df, algo, feature_cols)
        records.append(rec)

    res_df = pd.DataFrame(records)
    res_df.to_csv(METRICS_DIR / "algorithm_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("ALGORITHM BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
