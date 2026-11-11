"""Historical 15-Location Benchmark Reproduction Script.

Reproduces and audits the historical August 2024 15-location experiment:
- GFS baseline metrics across 11,160 rows
- 5-Fold GroupKFold Spatial CV for Model A (Weather + Time)
- 5-Fold GroupKFold Spatial CV for Model B1 (Weather + Time + Point Elevation)
- Demonstrates the 'Elevation Paradox' where scalar elevation worsens performance in 4/5 folds
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_unified_dataset
from src.features.pipeline import FEATURE_SETS
from src.validation.spatial_cv import run_spatial_cv_experiment

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1: Ingesting / Loading Unified Dataset (15 Locations, Aug 2024)")
    print("=" * 70)
    df = build_unified_dataset(start_date="2024-08-01", end_date="2024-08-31")
    print(f"Dataset shape: {df.shape} (Expected: 11,160 rows, 15 locations x 744 hours)")
    assert len(df) == 11160, f"Expected 11160 rows, found {len(df)}"
    assert df.isnull().sum().sum() == 0, "Found missing values in dataset!"

    y_true = df["temperature_2m_era5"].to_numpy()
    gfs_temp = df["temperature_2m_gfs"].to_numpy()

    # Baseline GFS verification
    gfs_mae = mean_absolute_error(y_true, gfs_temp)
    gfs_rmse = np.sqrt(mean_squared_error(y_true, gfs_temp))
    gfs_bias = float(np.mean(gfs_temp - y_true))

    print("\n" + "=" * 70)
    print(f"GFS RAW BASELINE METRICS (N = {len(df)}):")
    print(f"  MAE  = {gfs_mae:.4f} °C (Historical Target: ~1.6754 °C)")
    print(f"  RMSE = {gfs_rmse:.4f} °C (Historical Target: ~2.1482 °C)")
    print(f"  Bias = {gfs_bias:+.4f} °C (Historical Target: ~+1.0987 °C)")
    print("=" * 70)

    # 1. Evaluate Model A: Weather + Cyclical Time
    print("\nEvaluating Model A (Weather + Time) across 5-Fold Spatial CV...")
    res_a = run_spatial_cv_experiment(
        df,
        feature_cols=FEATURE_SETS["Model_1_Weather_Time"],
        model_type="lightgbm",
        model_name="Model_A_Weather_Time"
    )

    # 2. Evaluate Model B1: Weather + Time + Point Elevation
    print("Evaluating Model B1 (Weather + Time + Point Elevation) across 5-Fold Spatial CV...")
    res_b1 = run_spatial_cv_experiment(
        df,
        feature_cols=FEATURE_SETS["Model_B1_Weather_Time_Elevation"],
        model_type="lightgbm",
        model_name="Model_B1_Weather_Time_Elevation"
    )

    # Comparison Table
    print("\n" + "=" * 70)
    print("FOLD-BY-FOLD COMPARISON: MODEL A vs MODEL B1")
    print("=" * 70)
    comp_records = []
    for f in range(1, 6):
        a_row = res_a["fold_df"].loc[res_a["fold_df"]["fold"] == f].iloc[0]
        b1_row = res_b1["fold_df"].loc[res_b1["fold_df"]["fold"] == f].iloc[0]
        delta_mae = b1_row["ref_mae"] - a_row["ref_mae"]
        verdict = "Worsened" if delta_mae > 0 else "Improved"

        comp_records.append({
            "Fold": f,
            "Test Locations": a_row["test_locs"],
            "GFS MAE (°C)": a_row["gfs_mae"],
            "Model A MAE (°C)": a_row["ref_mae"],
            "Model B1 MAE (°C)": b1_row["ref_mae"],
            "Delta (B1 - A) (°C)": round(delta_mae, 4),
            "Verdict": verdict
        })
        print(f"Fold {f} ({a_row['test_locs']}): GFS={a_row['gfs_mae']:.4f} | Model A={a_row['ref_mae']:.4f} | Model B1={b1_row['ref_mae']:.4f} | Delta={delta_mae:+.4f} ({verdict})")

    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(METRICS_DIR / "historical_reproduction_folds.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON ACROSS SPATIAL FOLDS (Mean ± Std):")
    print(f"  GFS MAE:      {res_a['summary']['mean_gfs_mae']:.4f} ± {res_a['summary']['std_gfs_mae']:.4f} °C")
    print(f"  Model A MAE:  {res_a['summary']['mean_ref_mae']:.4f} ± {res_a['summary']['std_ref_mae']:.4f} °C (Reduction: {res_a['summary']['mean_mae_reduction_pct']:.2f}%)")
    print(f"  Model B1 MAE: {res_b1['summary']['mean_ref_mae']:.4f} ± {res_b1['summary']['std_ref_mae']:.4f} °C (Reduction: {res_b1['summary']['mean_mae_reduction_pct']:.2f}%)")
    elev_diff = res_b1['summary']['mean_ref_mae'] - res_a['summary']['mean_ref_mae']
    print(f"  Elevation Effect (B1 - A): {elev_diff:+.4f} °C")
    print("=" * 70)


if __name__ == "__main__":
    main()
