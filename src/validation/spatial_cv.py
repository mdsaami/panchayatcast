"""5-Fold Grouped Spatial Cross-Validation by location.

Ensures strict zero-spatial-leakage:
- 12 training locations (8,928 rows)
- 3 unseen test locations (2,232 rows) per fold
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.residual_model import DownscalingResidualModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "locations.json"


def get_spatial_folds() -> Dict[int, Tuple[List[str], List[str]]]:
    """Return dictionary of fold_num -> (train_locs, test_locs)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    all_locs = [loc["id"] for loc in cfg["locations"]]
    folds = {}
    for fold_str, test_locs in cfg["folds"].items():
        fold_idx = int(fold_str)
        train_locs = [loc for loc in all_locs if loc not in test_locs]
        folds[fold_idx] = (train_locs, test_locs)
    return folds


def run_spatial_cv_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "lightgbm",
    model_name: str = "CustomModel"
) -> Dict[str, Any]:
    """Execute 5-fold spatial cross-validation and return fold-wise and summary metrics."""
    folds = get_spatial_folds()
    
    fold_records = []
    all_test_preds = []
    
    for fold_idx in range(1, 6):
        train_locs, test_locs = folds[fold_idx]
        
        train_df = df[df["location_id"].isin(train_locs)].copy()
        test_df = df[df["location_id"].isin(test_locs)].copy()

        y_train = train_df["temperature_2m_era5"].to_numpy()
        y_test = test_df["temperature_2m_era5"].to_numpy()
        gfs_test = test_df["temperature_2m_gfs"].to_numpy()

        # Fit model
        model = DownscalingResidualModel(model_type=model_type)
        model.fit(train_df, y_train, feature_cols)
        
        y_pred = model.predict(test_df)

        # Baseline metrics
        gfs_mae = mean_absolute_error(y_test, gfs_test)
        gfs_rmse = np.sqrt(mean_squared_error(y_test, gfs_test))
        gfs_bias = np.mean(gfs_test - y_test)

        # Refined metrics
        ref_mae = mean_absolute_error(y_test, y_pred)
        ref_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ref_bias = np.mean(y_pred - y_test)
        ref_r2 = r2_score(y_test, y_pred)

        mae_reduct = ((gfs_mae - ref_mae) / gfs_mae) * 100.0
        rmse_reduct = ((gfs_rmse - ref_rmse) / gfs_rmse) * 100.0

        fold_records.append({
            "fold": fold_idx,
            "test_locs": ",".join(test_locs),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "gfs_mae": round(gfs_mae, 4),
            "gfs_rmse": round(gfs_rmse, 4),
            "gfs_bias": round(gfs_bias, 4),
            "ref_mae": round(ref_mae, 4),
            "ref_rmse": round(ref_rmse, 4),
            "ref_bias": round(ref_bias, 4),
            "ref_r2": round(ref_r2, 4),
            "mae_reduction_pct": round(mae_reduct, 2),
            "rmse_reduction_pct": round(rmse_reduct, 2)
        })

        test_copy = test_df[["time", "location_id", "temperature_2m_gfs", "temperature_2m_era5"]].copy()
        test_copy["temp_refined"] = y_pred
        test_copy["fold"] = fold_idx
        all_test_preds.append(test_copy)

    fold_df = pd.DataFrame(fold_records)
    preds_df = pd.concat(all_test_preds, ignore_index=True)

    summary = {
        "model_name": model_name,
        "features": feature_cols,
        "mean_gfs_mae": round(float(fold_df["gfs_mae"].mean()), 4),
        "std_gfs_mae": round(float(fold_df["gfs_mae"].std()), 4),
        "mean_gfs_rmse": round(float(fold_df["gfs_rmse"].mean()), 4),
        "std_gfs_rmse": round(float(fold_df["gfs_rmse"].std()), 4),
        "mean_ref_mae": round(float(fold_df["ref_mae"].mean()), 4),
        "std_ref_mae": round(float(fold_df["ref_mae"].std()), 4),
        "mean_ref_rmse": round(float(fold_df["ref_rmse"].mean()), 4),
        "std_ref_rmse": round(float(fold_df["ref_rmse"].std()), 4),
        "mean_mae_reduction_pct": round(float(fold_df["mae_reduction_pct"].mean()), 2),
        "std_mae_reduction_pct": round(float(fold_df["mae_reduction_pct"].std()), 2),
        "mean_rmse_reduction_pct": round(float(fold_df["rmse_reduction_pct"].mean()), 2),
        "std_rmse_reduction_pct": round(float(fold_df["rmse_reduction_pct"].std()), 2),
    }

    return {
        "summary": summary,
        "fold_df": fold_df,
        "predictions_df": preds_df
    }
