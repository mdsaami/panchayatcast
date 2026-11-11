"""Temporal holdout validation module (train on earlier dates, test on future dates)."""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.residual_model import DownscalingResidualModel


def run_temporal_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    split_date: str = "2024-08-22T00:00:00",
    model_type: str = "lightgbm",
    model_name: str = "CustomModel"
) -> Dict[str, Any]:
    """Train on dates < split_date, test on dates >= split_date."""
    train_df = df[df["time"] < split_date].copy()
    test_df = df[df["time"] >= split_date].copy()

    y_train = train_df["temperature_2m_era5"].to_numpy()
    y_test = test_df["temperature_2m_era5"].to_numpy()
    gfs_test = test_df["temperature_2m_gfs"].to_numpy()

    model = DownscalingResidualModel(model_type=model_type)
    model.fit(train_df, y_train, feature_cols)
    y_pred = model.predict(test_df)

    gfs_mae = mean_absolute_error(y_test, gfs_test)
    gfs_rmse = np.sqrt(mean_squared_error(y_test, gfs_test))
    gfs_bias = float(np.mean(gfs_test - y_test))

    ref_mae = mean_absolute_error(y_test, y_pred)
    ref_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ref_bias = float(np.mean(y_pred - y_test))
    ref_r2 = float(r2_score(y_test, y_pred))

    mae_reduct = ((gfs_mae - ref_mae) / gfs_mae) * 100.0
    rmse_reduct = ((gfs_rmse - ref_rmse) / gfs_rmse) * 100.0

    return {
        "model_name": model_name,
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
        "rmse_reduction_pct": round(rmse_reduct, 2),
        "predictions": y_pred
    }
