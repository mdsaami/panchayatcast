"""Baseline forecasting models: Raw GFS, Mean Bias Correction, and Ridge Regression."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute standard meteorological verification metrics: MAE, RMSE, Mean Bias, R2, Skill Score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "mae": float(round(mae, 4)),
        "rmse": float(round(rmse, 4)),
        "bias": float(round(bias, 4)),
        "r2": float(round(r2, 4))
    }

    if baseline_pred is not None:
        base_mae = mean_absolute_error(y_true, baseline_pred)
        base_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
        mae_reduct = ((base_mae - mae) / base_mae) * 100.0
        rmse_reduct = ((base_rmse - rmse) / base_rmse) * 100.0
        metrics["mae_reduction_pct"] = float(round(mae_reduct, 2))
        metrics["rmse_reduction_pct"] = float(round(rmse_reduct, 2))

    return metrics


class RawGFSBaseline:
    """Zero-correction GFS forecast baseline."""
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["temperature_2m_gfs"].to_numpy()


class MeanBiasCorrector:
    """Mean historical forecast residual offset corrector."""
    def __init__(self):
        self.mean_bias: float = 0.0

    def fit(self, X: pd.DataFrame, y_true: np.ndarray):
        gfs = X["temperature_2m_gfs"].to_numpy()
        residuals = y_true - gfs
        self.mean_bias = float(np.mean(residuals))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        gfs = X["temperature_2m_gfs"].to_numpy()
        return gfs + self.mean_bias


class RidgeResidualModel:
    """Ridge regularized linear regression on forecast residual delta_T."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y_true: np.ndarray, feature_cols: list[str]):
        self.feature_cols = feature_cols
        gfs = X["temperature_2m_gfs"].to_numpy()
        delta_T = y_true - gfs
        self.model.fit(X[feature_cols], delta_T)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        gfs = X["temperature_2m_gfs"].to_numpy()
        delta_T_pred = self.model.predict(X[self.feature_cols])
        return gfs + delta_T_pred
