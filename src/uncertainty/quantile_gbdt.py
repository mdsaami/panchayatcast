"""Quantile Gradient Boosting for direct probabilistic forecast interval estimation."""

import numpy as np
import pandas as pd
from typing import List, Tuple
import lightgbm as lgb


class QuantileGBDTPredictor:
    """Trains separate GBDTs for P10, P50, and P90 quantiles to provide an asymmetric prediction interval."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.feature_cols: List[str] = []

        self.model_p10 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
        self.model_p50 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.50,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
        self.model_p90 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y_true: np.ndarray, feature_cols: List[str]):
        """Fit residual quantiles on delta_T."""
        self.feature_cols = feature_cols
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        delta_T = y_true - gfs_temp

        self.model_p10.fit(X[feature_cols], delta_T)
        self.model_p50.fit(X[feature_cols], delta_T)
        self.model_p90.fit(X[feature_cols], delta_T)

    def predict_quantiles(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower_p10, median_p50, upper_p90) refined temperature forecasts."""
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        feat_df = X[self.feature_cols]

        d10 = self.model_p10.predict(feat_df)
        d50 = self.model_p50.predict(feat_df)
        d90 = self.model_p90.predict(feat_df)

        return gfs_temp + d10, gfs_temp + d50, gfs_temp + d90
