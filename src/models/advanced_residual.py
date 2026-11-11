"""Advanced literature-backed residual downscaling models for PanchayatCast.

Combines:
1. Multi-Quantile Loss (Pinball Loss at P10, P50, P90) - Cannon (2011), Rasp & Lerch (2018)
2. Stacking Ensemble of Diverse Learners (LightGBM + HistGBDT + Ridge Regularized)
3. Physical Residual Formulation: T_panchayat = T_baseline + delta_T_ML
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb


class MultiQuantileResidualModel:
    """Predicts median residual (P50) and calibrated probabilistic bounds (P10, P90).
    
    References:
    - Cannon, A. J. (2011): Quantile regression for downscaling
    - Rasp & Lerch (2018): Probabilistic post-processing of NWP
    """
    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42
    ):
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y_true: np.ndarray, feature_cols: List[str]):
        """Fit quantile regressors on forecast residuals: delta_T = y_true - T_gfs."""
        self.feature_cols = feature_cols
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        residuals = y_true - gfs_temp

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X[feature_cols], residuals)
            self.models[q] = model

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate point forecast and uncertainty bounds.
        
        Returns:
            Dict containing:
            - 't_pred': Median estimate (P50)
            - 'delta_pred': Predicted residual
            - 'p10': 10th percentile lower bound
            - 'p50': 50th percentile median
            - 'p90': 90th percentile upper bound
            - 'interval_width': p90 - p10
        """
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        quantile_preds = {}

        for q, model in self.models.items():
            delta_q = model.predict(X[self.feature_cols])
            quantile_preds[q] = gfs_temp + delta_q

        # Ensure monotonicity: P10 <= P50 <= P90
        p10 = quantile_preds[0.10]
        p50 = quantile_preds[0.50]
        p90 = quantile_preds[0.90]
        
        # Clip any quantile crossing
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        delta_pred = self.models[0.50].predict(X[self.feature_cols])

        return {
            "t_pred": p50,
            "delta_pred": delta_pred,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "interval_width": p90 - p10
        }


class StackedResidualEnsemble:
    """Multi-Algorithm Residual Ensemble combining:
    1. LightGBM (Gradient Boosted Trees - fast, robust)
    2. HistGradientBoostingRegressor (Exact split histogram gradient boosting)
    3. Ridge Regression (L2 regularized linear model for baseline stability)
    
    Meta-learner: Ridge regression blending weights on residual predictions.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.feature_cols: List[str] = []
        
        self.lgb = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
        self.hgb = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=random_state
        )
        self.ridge = Ridge(alpha=10.0)
        self.meta_learner = Ridge(alpha=1.0, positive=True)

    def fit(self, X: pd.DataFrame, y_true: np.ndarray, feature_cols: List[str]):
        self.feature_cols = feature_cols
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        residuals = y_true - gfs_temp

        X_feats = X[feature_cols].to_numpy()

        # Fit base models
        self.lgb.fit(X[feature_cols], residuals)
        self.hgb.fit(X_feats, residuals)
        self.ridge.fit(X_feats, residuals)

        # Generate base predictions for meta-blending
        pred_lgb = self.lgb.predict(X[feature_cols])
        pred_hgb = self.hgb.predict(X_feats)
        pred_ridge = self.ridge.predict(X_feats)

        stacked_features = np.column_stack([pred_lgb, pred_hgb, pred_ridge])
        self.meta_learner.fit(stacked_features, residuals)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (T_refined, delta_T_predicted)."""
        gfs_temp = X["temperature_2m_gfs"].to_numpy()
        X_feats = X[self.feature_cols].to_numpy()

        pred_lgb = self.lgb.predict(X[self.feature_cols])
        pred_hgb = self.hgb.predict(X_feats)
        pred_ridge = self.ridge.predict(X_feats)

        stacked_features = np.column_stack([pred_lgb, pred_hgb, pred_ridge])
        delta_pred = self.meta_learner.predict(stacked_features)
        
        t_refined = gfs_temp + delta_pred
        return t_refined, delta_pred
