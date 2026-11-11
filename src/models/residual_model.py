"""ML residual downscaling model: T_refined = T_baseline + delta_T_ML."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb


class DownscalingResidualModel:
    """Residual learner that estimates forecast error delta_T = T_obs - T_gfs.
    
    Prediction: T_refined = T_gfs + delta_T_predicted.
    """
    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.feature_cols: List[str] = []

        if model_type == "lightgbm":
            self.estimator = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )
        elif model_type == "random_forest":
            self.estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth + 4,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "hist_gbdt":
            self.estimator = HistGradientBoostingRegressor(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _get_baseline_temperature(self, X: pd.DataFrame) -> np.ndarray:
        """Resolve the canonical or legacy baseline temperature column.

        Canonical PanchayatCast data uses ``baseline_temperature``.
        Legacy prototype data uses ``temperature_2m_gfs``.
        Canonical naming takes precedence when both are present.
        """
        if "baseline_temperature" in X.columns:
            return X["baseline_temperature"].to_numpy()

        if "temperature_2m_gfs" in X.columns:
            return X["temperature_2m_gfs"].to_numpy()

        raise KeyError(
            "Missing baseline temperature. Expected either "
            "'baseline_temperature' or legacy 'temperature_2m_gfs'."
        )    

    def fit(self, X: pd.DataFrame, y_true: np.ndarray, feature_cols: List[str]):
        """Fit residual estimator on delta_T = y_true - T_gfs."""
        self.feature_cols = feature_cols
        baseline_temp = self._get_baseline_temperature(X)
        delta_T = y_true - baseline_temp
        self.estimator.fit(X[feature_cols], delta_T)

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """Predict only the residual delta_T."""
        return self.estimator.predict(X[self.feature_cols])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict refined temperature: T_refined = T_gfs + delta_T."""
        baseline_temp = self._get_baseline_temperature(X)
        delta_T_pred = self.predict_residual(X)
        return baseline_temp + delta_T_pred

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance table."""
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
            # Normalize to sum to 1.0 or percentage
            norm_imp = importances / max(1e-9, np.sum(importances))
            df = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": importances,
                "relative_importance_pct": norm_imp * 100.0
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df
        else:
            return pd.DataFrame({"feature": self.feature_cols, "importance": [0.0]*len(self.feature_cols)})

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "feature_cols": self.feature_cols, "type": self.model_type}, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "DownscalingResidualModel":
        data = joblib.load(file_path)
        instance = cls(model_type=data["type"])
        instance.estimator = data["estimator"]
        instance.feature_cols = data["feature_cols"]
        return instance
