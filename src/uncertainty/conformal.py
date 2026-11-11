"""Conformal prediction module for mathematically guaranteed prediction intervals."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class ConformalPredictor:
    """Split-conformal prediction intervals with empirical coverage guarantee.
    
    Given calibration residuals R_i = |y_i - y_hat_i|, computes quantile q_(1-alpha).
    Prediction interval: [y_hat - q, y_hat + q].
    """
    def __init__(self, coverage: float = 0.90):
        self.coverage = coverage
        self.alpha = 1.0 - coverage
        self.q_val: float = 1.0

    def calibrate(self, y_calib_true: np.ndarray, y_calib_pred: np.ndarray):
        """Compute conformal calibration quantile threshold."""
        residuals = np.abs(y_calib_true - y_calib_pred)
        n = len(residuals)
        # Finite-sample correction quantile level: ceil((n+1)*(1-alpha)) / n
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.q_val = float(np.quantile(residuals, q_level))

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower_bound, point_estimate, upper_bound)."""
        lower = y_pred - self.q_val
        upper = y_pred + self.q_val
        return lower, y_pred, upper

    def evaluate_coverage(self, y_test_true: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Compute empirical coverage and mean interval width on test set."""
        lower, _, upper = self.predict_interval(y_test_pred)
        covered = (y_test_true >= lower) & (y_test_true <= upper)
        empirical_cov = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(upper - lower))
        
        return {
            "target_coverage_pct": self.coverage * 100.0,
            "empirical_coverage_pct": round(empirical_cov, 2),
            "mean_interval_width_deg": round(mean_width, 4),
            "margin_error_q_deg": round(self.q_val, 4)
        }

    def categorize_confidence(self, interval_width: float) -> str:
        """Categorize forecast confidence based on interval width."""
        if interval_width <= 1.5:
            return "High Confidence"
        elif interval_width <= 2.5:
            return "Moderate Confidence"
        else:
            return "Low Confidence / High Microclimate Volatility"
