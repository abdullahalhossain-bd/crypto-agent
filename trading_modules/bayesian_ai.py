"""
Bayesian AI Module — Uncertainty Estimation
=============================================

Goes beyond "80% confidence" to also say HOW UNCERTAIN the model is.

Two approaches:
  1. Bayesian Neural Network — MC Dropout (already in deep_learning.py)
  2. Conformal Prediction — distribution-free uncertainty bounds
  3. Bayesian Linear Regression — analytical posterior

Conformal prediction wraps ANY model (LSTM, XGBoost, LLM) and provides
calibrated prediction intervals with coverage guarantees.

Source: ml4t-3e (review #18) — conformal prediction + Bayesian methods

Usage:
    from trading_modules.bayesian_ai import ConformalPredictor, BayesianLinearRegression

    # Conformal: wrap any model
    cp = ConformalPredictor()
    cp.fit calibration_predictions, actual_outcomes)
    interval = cp.predict(point_prediction=0.03)
    # → (0.01, 0.05) = 90% confidence interval

    # Bayesian LR: analytical posterior
    blr = BayesianLinearRegression(n_features=26)
    blr.fit(X_train, y_train)
    pred_mean, pred_std = blr.predict(X_test)
    # → mean=0.02, std=0.015 (uncertainty estimate)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 1. Conformal Prediction
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConformalInterval:
    """Prediction interval from conformal prediction."""
    lower: float
    upper: float
    center: float
    width: float
    alpha: float  # Significance level (1 - coverage)

    def to_dict(self) -> dict:
        return {
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "center": round(self.center, 6),
            "width": round(self.width, 6),
            "coverage": f"{1-self.alpha:.0%}",
        }


class ConformalPredictor:
    """
    Distribution-free conformal prediction.

    Wraps ANY point predictor and produces calibrated prediction intervals
    with guaranteed coverage (e.g., 90% of true values fall in interval).

    Pipeline:
      1. Fit: collect (prediction, actual) pairs on calibration set
      2. Compute nonconformity scores: |prediction - actual|
      3. Predict: interval = point_pred ± quantile(scores, 1-alpha)

    This is model-agnostic — works with LSTM, XGBoost, LLM, any model.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (0.1 = 90% coverage, 0.05 = 95%)
        """
        self.alpha = alpha
        self.scores: np.ndarray = np.array([])
        self.is_fitted = False

    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit on calibration data.

        Args:
            predictions: Model predictions on calibration set
            actuals: True values on calibration set
        """
        self.scores = np.abs(predictions - actuals)
        self.is_fitted = True

    def predict(self, point_prediction: float) -> ConformalInterval:
        """
        Produce prediction interval for a new prediction.

        Args:
            point_prediction: Model's point estimate

        Returns:
            ConformalInterval with lower/upper bounds
        """
        if not self.is_fitted or len(self.scores) == 0:
            # Not fitted — return wide interval
            return ConformalInterval(
                lower=point_prediction - 0.1,
                upper=point_prediction + 0.1,
                center=point_prediction,
                width=0.2,
                alpha=self.alpha,
            )

        # Quantile of nonconformity scores
        q = np.quantile(self.scores, 1 - self.alpha)

        return ConformalInterval(
            lower=point_prediction - q,
            upper=point_prediction + q,
            center=point_prediction,
            width=2 * q,
            alpha=self.alpha,
        )

    def predict_batch(self, predictions: np.ndarray) -> list[ConformalInterval]:
        """Batch prediction."""
        return [self.predict(p) for p in predictions]


# ═══════════════════════════════════════════════════════════════
# 2. Bayesian Linear Regression
# ═══════════════════════════════════════════════════════════════

class BayesianLinearRegression:
    """
    Bayesian Linear Regression with analytical posterior.

    Unlike regular LR which gives point estimates, Bayesian LR gives
    a posterior distribution over weights, enabling uncertainty estimates.

    Model: y = w^T x + ε, where w ~ N(μ, Σ) and ε ~ N(0, σ²)

    Posterior update (conjugate prior):
      Σ_post = (Σ_prior^{-1} + X^T X / σ²)^{-1}
      μ_post = Σ_post (Σ_prior^{-1} μ_prior + X^T y / σ²)

    Prediction:
      mean = x^T μ_post
      var = x^T Σ_post x + σ²
    """

    def __init__(
        self,
        n_features: int = 26,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        noise_var: float = 0.01,
    ):
        """
        Args:
            n_features: Number of input features
            prior_mean: Prior mean for weights (0 = no prior knowledge)
            prior_var: Prior variance (higher = less informative prior)
            noise_var: Observation noise variance σ²
        """
        self.n_features = n_features
        self.noise_var = noise_var

        # Prior: w ~ N(μ_0, Σ_0)
        self.mu = np.full(n_features, prior_mean)
        self.Sigma = np.eye(n_features) * prior_var

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update posterior with training data.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) target values
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Posterior update
        Sigma_inv = np.linalg.inv(self.Sigma)
        XtX = X.T @ X

        self.Sigma = np.linalg.inv(Sigma_inv + XtX / self.noise_var)
        self.mu = self.Sigma @ (Sigma_inv @ self.mu + X.T @ y / self.noise_var)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Args:
            X: (n_samples, n_features) or (n_features,)

        Returns:
            (means, stds) — predicted mean and standard deviation
        """
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        means = X @ self.mu
        variances = np.array([x @ self.Sigma @ x + self.noise_var for x in X])
        stds = np.sqrt(variances)

        return means, stds

    def get_uncertainty_summary(self, X: np.ndarray) -> dict:
        """Get uncertainty statistics for a set of predictions."""
        means, stds = self.predict(X)
        return {
            "mean_prediction": float(np.mean(means)),
            "mean_uncertainty": float(np.mean(stds)),
            "max_uncertainty": float(np.max(stds)),
            "min_uncertainty": float(np.min(stds)),
            "high_uncertainty_pct": float(np.mean(stds > np.median(stds))),
        }


# ═══════════════════════════════════════════════════════════════
# 3. Uncertainty-Aware Decision
# ═══════════════════════════════════════════════════════════════

@dataclass
class UncertaintyAwareDecision:
    """Decision with explicit uncertainty quantification."""
    action: str  # "BUY" / "SELL" / "HOLD"
    expected_return: float
    confidence: float  # 0-1, how confident in the direction
    uncertainty: str  # "Low" / "Medium" / "High"
    interval_lower: float  # 90% CI lower bound
    interval_upper: float  # 90% CI upper bound
    risk_adjusted_size: float  # Position size factor (0-1, reduced by uncertainty)
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "expected_return": round(self.expected_return, 6),
            "confidence": round(self.confidence, 4),
            "uncertainty": self.uncertainty,
            "interval_90": [round(self.interval_lower, 6), round(self.interval_upper, 6)],
            "risk_adjusted_size": round(self.risk_adjusted_size, 4),
            "recommendation": self.recommendation,
        }


def make_uncertainty_aware_decision(
    prediction: float,
    lower: float,
    upper: float,
    confidence: float,
    base_position_size: float = 1.0,
) -> UncertaintyAwareDecision:
    """
    Convert a prediction + uncertainty into a trading decision.

    Logic:
      - If interval includes 0 → HOLD (direction uncertain)
      - Uncertainty level affects position sizing:
        Low → 100% size, Medium → 60%, High → 30%

    Args:
        prediction: Point prediction (expected return)
        lower: 90% CI lower bound
        upper: 90% CI upper bound
        confidence: Model confidence (0-1)
        base_position_size: Base position size to adjust

    Returns:
        UncertaintyAwareDecision
    """
    interval_width = upper - lower

    # Determine direction
    if lower > 0:
        action = "BUY"
    elif upper < 0:
        action = "SELL"
    else:
        action = "HOLD"  # Interval crosses zero — direction uncertain

    # Uncertainty level
    if interval_width < 0.02:
        uncertainty = "Low"
        size_factor = 1.0
    elif interval_width < 0.05:
        uncertainty = "Medium"
        size_factor = 0.6
    else:
        uncertainty = "High"
        size_factor = 0.3

    # If HOLD, no position
    if action == "HOLD":
        size_factor = 0.0

    # Reduce size further if confidence is low
    size_factor *= confidence

    if action == "HOLD":
        rec = f"HOLD — prediction interval [{lower:.4f}, {upper:.4f}] includes 0, direction uncertain"
    elif uncertainty == "High":
        rec = f"{action} with caution — high uncertainty (width={interval_width:.4f}), size reduced to {size_factor:.0%}"
    elif uncertainty == "Medium":
        rec = f"{action} — moderate uncertainty, size at {size_factor:.0%}"
    else:
        rec = f"{action} — low uncertainty, full size {size_factor:.0%}"

    return UncertaintyAwareDecision(
        action=action,
        expected_return=prediction,
        confidence=confidence,
        uncertainty=uncertainty,
        interval_lower=lower,
        interval_upper=upper,
        risk_adjusted_size=base_position_size * size_factor,
        recommendation=rec,
    )
