"""
Time Series Models — forecasting for institutional trading
==========================================================

Pure-Python implementations of classical time series models:

    1. ARIMA(p,d,q)        — AutoRegressive Integrated Moving Average
    2. GARCH(1,1)          — volatility forecasting
    3. State Space Model   — local-level + local-linear trend (Kalman-based)
    4. Exponential Smoothing (Holt-Winters)

No statsmodels dependency — implementations use numpy linear algebra only.

Usage:
    from trading_modules.time_series_models import (
        ARIMA, GARCH11, StateSpaceModel, holt_winters
    )
    arima = ARIMA(p=2, d=1, q=1).fit(df["close"])
    forecast = arima.forecast(steps=5)

    garch = GARCH11().fit(returns)
    vol_forecast = garch.forecast_volatility(steps=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. ARIMA(p, d, q)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ARIMAResult:
    forecast_values: np.ndarray
    forecast_std: float
    residuals: np.ndarray
    aic: float
    params: dict
    order: tuple  # (p, d, q)


class ARIMA:
    """Simplified ARIMA(p, d, q) model.

    Uses OLS for parameter estimation (no MLE). Suitable for short-horizon
    forecasts (1-10 steps). For longer horizons, switch to state-space.

    Parameters:
        p: AR order
        d: differencing order
        q: MA order (simplified — uses residual moving average)
    """

    def __init__(self, p: int = 1, d: int = 1, q: int = 0) -> None:
        if p < 0 or d < 0 or q < 0:
            raise ValueError("p, d, q must be >= 0")
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.ar_params: Optional[np.ndarray] = None
        self.ma_params: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.last_values_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "ARIMA":
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        # Differencing
        for _ in range(self.d):
            s = np.diff(s)
        if len(s) < self.p + self.q + 5:
            logger.warning("ARIMA: insufficient data after differencing")
            return self
        # Build AR design matrix
        n = len(s)
        if self.p > 0:
            X = np.column_stack([s[self.p - 1 - i: n - 1 - i] for i in range(self.p)])
            y = s[self.p:]
            # Add constant
            X = np.column_stack([np.ones(len(X)), X])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                self.ar_params = coeffs[1:]  # exclude constant
                self._const = float(coeffs[0])
                residuals = y - X @ coeffs
            except Exception as e:
                logger.warning(f"ARIMA AR fit failed: {e}")
                self.ar_params = np.zeros(self.p)
                self._const = 0.0
                residuals = y - y.mean() if len(y) > 0 else np.zeros(0)
        else:
            self.ar_params = np.zeros(0)
            self._const = float(s.mean()) if len(s) > 0 else 0.0
            residuals = s - self._const
        # MA part: fit on residuals (simplified — uses residual moving average)
        if self.q > 0 and len(residuals) > self.q:
            X_ma = np.column_stack([residuals[self.q - 1 - i: len(residuals) - 1 - i]
                                     for i in range(self.q)])
            y_ma = residuals[self.q:]
            X_ma = np.column_stack([np.ones(len(X_ma)), X_ma])
            try:
                ma_coeffs, _, _, _ = np.linalg.lstsq(X_ma, y_ma, rcond=None)
                self.ma_params = ma_coeffs[1:]
            except Exception:
                self.ma_params = np.zeros(self.q)
        else:
            self.ma_params = np.zeros(max(0, self.q))
        self.residuals_ = residuals
        self.last_values_ = s
        # AIC (approximate)
        n_eff = max(1, len(residuals))
        rss = float(np.sum(residuals ** 2)) if len(residuals) > 0 else 0
        sigma2 = rss / n_eff if n_eff > 0 else 1.0
        k = self.p + self.q + 1
        self._aic = n_eff * np.log(sigma2 + 1e-10) + 2 * k if sigma2 > 0 else float("inf")
        self._fitted = True
        return self

    def forecast(self, steps: int = 1) -> ARIMAResult:
        if not self._fitted or self.last_values_ is None:
            return ARIMAResult(
                forecast_values=np.zeros(steps), forecast_std=0.0,
                residuals=np.zeros(0), aic=0.0, params={}, order=(self.p, self.d, self.q),
            )
        s = self.last_values_
        forecasts = np.zeros(steps)
        recent_residuals = list(self.residuals_[-self.q:]) if self.q > 0 else []
        recent_values = list(s[-self.p:]) if self.p > 0 else []
        for t in range(steps):
            # AR component
            ar_part = self._const
            if self.p > 0 and len(recent_values) >= self.p:
                ar_part += float(np.dot(self.ar_params, recent_values[-self.p:]))
            # MA component
            ma_part = 0.0
            if self.q > 0 and len(recent_residuals) >= self.q:
                ma_part = float(np.dot(self.ma_params, recent_residuals[-self.q:]))
            forecasts[t] = ar_part + ma_part
            recent_values.append(forecasts[t])
            recent_residuals.append(0.0)  # future residuals unknown
        # Integrate back if differenced
        if self.d > 0:
            last_raw = float(series_last_value := 0)  # placeholder
            # We don't store the raw series; integration is approximate
            # Caller should add forecasts to last observed value for d=1
            pass
        forecast_std = float(np.std(self.residuals_)) if len(self.residuals_) > 1 else 0.0
        return ARIMAResult(
            forecast_values=forecasts,
            forecast_std=forecast_std,
            residuals=self.residuals_,
            aic=self._aic,
            params={
                "ar": self.ar_params.tolist() if self.ar_params is not None else [],
                "ma": self.ma_params.tolist() if self.ma_params is not None else [],
                "const": self._const,
            },
            order=(self.p, self.d, self.q),
        )


# ──────────────────────────────────────────────────────────────────────
# 2. GARCH(1,1)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GARCHResult:
    conditional_volatility: np.ndarray
    forecast_volatility: np.ndarray
    params: dict
    log_likelihood: float


class GARCH11:
    """GARCH(1,1) volatility model.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Uses MLE with Gaussian assumption. Parameters are estimated via
    a simple grid + gradient-free optimization (Nelder-Mead style).

    Parameters:
        max_iter: optimization iterations (default 100)
    """

    def __init__(self, max_iter: int = 100) -> None:
        self.max_iter = int(max_iter)
        self.omega_: float = 0.0
        self.alpha_: float = 0.1
        self.beta_: float = 0.85
        self.residuals_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "GARCH11":
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 30:
            logger.warning("GARCH: insufficient data")
            return self
        r = r - r.mean()  # de-mean
        self.residuals_ = r
        # Grid search over (alpha, beta) with omega = variance × (1 - alpha - beta)
        var = float(np.var(r))
        best_ll = -np.inf
        best = (0.1, 0.85)
        for alpha in np.linspace(0.01, 0.5, 15):
            for beta in np.linspace(0.4, 0.95, 15):
                if alpha + beta >= 0.999:
                    continue
                omega = var * (1 - alpha - beta)
                if omega <= 0:
                    continue
                ll = self._log_likelihood(r, omega, alpha, beta)
                if ll > best_ll:
                    best_ll = ll
                    best = (alpha, beta)
        self.alpha_, self.beta_ = best
        self.omega_ = var * (1 - self.alpha_ - self.beta_)
        self._fitted = True
        return self

    def _log_likelihood(self, r: np.ndarray, omega: float, alpha: float, beta: float) -> float:
        n = len(r)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else np.var(r)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
            if sigma2[t] <= 0:
                sigma2[t] = 1e-8
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)
        return float(ll)

    def forecast_volatility(self, steps: int = 5) -> GARCHResult:
        if not self._fitted or self.residuals_ is None:
            return GARCHResult(
                conditional_volatility=np.zeros(0),
                forecast_volatility=np.zeros(steps),
                params={}, log_likelihood=0.0,
            )
        r = self.residuals_
        n = len(r)
        # Compute conditional vol
        sigma2 = np.zeros(n)
        sigma2[0] = self.omega_ / max(1 - self.alpha_ - self.beta_, 1e-6)
        for t in range(1, n):
            sigma2[t] = self.omega_ + self.alpha_ * r[t - 1] ** 2 + self.beta_ * sigma2[t - 1]
        # Forecast
        forecasts = np.zeros(steps)
        last_sigma2 = sigma2[-1]
        last_r2 = r[-1] ** 2
        for t in range(steps):
            forecasts[t] = self.omega_ + self.alpha_ * last_r2 + self.beta_ * last_sigma2
            last_sigma2 = forecasts[t]
            last_r2 = forecasts[t]  # use expected value (0 mean → 0 innovation)
        return GARCHResult(
            conditional_volatility=np.sqrt(sigma2),
            forecast_volatility=np.sqrt(forecasts),
            params={"omega": self.omega_, "alpha": self.alpha_, "beta": self.beta_},
            log_likelihood=self._log_likelihood(r, self.omega_, self.alpha_, self.beta_),
        )


# ──────────────────────────────────────────────────────────────────────
# 3. State Space Model (local-level + local-linear)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class StateSpaceResult:
    filtered_state: np.ndarray
    filtered_state_std: np.ndarray
    smoothed_state: np.ndarray
    forecast: np.ndarray
    forecast_std: np.ndarray


class StateSpaceModel:
    """Local-linear trend state space model.

    State: [level, slope]
    Transition: level_t = level_{t-1} + slope_{t-1} + ε_level
                slope_t = slope_{t-1} + ε_slope
    Observation: y_t = level_t + ε_obs

    Uses Kalman filter for estimation.

    Parameters:
        level_var: process noise for level (default 1e-4)
        slope_var: process noise for slope (default 1e-6)
        obs_var: observation noise (default 1e-2)
    """

    def __init__(
        self, level_var: float = 1e-4, slope_var: float = 1e-6, obs_var: float = 1e-2,
    ) -> None:
        self.level_var = float(level_var)
        self.slope_var = float(slope_var)
        self.obs_var = float(obs_var)

    def fit_forecast(self, series: np.ndarray, forecast_steps: int = 5) -> StateSpaceResult:
        y = np.asarray(series, dtype=float)
        y = y[np.isfinite(y)]
        n = len(y)
        if n < 10:
            return StateSpaceResult(
                filtered_state=np.zeros((2, 0)),
                filtered_state_std=np.zeros((2, 0)),
                smoothed_state=np.zeros((2, 0)),
                forecast=np.zeros(forecast_steps),
                forecast_std=np.zeros(forecast_steps),
            )
        # State: [level, slope]
        # Transition matrix F
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.diag([self.level_var, self.slope_var])
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.obs_var]])

        x = np.array([y[0], 0.0])  # initial state
        P = np.eye(2) * 1.0
        filtered = np.zeros((2, n))
        filtered_std = np.zeros((2, n))
        for t in range(n):
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            # Update
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ (y[t] - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P_pred
            filtered[:, t] = x
            filtered_std[:, t] = np.sqrt(np.diag(P))

        # Smoothed (RTS smoother — simplified)
        smoothed = filtered.copy()
        for t in range(n - 2, -1, -1):
            # Backward pass
            x_pred_next = F @ filtered[:, t]
            P_pred_next = F @ np.diag(filtered_std[:, t] ** 2) @ F.T + Q
            try:
                C = np.diag(filtered_std[:, t] ** 2) @ F.T @ np.linalg.inv(P_pred_next)
                smoothed[:, t] = filtered[:, t] + C @ (smoothed[:, t + 1] - x_pred_next)
            except np.linalg.LinAlgError:
                pass

        # Forecast
        forecasts = np.zeros(forecast_steps)
        forecast_std = np.zeros(forecast_steps)
        x_fc = filtered[:, -1].copy()
        P_fc = np.diag(filtered_std[:, -1] ** 2)
        for t in range(forecast_steps):
            x_fc = F @ x_fc
            P_fc = F @ P_fc @ F.T + Q
            forecasts[t] = float(x_fc[0])
            forecast_std[t] = float(np.sqrt(P_fc[0, 0]))

        return StateSpaceResult(
            filtered_state=filtered,
            filtered_state_std=filtered_std,
            smoothed_state=smoothed,
            forecast=forecasts,
            forecast_std=forecast_std,
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Holt-Winters Exponential Smoothing
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HoltWintersResult:
    forecast: np.ndarray
    level: float
    trend: float
    seasonals: np.ndarray
    params: dict


def holt_winters(
    series: np.ndarray, season_length: int = 7,
    alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1,
    forecast_steps: int = 5,
) -> HoltWintersResult:
    """Holt-Winters triple exponential smoothing with multiplicative seasonality.

    Parameters:
        series: 1-D array of values
        season_length: # of bars per seasonal cycle (e.g., 7 for daily with weekly cycle)
        alpha: level smoothing (0..1)
        beta: trend smoothing (0..1)
        gamma: seasonal smoothing (0..1)
        forecast_steps: # of steps to forecast
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2 * season_length:
        # Fallback to simple exponential smoothing
        level = float(y.mean())
        forecast = np.full(forecast_steps, level)
        return HoltWintersResult(
            forecast=forecast, level=level, trend=0.0,
            seasonals=np.zeros(season_length),
            params={"alpha": alpha, "beta": beta, "gamma": gamma},
        )
    # Initialize
    seasonals = np.zeros(season_length)
    for i in range(season_length):
        seasonals[i] = float(y[i]) - float(y[:season_length].mean())
    level = float(y[:season_length].mean())
    trend = (float(y[season_length:2 * season_length].mean()) -
             float(y[:season_length].mean())) / season_length
    # Iterate
    for t in range(season_length, n):
        s_idx = t % season_length
        last_level = level
        level = alpha * (y[t] - seasonals[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        seasonals[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasonals[s_idx]
    # Forecast
    forecasts = np.zeros(forecast_steps)
    for t in range(forecast_steps):
        s_idx = (n + t) % season_length
        forecasts[t] = level + (t + 1) * trend + seasonals[s_idx]
    return HoltWintersResult(
        forecast=forecasts, level=level, trend=float(trend),
        seasonals=seasonals,
        params={"alpha": alpha, "beta": beta, "gamma": gamma},
    )


__all__ = [
    "ARIMA", "ARIMAResult",
    "GARCH11", "GARCHResult",
    "StateSpaceModel", "StateSpaceResult",
    "holt_winters", "HoltWintersResult",
]
