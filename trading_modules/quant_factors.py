"""
Quantitative Factors Library
============================

Pure-Python implementations of quantitative finance factors used by
institutional desks:

    1. Z-Score           — current price vs rolling mean/std
    2. Hurst Exponent    — trending (<0.5) vs mean-reverting (>0.5) vs random (=0.5)
    3. Cointegration     — two series that share a long-run equilibrium
    4. PCA               — reduce multi-asset returns to principal components
    5. Kalman Filter     — adaptive trend following with noise filtering
    6. Hidden Markov     — regime detection via state transitions
    7. Bayesian Update   — posterior win-rate update from new trade outcomes

All functions take numpy/pandas inputs and return floats or dataclasses.
No external dependencies beyond numpy/pandas.

Usage:
    from trading_modules.quant_factors import (
        zscore, hurst_exponent, kalman_filter, hmm_regime, bayesian_winrate
    )
    z = zscore(df["close"], window=50)
    h = hurst_exponent(df["close"].to_numpy())
    kf = kalman_filter(df["close"].to_numpy())
    regime = hmm_regime(df["close"].pct_change().dropna().to_numpy(), n_states=3)
    post = bayesian_winrate(prior_alpha=10, prior_beta=10, wins=15, losses=8)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Z-Score
# ──────────────────────────────────────────────────────────────────────
def zscore(series: pd.Series, window: int = 50) -> float:
    """Return the latest z-score of `series` over `window` bars.

    z = (x - mean) / std
    Values > 2 = overbought, < -2 = oversold (mean-reversion context).
    """
    if series is None or len(series) < window:
        return 0.0
    recent = series.tail(window).astype(float)
    mu = float(recent.mean())
    sd = float(recent.std(ddof=0))
    if sd <= 0:
        return 0.0
    return float((float(series.iloc[-1]) - mu) / sd)


def zscore_series(series: pd.Series, window: int = 50) -> pd.Series:
    """Rolling z-score for an entire series."""
    if series is None or len(series) < window:
        return pd.Series(dtype=float)
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


# ──────────────────────────────────────────────────────────────────────
# 2. Hurst Exponent
# ──────────────────────────────────────────────────────────────────────
def hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
    """Estimate the Hurst Exponent using R/S analysis.

    Interpretation:
        H < 0.5  → mean-reverting (anti-persistent)
        H ≈ 0.5  → random walk
        H > 0.5  → trending (persistent)
    """
    if prices is None or len(prices) < max_lag * 2:
        return 0.5  # default to random walk
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < max_lag * 2:
        return 0.5

    lags = range(2, max_lag + 1)
    rs_values: list[float] = []
    for lag in lags:
        chunks = [prices[i:i + lag] for i in range(0, len(prices) - lag + 1, lag)]
        if not chunks:
            continue
        rs_list: list[float] = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            deviation = np.cumsum(chunk - mean)
            r = float(deviation.max() - deviation.min())
            s = float(chunk.std(ddof=1))
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(float(np.mean(rs_list)))
    if len(rs_values) < 3:
        return 0.5
    # Fit log(R/S) = H * log(lag) + c
    log_lags = np.log(np.array(list(lags)[:len(rs_values)]))
    log_rs = np.log(np.array(rs_values))
    try:
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        # Clamp to plausible range
        return float(max(0.0, min(1.0, slope)))
    except Exception:
        return 0.5


# ──────────────────────────────────────────────────────────────────────
# 3. Cointegration (simplified Engle-Granger)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CointegrationResult:
    is_cointegrated: bool
    hedge_ratio: float          # β in y = α + β·x + ε
    half_life: float            # half-life of mean reversion (bars)
    spread_zscore: float        # current z-score of the residual spread
    p_value_approx: float       # approximate p-value (ADF-style heuristic)

    def to_dict(self) -> dict:
        return {
            "is_cointegrated": self.is_cointegrated,
            "hedge_ratio": round(self.hedge_ratio, 4),
            "half_life": round(self.half_life, 2),
            "spread_zscore": round(self.spread_zscore, 2),
            "p_value_approx": round(self.p_value_approx, 4),
        }


def cointegration_test(y: np.ndarray, x: np.ndarray) -> CointegrationResult:
    """Simplified Engle-Granger cointegration test.

    1. Regress y on x: y = α + β·x + ε
    2. Test residuals for stationarity (ADF-style)
    3. Compute half-life of mean reversion from Ornstein-Uhlenbeck

    Returns a CointegrationResult. Use this to find pairs to trade
    long-short (statistical arbitrage).
    """
    if y is None or x is None:
        return CointegrationResult(False, 0, 0, 0, 1.0)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(len(y), len(x))
    if n < 30:
        return CointegrationResult(False, 0, 0, 0, 1.0)
    y = y[:n]; x = x[:n]

    # OLS regression: y = α + β·x
    X = np.column_stack([np.ones(n), x])
    try:
        beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        # alpha, beta = coeffs
        hedge_ratio = float(beta)
    except Exception:
        return CointegrationResult(False, 0, 0, 0, 1.0)

    residuals = y - (X @ np.array([np.linalg.lstsq(X, y, rcond=None)[0][0], beta]))

    # ADF heuristic: regress Δresidual on residual_lagged
    # If coefficient is significantly negative → stationary
    try:
        resid_lag = residuals[:-1]
        resid_diff = np.diff(residuals)
        # Δresid = φ · resid_lagged + ε
        X2 = np.column_stack([np.ones(len(resid_lag)), resid_lag])
        coeffs = np.linalg.lstsq(X2, resid_diff, rcond=None)[0]
        phi = float(coeffs[1])
        # Approximate p-value: use a simple threshold on phi
        # φ < -0.1 → likely stationary, p < 0.05
        # φ > -0.02 → likely non-stationary, p > 0.5
        if phi < -0.2:
            p_value = 0.01
        elif phi < -0.1:
            p_value = 0.05
        elif phi < -0.05:
            p_value = 0.15
        elif phi < -0.02:
            p_value = 0.4
        else:
            p_value = 0.8
        is_coint = p_value < 0.05

        # Half-life: -ln(2) / ln(1 + φ)  (only valid when φ < 0)
        if phi < 0 and phi > -1:
            half_life = -np.log(2) / np.log(1 + phi)
            half_life = max(1.0, min(500.0, float(half_life)))
        else:
            half_life = 999.0

        # Spread z-score
        spread_mean = float(residuals.mean())
        spread_std = float(residuals.std(ddof=0))
        if spread_std > 0:
            spread_z = float((residuals[-1] - spread_mean) / spread_std)
        else:
            spread_z = 0.0
    except Exception as e:
        logger.warning(f"Cointegration test failed: {e}")
        return CointegrationResult(False, hedge_ratio, 999.0, 0.0, 1.0)

    return CointegrationResult(
        is_cointegrated=bool(is_coint),
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        spread_zscore=spread_z,
        p_value_approx=float(p_value),
    )


# ──────────────────────────────────────────────────────────────────────
# 4. PCA (Principal Component Analysis)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PCAResult:
    components: np.ndarray          # (n_components, n_features)
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    n_components: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data onto the principal components."""
        return (X - self.mean) @ self.components.T

    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in self.explained_variance_ratio],
            "mean": [round(float(m), 4) for m in self.mean],
        }


def pca(returns: pd.DataFrame, n_components: int = 2) -> PCAResult:
    """Compute PCA on a returns dataframe (rows = time, cols = assets).

    Useful for reducing multi-asset returns to a few principal risk factors.
    """
    if returns is None or returns.empty:
        return PCAResult(
            components=np.zeros((0, 0)),
            explained_variance_ratio=np.zeros(0),
            mean=np.zeros(0),
            n_components=0,
        )
    returns = returns.fillna(0)
    X = returns.to_numpy(dtype=float)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 1:
        return PCAResult(np.zeros((0, n_features)), np.zeros(0), np.zeros(n_features), 0)

    mean = X.mean(axis=0)
    X_centered = X - mean
    # Use SVD for numerical stability
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        components = Vt  # (n_features, n_features)
        var = (S ** 2) / (n_samples - 1) if n_samples > 1 else S ** 2
        total_var = float(var.sum())
        if total_var <= 0:
            return PCAResult(np.zeros((0, n_features)), np.zeros(0), mean, 0)
        explained_var_ratio = var / total_var
        k = min(n_components, len(components))
        return PCAResult(
            components=components[:k],
            explained_variance_ratio=explained_var_ratio[:k],
            mean=mean,
            n_components=k,
        )
    except Exception as e:
        logger.warning(f"PCA failed: {e}")
        return PCAResult(np.zeros((0, n_features)), np.zeros(0), mean, 0)


# ──────────────────────────────────────────────────────────────────────
# 5. Kalman Filter (1-D)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class KalmanResult:
    filtered: np.ndarray           # smoothed state estimates
    filtered_std: np.ndarray       # estimated std of state
    last_estimate: float
    last_std: float

    def to_dict(self) -> dict:
        return {
            "last_estimate": round(float(self.last_estimate), 6),
            "last_std": round(float(self.last_std), 6),
        }


def kalman_filter(
    prices: np.ndarray,
    process_var: float = 1e-5,
    measurement_var: float = 1e-3,
) -> KalmanResult:
    """1-D Kalman filter for adaptive trend estimation.

    State model: x[t] = x[t-1] + ε_p, ε_p ~ N(0, process_var)
    Measurement: z[t] = x[t] + ε_m, ε_m ~ N(0, measurement_var)

    Returns smoothed state estimates + their estimated std.

    Usage:
        kf = kalman_filter(df["close"].to_numpy())
        # When |price - filtered| > 2 * filtered_std → potential reversal
    """
    if prices is None or len(prices) < 2:
        return KalmanResult(np.zeros(0), np.zeros(0), 0.0, 0.0)
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = prices[0]
    P[0] = 1.0
    for t in range(1, n):
        # Predict
        x_pred = x_hat[t - 1]
        P_pred = P[t - 1] + process_var
        # Update
        K = P_pred / (P_pred + measurement_var)
        x_hat[t] = x_pred + K * (prices[t] - x_pred)
        P[t] = (1 - K) * P_pred
    return KalmanResult(
        filtered=x_hat,
        filtered_std=np.sqrt(P),
        last_estimate=float(x_hat[-1]),
        last_std=float(np.sqrt(P[-1])),
    )


# ──────────────────────────────────────────────────────────────────────
# 6. Hidden Markov Model (3-state Gaussian)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HMMResult:
    states: np.ndarray             # inferred state for each bar
    current_state: int
    state_means: np.ndarray        # mean return per state
    state_stds: np.ndarray         # std of return per state
    state_labels: dict[int, str]   # state_idx → "bull"/"bear"/"sideways"

    def to_dict(self) -> dict:
        return {
            "current_state": int(self.current_state),
            "current_label": self.state_labels.get(int(self.current_state), "?"),
            "state_means": [round(float(m), 6) for m in self.state_means],
            "state_stds": [round(float(s), 6) for s in self.state_stds],
            "state_labels": self.state_labels,
        }


def hmm_regime(
    returns: np.ndarray, n_states: int = 3, n_iter: int = 50,
) -> HMMResult:
    """3-state Gaussian HMM regime detection.

    Uses a simple Baum-Welch EM algorithm (no hmmlearn dependency).
    States are post-processed and labeled by their mean return:
        highest mean → "bull"
        lowest mean  → "bear"
        middle       → "sideways"

    Useful for separating trending vs choppy vs reversing regimes.
    """
    if returns is None or len(returns) < 50:
        return HMMResult(
            states=np.zeros(0, dtype=int),
            current_state=0,
            state_means=np.zeros(n_states),
            state_stds=np.zeros(n_states),
            state_labels={0: "unknown"},
        )
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 50:
        return HMMResult(np.zeros(0, int), 0, np.zeros(n_states), np.zeros(n_states), {0: "unknown"})

    # Initialize with k-means-like clustering
    sorted_rets = np.sort(returns)
    chunks = np.array_split(sorted_rets, n_states)
    means = np.array([float(c.mean()) for c in chunks])
    stds = np.array([max(float(c.std()), 1e-8) for c in chunks])
    # Transition matrix (start with high self-transition)
    A = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(A, 0.9)

    for _ in range(n_iter):
        # Forward (alpha)
        alpha = np.zeros((n, n_states))
        # Initial uniform
        alpha[0] = 1.0 / n_states
        # Emission: Gaussian
        for t in range(1, n):
            for j in range(n_states):
                emit = _gaussian_pdf(returns[t], means[j], stds[j])
                alpha[t, j] = emit * (alpha[t - 1] @ A[:, j])
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s

        # Backward (beta)
        beta = np.zeros((n, n_states))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            for i in range(n_states):
                emit_vec = np.array([
                    _gaussian_pdf(returns[t + 1], means[j], stds[j])
                    for j in range(n_states)
                ])
                # beta[t, i] = sum_j A[i, j] * emit_vec[j] * beta[t+1, j]
                beta[t, i] = float(np.sum(A[i] * emit_vec * beta[t + 1]))
            s = beta[t].sum()
            if s > 0:
                beta[t] /= s

        # Posterior (gamma)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
        gamma /= gamma_sum

        # M-step
        for j in range(n_states):
            g = gamma[:, j]
            total_g = float(g.sum())
            if total_g > 1e-10:
                means[j] = float((g * returns).sum() / total_g)
                stds[j] = max(float(np.sqrt((g * (returns - means[j]) ** 2).sum() / total_g)), 1e-8)
        # Update transition matrix
        new_A = np.zeros_like(A)
        for t in range(n - 1):
            for i in range(n_states):
                for j in range(n_states):
                    emit = _gaussian_pdf(returns[t + 1], means[j], stds[j])
                    new_A[i, j] += gamma[t, i] * emit * A[i, j] * beta[t + 1, j] / \
                                   max(gamma[t + 1, j], 1e-10)
        # Normalize
        row_sums = new_A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        A = new_A / row_sums

    # Label states by mean
    sorted_idx = np.argsort(means)  # ascending
    labels: dict[int, str] = {}
    labels[int(sorted_idx[0])] = "bear"
    labels[int(sorted_idx[-1])] = "bull"
    for i in range(1, n_states - 1):
        labels[int(sorted_idx[i])] = "sideways"

    return HMMResult(
        states=gamma.argmax(axis=1),
        current_state=int(gamma[-1].argmax()),
        state_means=means,
        state_stds=stds,
        state_labels=labels,
    )


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal PDF."""
    if sigma <= 0:
        return 1e-10
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi)))


# ──────────────────────────────────────────────────────────────────────
# 7. Bayesian Win-Rate Update
# ──────────────────────────────────────────────────────────────────────
@dataclass
class BayesianWinRate:
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float          # estimated win probability
    posterior_std: float
    credible_interval_95: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "posterior_alpha": round(self.posterior_alpha, 2),
            "posterior_beta": round(self.posterior_beta, 2),
            "posterior_mean": round(self.posterior_mean, 4),
            "posterior_std": round(self.posterior_std, 4),
            "credible_interval_95": [round(self.credible_interval_95[0], 4),
                                     round(self.credible_interval_95[1], 4)],
        }


def bayesian_winrate(
    prior_alpha: float = 10.0, prior_beta: float = 10.0,
    wins: int = 0, losses: int = 0,
) -> BayesianWinRate:
    """Update a Beta(α, β) prior with observed wins/losses.

    Returns the posterior Beta distribution parameters + mean + 95%
    credible interval. Use this to track strategy win-rate uncertainty
    rather than point estimates.

    Example:
        # Prior: weak belief 50% win rate (10 wins, 10 losses)
        # After 30 trades: 18 wins, 12 losses
        posterior = bayesian_winrate(10, 10, 18, 12)
        print(posterior.posterior_mean)        # ≈ 0.56
        print(posterior.credible_interval_95)  # ≈ (0.43, 0.69)
    """
    alpha = float(prior_alpha) + int(wins)
    beta = float(prior_beta) + int(losses)
    total = alpha + beta
    if total <= 0:
        mean = 0.5; std = 0.0
        ci = (0.5, 0.5)
    else:
        mean = alpha / total
        std = float(np.sqrt(alpha * beta / (total ** 2 * (total + 1))))
        # Approximate 95% CI via normal approximation (good when α, β > 5)
        from scipy.stats import beta as beta_dist  # type: ignore
        try:
            ci = (float(beta_dist.ppf(0.025, alpha, beta)),
                  float(beta_dist.ppf(0.975, alpha, beta)))
        except ImportError:
            ci = (max(0.0, mean - 1.96 * std), min(1.0, mean + 1.96 * std))
    return BayesianWinRate(
        posterior_alpha=alpha, posterior_beta=beta,
        posterior_mean=mean, posterior_std=std, credible_interval_95=ci,
    )


__all__ = [
    "zscore", "zscore_series", "hurst_exponent",
    "CointegrationResult", "cointegration_test",
    "PCAResult", "pca",
    "KalmanResult", "kalman_filter",
    "HMMResult", "hmm_regime",
    "BayesianWinRate", "bayesian_winrate",
]
