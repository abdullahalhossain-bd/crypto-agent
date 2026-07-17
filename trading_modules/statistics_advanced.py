"""
Advanced Statistics — bootstrap, EVT, copulas, Gaussian processes
==================================================================

Pure-Python implementations of advanced statistical methods:

    1. Bootstrap           — resample-based confidence intervals
    2. Extreme Value Theory — fit Generalized Pareto Distribution to tail
    3. Copulas             — capture dependency structure between assets
    4. Gaussian Process    — non-parametric regression for curve fitting

No scipy dependency (uses numpy linear algebra only).

Usage:
    from trading_modules.statistics_advanced import (
        bootstrap_ci, evt_tail_risk, gaussian_copula, gaussian_process_fit
    )
    # 95% CI for mean return
    ci = bootstrap_ci(returns, statistic=np.mean, n_resamples=10000)

    # Tail risk (95% percentile of losses)
    evt = evt_tail_risk(returns, threshold_pct=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Bootstrap
# ──────────────────────────────────────────────────────────────────────
@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray
    n_resamples: int

    def to_dict(self) -> dict:
        return {
            "point_estimate": round(self.point_estimate, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "n_resamples": self.n_resamples,
        }


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Non-parametric bootstrap confidence interval.

    Args:
        data: 1-D array of observations
        statistic: function that takes a sample and returns a scalar
        n_resamples: # of bootstrap resamples
        confidence: CI level (0..1)
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return BootstrapResult(0.0, 0.0, 0.0, np.zeros(0), 0)
    rng = np.random.default_rng(seed)
    point_estimate = float(statistic(data))
    boot_stats = np.zeros(n_resamples)
    n = len(data)
    for i in range(n_resamples):
        resample = data[rng.integers(0, n, n)]
        boot_stats[i] = float(statistic(resample))
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower, ci_upper=ci_upper,
        bootstrap_samples=boot_stats, n_resamples=n_resamples,
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Extreme Value Theory (EVT) — Generalized Pareto Distribution
# ──────────────────────────────────────────────────────────────────────
@dataclass
class EVTResult:
    threshold: float
    shape_param: float          # ξ (xi) — tail index
    scale_param: float          # β (beta)
    var_estimate: float         # Value at Risk from EVT
    cvar_estimate: float        # CVaR (expected shortfall)
    n_exceedances: int

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6),
            "shape_param": round(self.shape_param, 4),
            "scale_param": round(self.scale_param, 4),
            "var_estimate": round(self.var_estimate, 6),
            "cvar_estimate": round(self.cvar_estimate, 6),
            "n_exceedances": self.n_exceedances,
        }


def evt_tail_risk(
    returns: np.ndarray, threshold_pct: float = 5.0,
) -> EVTResult:
    """Fit Generalized Pareto Distribution to tail losses.

    Args:
        returns: 1-D array of returns (we look at the left tail = losses)
        threshold_pct: percentile for threshold (5 = bottom 5%)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 50:
        return EVTResult(0, 0, 0, 0, 0, 0)
    # Work with losses (negative returns)
    losses = -r[r < 0]
    if len(losses) < 10:
        return EVTResult(0, 0, 0, 0, 0, 0)
    threshold = float(np.percentile(losses, 100 - threshold_pct))
    exceedances = losses[losses > threshold] - threshold
    n_exc = len(exceedances)
    if n_exc < 5:
        return EVTResult(threshold, 0, 0, 0, 0, n_exc)
    # Estimate GPD parameters via method of moments (simplified)
    # Shape: ξ = 0.5 × (1 - (mean / std)^2)  — Hill estimator approximation
    mean_exc = float(exceedances.mean())
    var_exc = float(exceedances.var(ddof=1))
    if var_exc <= 0 or mean_exc <= 0:
        return EVTResult(threshold, 0, mean_exc, threshold + mean_exc, threshold + mean_exc, n_exc)
    # Hill estimator (simplified)
    sorted_exc = np.sort(exceedances)[::-1]  # descending
    k = min(n_exc - 1, max(5, n_exc // 4))
    if k > 0 and sorted_exc[k] > 0:
        hill_xi = float((np.mean(np.log(sorted_exc[:k] / sorted_exc[k]))))
    else:
        hill_xi = 0.1
    # Scale: β = mean × (1 - ξ) / 2  (approximation)
    scale = mean_exc * (1 - 0.5 * hill_xi) / 2 if mean_exc > 0 else var_exc
    if scale <= 0:
        scale = mean_exc
    # VaR from EVT (95%)
    n = len(r)
    var_prob = 0.05
    if hill_xi != 0:
        var_multiplier = (scale / hill_xi) * (
            (n / n_exc * var_prob) ** (-hill_xi) - 1
        )
    else:
        var_multiplier = scale * np.log(n / n_exc * var_prob)
    var_estimate = threshold + max(var_multiplier, 0)
    # CVaR from EVT
    if hill_xi < 1 and hill_xi != 0:
        cvar_estimate = var_estimate * (1 + (scale - hill_xi * (var_estimate - threshold)) /
                                         ((1 - hill_xi) * (var_estimate - threshold + 1e-10)))
    else:
        cvar_estimate = var_estimate * 1.5
    return EVTResult(
        threshold=threshold,
        shape_param=hill_xi,
        scale_param=float(scale),
        var_estimate=float(var_estimate),
        cvar_estimate=float(cvar_estimate),
        n_exceedances=n_exc,
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Gaussian Copula
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CopulaResult:
    correlation_matrix: np.ndarray
    kendall_tau: np.ndarray
    spearman_rho: np.ndarray
    tail_dependence: dict  # pair_name → {lower, upper}

    def to_dict(self) -> dict:
        return {
            "correlation_matrix": self.correlation_matrix.tolist(),
            "kendall_tau": self.kendall_tau.tolist(),
            "spearman_rho": self.spearman_rho.tolist(),
            "tail_dependence": self.tail_dependence,
        }


def gaussian_copula(returns: pd.DataFrame) -> CopulaResult:
    """Fit a Gaussian copula to a returns DataFrame.

    The copula captures the dependency structure separately from the
    marginal distributions. Useful for stress-testing correlated assets.

    Args:
        returns: DataFrame with one column per asset
    """
    if returns is None or returns.empty:
        return CopulaResult(np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), {})
    # Convert to uniform via empirical CDF (rank-based)
    n, p = returns.shape
    # Rank each column and divide by (n+1) to get pseudo-observations
    ranks = returns.rank(axis=0) / (n + 1)
    # Transform to standard normal via inverse CDF (probit)
    # z = sqrt(2) * erfinv(2u - 1)
    z = np.sqrt(2) * np.vectorize(_erfinv)(2 * ranks.values - 1)
    # Correlation of the z-scores is the copula correlation
    corr = np.corrcoef(z.T) if p > 1 else np.array([[1.0]])
    # Kendall's tau
    kendall = returns.corr(method="kendall").values
    # Spearman's rho
    spearman = returns.corr(method="spearman").values
    # Tail dependence (for Gaussian copula, tail dependence = 0 in theory,
    # but we compute empirical tail dependence coefficients)
    tail_dep: dict[str, dict] = {}
    cols = list(returns.columns)
    for i in range(p):
        for j in range(i + 1, p):
            x = returns.iloc[:, i].values
            y = returns.iloc[:, j].values
            pair_name = f"{cols[i]}_{cols[j]}"
            # Lower tail dependence: P(Y < q_5 | X < q_5)
            q5_x = np.percentile(x, 5)
            q5_y = np.percentile(y, 5)
            x_low = x < q5_x
            y_low = y < q5_y
            n_x_low = int(x_low.sum())
            if n_x_low > 0:
                lower_td = float(y_low[x_low].sum() / n_x_low)
            else:
                lower_td = 0.0
            # Upper tail dependence
            q95_x = np.percentile(x, 95)
            q95_y = np.percentile(y, 95)
            x_high = x > q95_x
            y_high = y > q95_y
            n_x_high = int(x_high.sum())
            if n_x_high > 0:
                upper_td = float(y_high[x_high].sum() / n_x_high)
            else:
                upper_td = 0.0
            tail_dep[pair_name] = {"lower": round(lower_td, 3), "upper": round(upper_td, 3)}
    return CopulaResult(
        correlation_matrix=corr,
        kendall_tau=kendall,
        spearman_rho=spearman,
        tail_dependence=tail_dep,
    )


def _erfinv(x: float) -> float:
    """Approximate inverse error function (Winitzki approximation)."""
    if x >= 1:
        return 5.0
    if x <= -1:
        return -5.0
    a = 0.147
    ln_term = np.log(1 - x * x)
    factor = 2 / (np.pi * a) + ln_term / 2
    sqrt_term = np.sqrt(factor ** 2 - ln_term / a)
    result = sqrt_term - factor
    return float(np.sign(x) * np.sqrt(result))


# ──────────────────────────────────────────────────────────────────────
# 4. Gaussian Process (simplified)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GPResult:
    mean: np.ndarray
    std: np.ndarray
    log_marginal_likelihood: float


def gaussian_process_fit(
    X: np.ndarray, y: np.ndarray,
    X_test: np.ndarray,
    length_scale: float = 1.0, noise: float = 0.1,
) -> GPResult:
    """Simplified Gaussian Process regression with RBF kernel.

    Args:
        X: training inputs (n, 1)
        y: training outputs (n,)
        X_test: test inputs (m, 1)
        length_scale: RBF kernel length scale
        noise: observation noise variance
    """
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float).reshape(-1, 1)
    n = len(X)
    if n == 0 or len(X_test) == 0:
        return GPResult(np.zeros(len(X_test)), np.zeros(len(X_test)), 0.0)
    # RBF kernel: K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
    def rbf(a, b):
        sq_dist = (a - b.T) ** 2
        return np.exp(-sq_dist / (2 * length_scale ** 2))
    K = rbf(X, X) + noise ** 2 * np.eye(n)
    K_star = rbf(X_test, X)
    K_star_star = rbf(X_test, X_test)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 0)  # numerical stability
        std = np.sqrt(var)
        # Log marginal likelihood
        lml = -0.5 * (y @ alpha + np.sum(np.log(np.diag(L))) + n * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        mean = np.full(len(X_test), float(y.mean()))
        std = np.full(len(X_test), float(y.std()))
        lml = 0.0
    return GPResult(mean=mean, std=std, log_marginal_likelihood=float(lml))


__all__ = [
    "BootstrapResult", "bootstrap_ci",
    "EVTResult", "evt_tail_risk",
    "CopulaResult", "gaussian_copula",
    "GPResult", "gaussian_process_fit",
]
