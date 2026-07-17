"""
Portfolio Theory — institutional asset allocation
==================================================

Pure-Python implementations of portfolio construction methods:

    1. Modern Portfolio Theory (MPT) — Markowitz mean-variance
    2. Minimum Variance Portfolio
    3. Black-Litterman — combine market prior with investor views
    4. Risk Parity — equal risk contribution
    5. Hierarchical Risk Parity (HRP) — using correlation clustering
    6. Kelly Criterion portfolio — multi-asset Kelly

All functions take a returns DataFrame (rows = time, cols = assets) and
return weight vectors.

Usage:
    from trading_modules.portfolio_theory import (
        markowitz_mpt, min_variance_portfolio, black_litterman,
        risk_parity, hierarchical_risk_parity
    )
    weights = min_variance_portfolio(returns_df)
    bl_weights = black_litterman(
        returns_df, views=[("BTCUSD", 0.05)], view_confidences=[0.6],
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe: float
    method: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 4),
            "expected_volatility": round(self.expected_volatility, 4),
            "sharpe": round(self.sharpe, 3),
            "method": self.method,
            "notes": self.notes,
        }


# ──────────────────────────────────────────────────────────────────────
# 1. MPT (Markowitz mean-variance)
# ──────────────────────────────────────────────────────────────────────
def markowitz_mpt(
    returns: pd.DataFrame, risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
) -> PortfolioResult:
    """Markowitz mean-variance optimization (max Sharpe).

    If target_return is provided, finds min-variance portfolio with that
    return. Otherwise, finds the maximum-Sharpe (tangency) portfolio.
    """
    if returns is None or returns.empty:
        return PortfolioResult({}, 0, 0, 0, "mpt", ["no data"])
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)
    if n == 0:
        return PortfolioResult({}, 0, 0, 0, "mpt", ["no assets"])

    # Grid search over weight combinations (simplified — no scipy)
    best_sharpe = -np.inf
    best_w = np.ones(n) / n
    # Generate random portfolios + equal-weight
    rng = np.random.default_rng(42)
    for _ in range(2000):
        w = rng.dirichlet(np.ones(n))
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ Sigma @ w))
        if vol <= 0:
            continue
        sharpe = (ret - risk_free_rate) / vol
        if target_return is not None and abs(ret - target_return) > 0.001:
            continue
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w
    # Also test equal weight
    w_eq = np.ones(n) / n
    ret_eq = float(w_eq @ mu)
    vol_eq = float(np.sqrt(w_eq @ Sigma @ w_eq))
    sharpe_eq = (ret_eq - risk_free_rate) / vol_eq if vol_eq > 0 else 0
    if sharpe_eq > best_sharpe:
        best_w = w_eq
        best_sharpe = sharpe_eq
    weights = {col: float(w) for col, w in zip(returns.columns, best_w)}
    exp_ret = float(best_w @ mu)
    exp_vol = float(np.sqrt(best_w @ Sigma @ best_w))
    return PortfolioResult(
        weights=weights, expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe=(exp_ret - risk_free_rate) / exp_vol if exp_vol > 0 else 0,
        method="markowitz_mpt",
        notes=[f"tested 2000 random portfolios + equal-weight"],
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Minimum Variance Portfolio
# ──────────────────────────────────────────────────────────────────────
def min_variance_portfolio(returns: pd.DataFrame) -> PortfolioResult:
    """Find the minimum-variance portfolio (long-only)."""
    if returns is None or returns.empty:
        return PortfolioResult({}, 0, 0, 0, "min_variance", ["no data"])
    Sigma = returns.cov().values
    n = Sigma.shape[0]
    if n == 0:
        return PortfolioResult({}, 0, 0, 0, "min_variance", ["no assets"])
    # Analytical solution: w = (Sigma^-1 · 1) / (1' · Sigma^-1 · 1)
    try:
        Sigma_inv = np.linalg.inv(Sigma + np.eye(n) * 1e-8)
        ones = np.ones(n)
        w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
        # Clip negative weights (long-only)
        w = np.clip(w, 0, 1)
        w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    mu = returns.mean().values
    weights = {col: float(x) for col, x in zip(returns.columns, w)}
    exp_ret = float(w @ mu)
    exp_vol = float(np.sqrt(w @ Sigma @ w))
    return PortfolioResult(
        weights=weights, expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe=exp_ret / exp_vol if exp_vol > 0 else 0,
        method="min_variance",
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Black-Litterman
# ──────────────────────────────────────────────────────────────────────
def black_litterman(
    returns: pd.DataFrame,
    views: list[tuple[str, float]],
    view_confidences: Optional[list[float]] = None,
    risk_free_rate: float = 0.0,
    tau: float = 0.025,
) -> PortfolioResult:
    """Black-Litterman: combine market prior with investor views.

    Args:
        returns: historical returns DataFrame
        views: list of (asset_name, expected_return) — e.g., [("BTCUSD", 0.05)]
        view_confidences: list of 0..1 confidences per view (1 = strong)
        risk_free_rate: risk-free rate
        tau: prior confidence scalar (default 0.025)
    """
    if returns is None or returns.empty:
        return PortfolioResult({}, 0, 0, 0, "black_litterman", ["no data"])
    assets = list(returns.columns)
    n = len(assets)
    Sigma = returns.cov().values * 252  # annualized
    # Market prior: use historical mean as "implied equilibrium returns"
    pi = returns.mean().values * 252  # annualized
    # Build P (views matrix) and Q (view returns)
    P = np.zeros((len(views), n))
    Q = np.zeros(len(views))
    for i, (asset, view_ret) in enumerate(views):
        if asset in assets:
            P[i, assets.index(asset)] = 1.0
        Q[i] = float(view_ret)
    # Omega (view uncertainty) — diagonal, scaled by tau * P @ Sigma @ P.T
    if view_confidences:
        Omega = np.diag([max(1e-6, (1 - c) * tau) for c in view_confidences]) * np.diag(P @ Sigma @ P.T)
    else:
        Omega = np.diag(np.diag(tau * P @ Sigma @ P.T)) * 0.1 + np.eye(len(views)) * 1e-6
    # Posterior expected returns
    try:
        tau_Sigma = tau * Sigma
        M = np.linalg.inv(np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P)
        posterior_returns = M @ (np.linalg.inv(tau_Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    except np.linalg.LinAlgError:
        posterior_returns = pi
    # Optimal weights (mean-variance with posterior returns)
    try:
        lambda_risk = 2.5  # risk aversion
        w = np.linalg.inv(lambda_risk * Sigma) @ posterior_returns
        w = np.clip(w, 0, 1)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(n) / n
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    weights = {a: float(x) for a, x in zip(assets, w)}
    exp_ret = float(w @ posterior_returns)
    exp_vol = float(np.sqrt(w @ Sigma @ w))
    return PortfolioResult(
        weights=weights, expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe=(exp_ret - risk_free_rate) / exp_vol if exp_vol > 0 else 0,
        method="black_litterman",
        notes=[f"applied {len(views)} views, tau={tau}"],
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Risk Parity
# ──────────────────────────────────────────────────────────────────────
def risk_parity(returns: pd.DataFrame) -> PortfolioResult:
    """Equal risk contribution portfolio.

    Each asset contributes the same amount of risk to the portfolio.
    """
    if returns is None or returns.empty:
        return PortfolioResult({}, 0, 0, 0, "risk_parity", ["no data"])
    Sigma = returns.cov().values
    n = Sigma.shape[0]
    if n == 0:
        return PortfolioResult({}, 0, 0, 0, "risk_parity", ["no assets"])
    # Iterative solution: w_i ∝ 1 / σ_i, then normalize
    vol = np.sqrt(np.diag(Sigma))
    vol = np.where(vol > 0, vol, 1.0)
    w = 1.0 / vol
    w = w / w.sum()
    # Refine with 10 iterations
    for _ in range(10):
        port_var = w @ Sigma @ w
        if port_var <= 0:
            break
        marginal = Sigma @ w
        contrib = w * marginal
        target = port_var / n
        # Adjust
        w = w * (target / np.where(contrib > 0, contrib, target)) ** 0.5
        w = w / w.sum()
    mu = returns.mean().values
    weights = {col: float(x) for col, x in zip(returns.columns, w)}
    exp_ret = float(w @ mu)
    exp_vol = float(np.sqrt(w @ Sigma @ w))
    return PortfolioResult(
        weights=weights, expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe=exp_ret / exp_vol if exp_vol > 0 else 0,
        method="risk_parity",
    )


# ──────────────────────────────────────────────────────────────────────
# 5. Hierarchical Risk Parity (HRP)
# ──────────────────────────────────────────────────────────────────────
def hierarchical_risk_parity(returns: pd.DataFrame) -> PortfolioResult:
    """HRP — Marcos López de Prado's correlation-clustering allocation.

    Builds a hierarchical tree from correlations, then allocates down the
    tree using recursive bisection.
    """
    if returns is None or returns.empty:
        return PortfolioResult({}, 0, 0, 0, "hrp", ["no data"])
    corr = returns.corr().values
    n = corr.shape[0]
    if n == 0:
        return PortfolioResult({}, 0, 0, 0, "hrp", ["no assets"])
    # Distance matrix: d = sqrt(0.5 × (1 - corr))
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    # Simple linkage: sort assets by distance to first asset (simplified)
    # For a true HRP, we'd do full agglomerative clustering; this is a fast approximation
    order = list(range(n))
    # Cluster by sorting on the first principal component of dist
    try:
        from numpy.linalg import eigh
        eigvals, eigvecs = eigh(dist)
        order = list(np.argsort(eigvecs[:, 0]))
    except Exception:
        pass
    # Recursive bisection
    w = np.ones(n)
    clusters = [order]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            # Compute cluster variances
            var_left = float(np.mean(np.diag(returns.cov().values)[left]))
            var_right = float(np.mean(np.diag(returns.cov().values)[right]))
            if var_left + var_right <= 0:
                alpha = 0.5
            else:
                alpha = 1 - var_left / (var_left + var_right)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1 - alpha)
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    mu = returns.mean().values
    Sigma = returns.cov().values
    weights = {col: float(x) for col, x in zip(returns.columns, w)}
    exp_ret = float(w @ mu)
    exp_vol = float(np.sqrt(w @ Sigma @ w))
    return PortfolioResult(
        weights=weights, expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe=exp_ret / exp_vol if exp_vol > 0 else 0,
        method="hrp",
    )


__all__ = [
    "PortfolioResult",
    "markowitz_mpt", "min_variance_portfolio", "black_litterman",
    "risk_parity", "hierarchical_risk_parity",
]
