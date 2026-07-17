"""
Quant Risk Metrics — extended performance & risk ratios
=======================================================

Beyond the basic Sharpe/Sortino, this module provides:

    1. Calmar Ratio       — annual return / max drawdown
    2. Omega Ratio        — probability-weighted gains/losses
    3. Ulcer Index        — depth × duration of drawdowns
    4. MAR Ratio          — CAGR / max drawdown
    5. Information Ratio  — alpha / tracking error
    6. Treynor Ratio      — excess return / beta
    7. Sterling Ratio     — annual return / avg max drawdown
    8. Burke Ratio        — return / sqrt(sum of squared drawdowns)
    9. Pain Index         — avg drawdown over time

All functions accept a returns series (daily %) and return floats.

Usage:
    from trading_modules.quant_risk_metrics import (
        calmar_ratio, omega_ratio, ulcer_index, mar_ratio,
        information_ratio, treynor_ratio, all_metrics
    )
    metrics = all_metrics(returns_series, benchmark_series=None)
    print(f"Calmar: {metrics['calmar']:.2f}")
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _annualize_return(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = float(np.prod(1 + returns) - 1)
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    if cumulative <= -1:
        return -1.0
    return float((1 + cumulative) ** (1 / n_years) - 1)


def _max_drawdown(returns: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (max_drawdown_pct, drawdown_series)."""
    if len(returns) == 0:
        return 0.0, np.zeros(0)
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    return float(drawdowns.max()), drawdowns


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar = annualized return / max drawdown."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    ann_ret = _annualize_return(r)
    max_dd, _ = _max_drawdown(r)
    if max_dd <= 0:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / max_dd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega = E[max(r - τ, 0)] / E[max(τ - r, 0)] where τ is threshold."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    gains = r[r > threshold] - threshold
    losses = threshold - r[r < threshold]
    exp_gain = float(gains.sum()) if len(gains) > 0 else 0.0
    exp_loss = float(losses.sum()) if len(losses) > 0 else 1e-10
    if exp_loss <= 0:
        return float("inf")
    return exp_gain / exp_loss


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer = sqrt(mean(drawdown²))  — measures depth × duration of drawdowns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    _, drawdowns = _max_drawdown(r)
    return float(np.sqrt(np.mean(drawdowns ** 2)))


def mar_ratio(returns: pd.Series) -> float:
    """MAR = CAGR / max drawdown."""
    return calmar_ratio(returns)  # same calculation


def information_ratio(
    returns: pd.Series, benchmark: pd.Series,
) -> float:
    """IR = (alpha - benchmark) / tracking_error (annualized)."""
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(len(r), len(b))
    if n < 2:
        return 0.0
    r = r[:n]; b = b[:n]
    active = r - b
    te = float(active.std(ddof=1))
    if te <= 0:
        return 0.0
    mean_active = float(active.mean()) * TRADING_DAYS
    return mean_active / (te * np.sqrt(TRADING_DAYS))


def treynor_ratio(
    returns: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.0,
) -> float:
    """Treynor = (ann_return - rf) / beta."""
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(len(r), len(b))
    if n < 5:
        return 0.0
    r = r[:n]; b = b[:n]
    # Beta
    cov = float(np.cov(r, b, ddof=1)[0, 1])
    var_b = float(np.var(b, ddof=1))
    if var_b <= 0:
        return 0.0
    beta = cov / var_b
    if beta == 0:
        return 0.0
    ann_ret = _annualize_return(r)
    return float((ann_ret - risk_free_rate) / beta)


def sterling_ratio(returns: pd.Series, period_days: int = 252) -> float:
    """Sterling = annualized return / avg of worst drawdowns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < period_days:
        return 0.0
    ann_ret = _annualize_return(r)
    # Compute drawdowns over each period
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    # Worst drawdown per year
    n_years = len(r) // period_days
    if n_years == 0:
        return 0.0
    worst_dds = []
    for y in range(n_years):
        chunk = drawdowns[y * period_days: (y + 1) * period_days]
        if len(chunk) > 0:
            worst_dds.append(float(chunk.max()))
    avg_dd = float(np.mean(worst_dds)) if worst_dds else 0.0
    if avg_dd <= 0:
        return float("inf") if ann_ret > 0 else 0.0
    return ann_ret / avg_dd


def burke_ratio(returns: pd.Series) -> float:
    """Burke = annualized return / sqrt(sum of squared drawdowns)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    ann_ret = _annualize_return(r)
    _, drawdowns = _max_drawdown(r)
    # Sum of squared drawdowns
    ssd = float(np.sum(drawdowns ** 2))
    if ssd <= 0:
        return float("inf") if ann_ret > 0 else 0.0
    return ann_ret / np.sqrt(ssd)


def pain_index(returns: pd.Series) -> float:
    """Pain Index = average drawdown over time."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    _, drawdowns = _max_drawdown(r)
    return float(np.mean(drawdowns))


def all_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute all available risk metrics at once."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    metrics = {}
    metrics["calmar"] = calmar_ratio(returns)
    metrics["omega"] = omega_ratio(returns, threshold=risk_free_rate)
    metrics["ulcer_index"] = ulcer_index(returns)
    metrics["mar"] = mar_ratio(returns)
    metrics["sterling"] = sterling_ratio(returns)
    metrics["burke"] = burke_ratio(returns)
    metrics["pain_index"] = pain_index(returns)
    if benchmark is not None:
        metrics["information_ratio"] = information_ratio(returns, benchmark)
        metrics["treynor"] = treynor_ratio(returns, benchmark, risk_free_rate)
    # Also include Sharpe/Sortino for completeness
    if len(r) > 1 and r.std() > 0:
        metrics["sharpe"] = float((r.mean() - risk_free_rate / TRADING_DAYS) /
                                  r.std() * np.sqrt(TRADING_DAYS))
        downside = r[r < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics["sortino"] = float((r.mean() - risk_free_rate / TRADING_DAYS) /
                                       downside.std() * np.sqrt(TRADING_DAYS))
        else:
            metrics["sortino"] = 0.0
    else:
        metrics["sharpe"] = 0.0
        metrics["sortino"] = 0.0
    # Max drawdown
    max_dd, _ = _max_drawdown(r)
    metrics["max_drawdown"] = max_dd
    metrics["annualized_return"] = _annualize_return(r)
    metrics["annualized_volatility"] = float(r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
    return metrics


__all__ = [
    "calmar_ratio", "omega_ratio", "ulcer_index", "mar_ratio",
    "information_ratio", "treynor_ratio", "sterling_ratio",
    "burke_ratio", "pain_index", "all_metrics",
]
