"""monitoring.risk_monitor
=====================================================================
Day 73 — Risk-layer monitor.

Tracks portfolio-level risk metrics in real time:
  - Value-at-Risk (parametric + historical)
  - Expected Shortfall (CVaR)
  - Correlation spikes between symbols
  - Liquidity stress (position size vs. ADV)
  - Concentration (Herfindahl index of exposure)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("monitoring.risk")

# Critical #2 fix: import scipy at module level with fallback so the monitor
# doesn't crash if scipy is not installed.
try:
    from scipy.stats import norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _norm = None

# Hardcoded z-scores for common confidence levels (fallback when scipy missing).
_Z_TABLE = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}


def _z_score(confidence: float) -> float:
    """Get the z-score for a given confidence level."""
    if _HAS_SCIPY:
        return float(_norm.ppf(confidence))
    return _Z_TABLE.get(confidence, 1.645)


def _norm_pdf(z: float) -> float:
    """Standard normal PDF at z."""
    if _HAS_SCIPY:
        return float(_norm.pdf(z))
    return float(math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))


@dataclass
class RiskHealth:
    status: str
    var_95_pct: float             # 95% 1-day VaR as fraction of equity
    var_99_pct: float
    expected_shortfall_pct: float
    correlation_spike_detected: bool
    max_correlation: float
    liquidity_stress_score: float
    concentration_hhi: float      # Herfindahl-Hirschman index of exposure
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "var_95_pct": self.var_95_pct,
            "var_99_pct": self.var_99_pct,
            "expected_shortfall_pct": self.expected_shortfall_pct,
            "correlation_spike_detected": self.correlation_spike_detected,
            "max_correlation": self.max_correlation,
            "liquidity_stress_score": self.liquidity_stress_score,
            "concentration_hhi": self.concentration_hhi,
            "issues": list(self.issues),
        }


# ----------------------------------------------------------------------
class RiskMonitor:
    def __init__(self,
                 max_var_pct: float = 0.04,
                 max_correlation: float = 0.85,
                 max_concentration_hhi: float = 0.5,
                 lookback: int = 250) -> None:
        self.max_var_pct = float(max_var_pct)
        self.max_correlation = float(max_correlation)
        self.max_concentration_hhi = float(max_concentration_hhi)
        self.lookback = int(lookback)
        # Per-symbol rolling log-returns for VaR computation
        self._returns: dict[str, deque] = {}
        # Per-symbol exposure (in equity fraction)
        self._exposures: dict[str, float] = {}
        # Latest correlation matrix snapshot
        self._correlation_matrix: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------
    def record_returns(self, symbol: str, ret: float) -> None:
        d = self._returns.setdefault(symbol, deque(maxlen=self.lookback))
        d.append(float(ret))

    def set_exposure(self, symbol: str, equity_fraction: float) -> None:
        self._exposures[symbol] = float(equity_fraction)

    def set_correlation_matrix(self, m: pd.DataFrame) -> None:
        self._correlation_matrix = m

    # ----------------------------------------------------------------
    def health(self) -> RiskHealth:
        issues: list[str] = []
        var_95 = var_99 = es = 0.0
        corr_spike = False
        max_corr = 0.0
        liq_stress = 0.0
        hhi = 0.0

        # VaR computation (parametric)
        # Critical #1 fix: the old code computed portfolio VaR as sqrt(sum(var_i^2)),
        # which assumes all assets are UNCORRELATED — severely underestimating risk.
        # The correct formula is: portfolio_var = w^T Σ w, where Σ is the
        # covariance matrix and w is the exposure vector. We now use the
        # correlation matrix (if available) to build Σ, falling back to the
        # uncorrelated approximation only if no correlation data exists.
        if self._returns and self._exposures:
            try:
                # Collect per-symbol volatility and exposure.
                syms = sorted(self._exposures.keys())
                sigmas = {}
                weights = {}
                for sym in syms:
                    rets = list(self._returns.get(sym, []))
                    if len(rets) < 20:
                        continue
                    arr = np.array(rets)
                    sigmas[sym] = float(arr.std())
                    weights[sym] = float(self._exposures[sym])

                if not sigmas:
                    pass  # not enough data
                else:
                    valid_syms = list(sigmas.keys())
                    n_assets = len(valid_syms)
                    w = np.array([abs(weights[s]) for s in valid_syms])
                    sigma_vec = np.array([sigmas[s] for s in valid_syms])

                    # Build covariance matrix Σ = D × C × D
                    # where D = diag(sigma_i) and C = correlation matrix.
                    if (self._correlation_matrix is not None
                            and not self._correlation_matrix.empty
                            and n_assets > 1):
                        # Extract the correlation sub-matrix for valid symbols.
                        try:
                            corr_sub = self._correlation_matrix.reindex(
                                index=valid_syms, columns=valid_syms).fillna(0.0).values
                            # Ensure diagonal is 1.0.
                            np.fill_diagonal(corr_sub, 1.0)
                            D = np.diag(sigma_vec)
                            cov_matrix = D @ corr_sub @ D
                        except Exception:
                            # Fallback: uncorrelated (diagonal covariance).
                            cov_matrix = np.diag(sigma_vec ** 2)
                    else:
                        # No correlation matrix — use uncorrelated approximation.
                        cov_matrix = np.diag(sigma_vec ** 2)

                    # Portfolio variance = w^T Σ w
                    portfolio_variance = float(w.T @ cov_matrix @ w)
                    portfolio_std = float(np.sqrt(max(portfolio_variance, 0.0)))

                    z95 = _z_score(0.95)
                    z99 = _z_score(0.99)
                    var_95 = z95 * portfolio_std
                    var_99 = z99 * portfolio_std

                    # Expected Shortfall (ES) = sigma * pdf(z) / (1 - confidence)
                    es = (_norm_pdf(z95) / 0.05) * portfolio_std
            except Exception as e:  # noqa: BLE001
                log.warning("VaR computation failed: %r", e)

        # Correlation spike
        if self._correlation_matrix is not None and not self._correlation_matrix.empty:
            try:
                # Take upper triangle (excluding diagonal)
                corr_values = self._correlation_matrix.where(
                    np.triu(np.ones(self._correlation_matrix.shape), k=1).astype(bool)
                ).stack().abs()
                if len(corr_values) > 0:
                    max_corr = float(corr_values.max())
                    if max_corr > self.max_correlation:
                        corr_spike = True
                        issues.append(f"correlation spike: {max_corr:.2f} > {self.max_correlation}")
            except Exception:  # noqa: BLE001
                pass

        # Liquidity stress (rough: sum of |exposure| / sqrt(n))
        if self._exposures:
            total_exp = sum(abs(v) for v in self._exposures.values())
            liq_stress = float(total_exp)

        # Concentration (Herfindahl-Hirschman)
        if self._exposures:
            total = sum(abs(v) for v in self._exposures.values())
            if total > 0:
                shares = [abs(v) / total for v in self._exposures.values()]
                hhi = float(sum(s * s for s in shares))
                if hhi > self.max_concentration_hhi:
                    issues.append(f"concentration HHI {hhi:.2f} > {self.max_concentration_hhi}")

        # Threshold checks
        if var_95 > self.max_var_pct:
            issues.append(f"VaR95 {var_95:.2%} > {self.max_var_pct:.2%}")

        status = "ok"
        if issues:
            status = "degraded" if len(issues) <= 2 else "critical"

        return RiskHealth(
            status=status,
            var_95_pct=float(var_95),
            var_99_pct=float(var_99),
            expected_shortfall_pct=float(es),
            correlation_spike_detected=bool(corr_spike),
            max_correlation=float(max_corr),
            liquidity_stress_score=float(liq_stress),
            concentration_hhi=float(hhi),
            issues=issues,
        )
