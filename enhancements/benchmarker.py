"""enhancements.benchmarker
=====================================================================
Day 159 — Performance benchmarker.

Compares the trading system's performance against benchmarks:
  - Buy-and-hold (BTCUSD, ETHUSD)
  - Equal-weight portfolio
  - 60/40 portfolio (60% crypto, 40% cash)
  - Risk-free rate (T-bill)

Outputs relative metrics:
  - Alpha (excess return vs benchmark)
  - Beta (correlation to benchmark)
  - Information ratio (alpha / tracking error)
  - Up/down capture ratios
  - Calmar ratio (annual return / max DD)

A trading system that returns 20% but buy-and-hold returned 50%
has NEGATIVE alpha — it destroyed value vs. simply holding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("enhancements.benchmarker")


@dataclass
class BenchmarkResult:
    strategy_name: str
    benchmark_name: str
    strategy_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float                    # strategy - benchmark
    beta: float                         # correlation * (strat_vol / bench_vol)
    information_ratio: float            # alpha / tracking_error
    tracking_error_pct: float
    up_capture_ratio: float             # strategy return in up markets / benchmark
    down_capture_ratio: float           # strategy return in down markets / benchmark
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_drawdown_pct: float
    benchmark_max_drawdown_pct: float
    calmar_ratio: float                 # annual return / max DD
    verdict: str                        # "outperform" / "underperform" / "neutral"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "benchmark_name": self.benchmark_name,
            "strategy_return_pct": self.strategy_return_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_pct": self.alpha_pct,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "tracking_error_pct": self.tracking_error_pct,
            "up_capture_ratio": self.up_capture_ratio,
            "down_capture_ratio": self.down_capture_ratio,
            "strategy_sharpe": self.strategy_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "strategy_max_drawdown_pct": self.strategy_max_drawdown_pct,
            "benchmark_max_drawdown_pct": self.benchmark_max_drawdown_pct,
            "calmar_ratio": self.calmar_ratio,
            "verdict": self.verdict,
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class Benchmarker:
    def __init__(self, annualisation_factor: int = 252) -> None:
        self.annual_factor = int(annualisation_factor)

    # ----------------------------------------------------------------
    def compare(
        self,
        strategy_equity: list[float] | np.ndarray,
        benchmark_equity: list[float] | np.ndarray,
        strategy_name: str = "strategy",
        benchmark_name: str = "buy_and_hold",
        risk_free_rate: float = 0.02,
    ) -> BenchmarkResult:
        """Compare strategy equity curve vs benchmark equity curve."""
        s = np.array(strategy_equity, dtype=float)
        b = np.array(benchmark_equity, dtype=float)
        if len(s) < 2 or len(b) < 2:
            raise ValueError("need at least 2 data points")
        # Align lengths
        min_len = min(len(s), len(b))
        s = s[-min_len:]
        b = b[-min_len:]
        # Returns
        s_ret = np.diff(s) / s[:-1]
        b_ret = np.diff(b) / b[:-1]
        # Total returns
        s_total = (s[-1] / s[0]) - 1.0
        b_total = (b[-1] / b[0]) - 1.0
        # Alpha
        alpha = s_total - b_total
        # Beta
        if b_ret.std() > 0:
            cov = np.cov(s_ret, b_ret)[0, 1]
            beta = float(cov / b_ret.var())
        else:
            beta = 0.0
        # Tracking error
        tracking_diff = s_ret - b_ret
        tracking_error = float(tracking_diff.std() * math.sqrt(self.annual_factor))
        # Information ratio
        ir = float((alpha / len(s_ret)) * self.annual_factor / tracking_error) if tracking_error > 0 else 0.0
        # Capture ratios
        up_mask = b_ret > 0
        down_mask = b_ret < 0
        if up_mask.sum() > 0 and b_ret[up_mask].mean() != 0:
            up_capture = float(s_ret[up_mask].mean() / b_ret[up_mask].mean())
        else:
            up_capture = 0.0
        if down_mask.sum() > 0 and b_ret[down_mask].mean() != 0:
            down_capture = float(s_ret[down_mask].mean() / b_ret[down_mask].mean())
        else:
            down_capture = 0.0
        # Sharpe ratios
        s_sharpe = self._sharpe(s_ret, risk_free_rate)
        b_sharpe = self._sharpe(b_ret, risk_free_rate)
        # Max drawdowns
        s_dd = self._max_drawdown(s)
        b_dd = self._max_drawdown(b)
        # Calmar
        n_periods = len(s_ret)
        years = n_periods / self.annual_factor if n_periods > 0 else 1
        annual_return = ((s[-1] / s[0]) ** (1 / years) - 1) if years > 0 else s_total
        calmar = float(annual_return / s_dd) if s_dd > 0 else 0.0
        # Verdict
        if alpha > 0.05 and ir > 0.5:
            verdict = "outperform"
        elif alpha < -0.05 and ir < -0.5:
            verdict = "underperform"
        else:
            verdict = "neutral"
        return BenchmarkResult(
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
            strategy_return_pct=float(s_total),
            benchmark_return_pct=float(b_total),
            alpha_pct=float(alpha),
            beta=beta,
            information_ratio=ir,
            tracking_error_pct=tracking_error,
            up_capture_ratio=up_capture,
            down_capture_ratio=down_capture,
            strategy_sharpe=s_sharpe,
            benchmark_sharpe=b_sharpe,
            strategy_max_drawdown_pct=float(s_dd),
            benchmark_max_drawdown_pct=float(b_dd),
            calmar_ratio=calmar,
            verdict=verdict,
            details={
                "n_periods": int(n_periods),
                "years": float(years),
                "risk_free_rate": float(risk_free_rate),
            },
        )

    # ----------------------------------------------------------------
    def buy_and_hold(self, prices: list[float] | np.ndarray,
                      initial_capital: float = 10_000.0) -> np.ndarray:
        """Generate buy-and-hold equity curve from price series."""
        p = np.array(prices, dtype=float)
        if len(p) < 2:
            return np.array([initial_capital])
        # Equity = initial_capital * (price / price[0])
        return initial_capital * (p / p[0])

    # ----------------------------------------------------------------
    def _sharpe(self, returns: np.ndarray, rf: float = 0.02) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns - rf / self.annual_factor
        return float(excess.mean() / returns.std() * math.sqrt(self.annual_factor))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(abs(dd.min())) if dd.size else 0.0
