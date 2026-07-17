"""trading_modules/institutional_performance_analytics.py
=====================================================================
Institutional Performance Analytics (Principle #197)
=====================================================================
Computes all institutional-grade performance metrics:
    - Sharpe Ratio (risk-adjusted return)
    - Sortino Ratio (downside-adjusted)
    - Calmar Ratio (return / max drawdown)
    - Expectancy (average $ per trade)
    - Profit Factor (gross profit / gross loss)
    - Maximum Drawdown (peak-to-trough)
    - Win Rate
    - Average R Multiple
    - Omega Ratio
    - Tail Ratio
    - Value at Risk (VaR)
    - Conditional VaR (CVaR)

Usage:
    analytics = InstitutionalPerformanceAnalytics()

    # Record trades
    for trade in trades:
        analytics.record_trade(pnl=42, r_multiple=1.8)

    # Get full report
    report = analytics.report()
    # report = {
    #     "sharpe": 1.85,
    #     "sortino": 2.34,
    #     "calmar": 0.92,
    #     "expectancy": 32.50,
    #     "profit_factor": 2.1,
    #     "max_drawdown_pct": 8.5,
    #     "win_rate": 0.62,
    #     "avg_r": 0.85,
    #     "omega": 1.75,
    #     "tail_ratio": 1.4,
    #     "var_95": 85.0,
    #     "cvar_95": 120.0,
    # }
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.institutional_performance_analytics")


@dataclass
class PerformanceReport:
    """Complete institutional performance report."""
    # Trade stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0  # avg $ per trade
    profit_factor: float = 0.0

    # R multiples
    avg_r: float = 0.0
    expectancy_r: float = 0.0

    # Risk-adjusted (annualized assuming 252*24*4 = 24192 15-min bars/year)
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    omega: float = 0.0
    tail_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0

    # Risk
    var_95: float = 0.0    # 95% Value at Risk
    cvar_95: float = 0.0   # Conditional VaR
    std_dev: float = 0.0   # standard deviation of returns
    downside_dev: float = 0.0  # downside deviation

    # Equity
    initial_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    total_return_pct: float = 0.0

    # Grade
    grade: str = "C"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "expectancy": round(self.expectancy, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_r": round(self.avg_r, 3),
            "expectancy_r": round(self.expectancy_r, 3),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "omega": round(self.omega, 3),
            "tail_ratio": round(self.tail_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
            "var_95": round(self.var_95, 2),
            "cvar_95": round(self.cvar_95, 2),
            "std_dev": round(self.std_dev, 2),
            "downside_dev": round(self.downside_dev, 2),
            "initial_equity": round(self.initial_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "grade": self.grade,
            "description": self.description,
        }


class InstitutionalPerformanceAnalytics:
    """Computes institutional-grade performance metrics."""

    # Annualization factor for 15-min bars
    BARS_PER_YEAR = 252 * 24 * 4

    def __init__(self,
                 initial_equity: float = 10000.0,
                 risk_free_rate: float = 0.02,
                 history_size: int = 2000):
        """Initialize analytics.

        Args:
            initial_equity: starting equity
            risk_free_rate: annual risk-free rate (for Sharpe)
            history_size: max trades to keep
        """
        self.initial_equity = initial_equity
        self.risk_free_rate = risk_free_rate
        self._lock = threading.RLock()
        self._trades: Deque[dict] = deque(maxlen=history_size)
        self._equity_curve: List[float] = [initial_equity]
        self._peak = initial_equity
        self._max_dd = 0.0
        self._max_dd_pct = 0.0

    def record_trade(self, pnl: float, r_multiple: float = 0.0,
                     hold_time_s: float = 0) -> None:
        """Record a completed trade."""
        with self._lock:
            self._trades.append({
                "timestamp": time.time(),
                "pnl": pnl,
                "r_multiple": r_multiple,
                "hold_time_s": hold_time_s,
                "win": pnl > 0,
            })
            # Update equity curve
            new_equity = self._equity_curve[-1] + pnl
            self._equity_curve.append(new_equity)
            if new_equity > self._peak:
                self._peak = new_equity
            # Drawdown
            dd = self._peak - new_equity
            dd_pct = (dd / max(self._peak, 1)) * 100
            if dd > self._max_dd:
                self._max_dd = dd
                self._max_dd_pct = dd_pct

    def report(self) -> PerformanceReport:
        """Generate full performance report."""
        r = PerformanceReport(
            initial_equity=self.initial_equity,
        )

        with self._lock:
            trades = list(self._trades)
            equity_curve = list(self._equity_curve)

        if not trades:
            r.description = "No trades recorded"
            return r

        # === Basic stats ===
        pnls = [t["pnl"] for t in trades]
        rs = [t["r_multiple"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        r.total_trades = len(trades)
        r.wins = len(wins)
        r.losses = len(losses)
        r.win_rate = len(wins) / max(len(trades), 1)

        r.total_pnl = sum(pnls)
        r.avg_pnl = r.total_pnl / len(pnls)
        r.avg_win = sum(wins) / max(len(wins), 1) if wins else 0
        r.avg_loss = sum(losses) / max(len(losses), 1) if losses else 0
        r.expectancy = r.avg_pnl

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        r.profit_factor = gross_profit / max(gross_loss, 0.01)

        r.avg_r = sum(rs) / len(rs) if rs else 0
        r.expectancy_r = r.avg_r

        # === Risk-adjusted ===
        if len(pnls) >= 10:
            returns = np.diff(equity_curve[-min(len(equity_curve), 500):])
            if len(returns) > 1 and np.std(returns) > 0:
                # Sharpe (annualized)
                r.std_dev = float(np.std(returns))
                r.sharpe = float(
                    (np.mean(returns) - self.risk_free_rate / self.BARS_PER_YEAR)
                    / max(r.std_dev, 1e-10)
                    * np.sqrt(self.BARS_PER_YEAR)
                )
                # Sortino (downside deviation only)
                downside = returns[returns < 0]
                if len(downside) > 0:
                    r.downside_dev = float(np.std(downside))
                    r.sortino = float(
                        (np.mean(returns) - self.risk_free_rate / self.BARS_PER_YEAR)
                        / max(r.downside_dev, 1e-10)
                        * np.sqrt(self.BARS_PER_YEAR)
                    )
                # Omega ratio
                if len(downside) > 0:
                    upside = returns[returns > 0]
                    r.omega = float(sum(upside) / max(abs(sum(downside)), 1e-10))
                # Tail ratio
                if len(returns) > 20:
                    p95 = float(np.percentile(returns, 95))
                    p5 = float(np.percentile(returns, 5))
                    r.tail_ratio = p95 / max(abs(p5), 1e-10)

        # === Drawdown ===
        r.max_drawdown = self._max_dd
        r.max_drawdown_pct = self._max_dd_pct
        current_equity = equity_curve[-1]
        r.current_equity = current_equity
        r.peak_equity = self._peak
        current_dd = self._peak - current_equity
        r.current_drawdown = current_dd
        r.current_drawdown_pct = (current_dd / max(self._peak, 1)) * 100

        # === Calmar ===
        if r.max_drawdown_pct > 0:
            total_return = (current_equity - self.initial_equity) / self.initial_equity
            r.calmar = float(total_return / max(r.max_drawdown_pct / 100, 0.01))

        r.total_return_pct = ((current_equity - self.initial_equity) / self.initial_equity) * 100

        # === VaR / CVaR ===
        if len(pnls) >= 20:
            r.var_95 = float(abs(np.percentile(pnls, 5)))  # 5th percentile loss
            tail_losses = [p for p in pnls if p <= np.percentile(pnls, 5)]
            r.cvar_95 = float(abs(np.mean(tail_losses))) if tail_losses else r.var_95

        # === Grade ===
        r.grade, r.description = self._grade(r)

        return r

    def _grade(self, r: PerformanceReport) -> tuple:
        """Generate grade based on metrics."""
        score = 0
        if r.sharpe > 2: score += 2
        elif r.sharpe > 1: score += 1
        if r.profit_factor > 2: score += 2
        elif r.profit_factor > 1.5: score += 1
        if r.win_rate > 0.55: score += 1
        if r.max_drawdown_pct < 5: score += 2
        elif r.max_drawdown_pct < 10: score += 1
        if r.expectancy_r > 0.5: score += 1

        if score >= 7:
            return "A+", "Excellent — institutional grade"
        elif score >= 5:
            return "A", "Very good performance"
        elif score >= 3:
            return "B", "Acceptable performance"
        elif score >= 1:
            return "C", "Below average — needs improvement"
        else:
            return "F", "Poor performance — review urgently"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Get quick summary."""
        r = self.report()
        return {
            "trades": r.total_trades,
            "sharpe": round(r.sharpe, 2),
            "profit_factor": round(r.profit_factor, 2),
            "win_rate": round(r.win_rate, 3),
            "max_dd_pct": round(r.max_drawdown_pct, 1),
            "grade": r.grade,
            "total_return_pct": round(r.total_return_pct, 1),
        }
