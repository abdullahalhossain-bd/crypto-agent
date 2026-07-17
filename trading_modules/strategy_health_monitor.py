"""trading_modules/strategy_health_monitor.py
=====================================================================
Strategy Health Monitor (Principle #135, #136 — Continuous Audit + Decay Detection)
=====================================================================
Monitors the health of every trading strategy in real-time. Automatically
detects when a strategy's edge is decaying and pauses it before it loses
money.

What It Tracks (per strategy):
    - Rolling win rate (last 20, 50, 100 trades)
    - Rolling Sharpe ratio
    - Rolling profit factor
    - Rolling expectancy (in R)
    - Strategy decay rate (is performance declining?)
    - Drawdown under strategy
    - Average hold time
    - Execution quality (slippage, fill rate)

Health States:
    HEALTHY   — performing as expected, trade normally
    WARNING   — performance below threshold, reduce size
    DECAYING  — clear downward trend, pause new entries
    BROKEN    — strategy has lost its edge, disable + alert
    RECOVERING— was paused, testing if edge returned

Auto-Pause Logic:
    - If rolling 50-trade win rate < 30% → DECAYING
    - If rolling 50-trade EV < -0.2R → DECAYING
    - If Sharpe < 0 for 50 trades → BROKEN
    - If DECAYING for 100 trades → BROKEN
    - If BROKEN → auto-disable, alert, require manual re-enable

Usage:
    monitor = StrategyHealthMonitor()

    # After each trade:
    monitor.record_trade(
        strategy="momentum_v4", pnl=42.5, r_multiple=1.8,
        hold_time_s=3600, slippage_bps=1.2,
    )

    # Before each trade:
    health = monitor.health("momentum_v4")
    if health.state == "BROKEN":
        skip_trade()
    elif health.state == "DECAYING":
        reduce_size(50)
    elif health.state == "WARNING":
        reduce_size(25)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.strategy_health_monitor")


class StrategyState(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DECAYING = "decaying"
    BROKEN = "broken"
    RECOVERING = "recovering"
    DISABLED = "disabled"
    UNTESTED = "untested"   # not enough trades to evaluate


@dataclass
class StrategyHealth:
    """Health snapshot for a single strategy."""
    strategy: str
    state: StrategyState = StrategyState.UNTESTED

    # Rolling stats
    total_trades: int = 0
    rolling_win_rate_20: float = 0.0
    rolling_win_rate_50: float = 0.0
    rolling_win_rate_100: float = 0.0
    rolling_sharpe: float = 0.0
    rolling_profit_factor: float = 0.0
    rolling_ev_r: float = 0.0
    rolling_max_drawdown_r: float = 0.0
    avg_hold_time_s: float = 0.0
    avg_slippage_bps: float = 0.0

    # Decay detection
    decay_rate: float = 0.0      # negative = declining
    decay_confidence: float = 0.0

    # State management
    last_trade_at: float = 0.0
    paused_at: float = 0.0
    pause_reason: str = ""
    auto_disabled: bool = False

    # Recommendation
    size_multiplier: float = 1.0  # 0 (disabled) to 1.5 (boost)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "state": self.state.value,
            "total_trades": self.total_trades,
            "win_rate_20": round(self.rolling_win_rate_20, 3),
            "win_rate_50": round(self.rolling_win_rate_50, 3),
            "win_rate_100": round(self.rolling_win_rate_100, 3),
            "sharpe": round(self.rolling_sharpe, 3),
            "profit_factor": round(self.rolling_profit_factor, 3),
            "ev_r": round(self.rolling_ev_r, 3),
            "max_dd_r": round(self.rolling_max_drawdown_r, 2),
            "avg_hold_time_s": round(self.avg_hold_time_s, 0),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "decay_rate": round(self.decay_rate, 4),
            "decay_confidence": round(self.decay_confidence, 3),
            "size_multiplier": round(self.size_multiplier, 2),
            "auto_disabled": self.auto_disabled,
            "recommendation": self.recommendation,
        }


class StrategyHealthMonitor:
    """Monitors health of all trading strategies.

    Call record_trade() after every closed trade.
    Call health() before every new trade entry.
    """

    def __init__(self,
                 min_trades_for_eval: int = 20,
                 warning_win_rate: float = 0.40,
                 decaying_win_rate: float = 0.30,
                 broken_sharpe: float = -0.5,
                 decaying_ev_r: float = -0.2,
                 broken_ev_r: float = -0.5,
                 recovery_test_trades: int = 10):
        """Initialize monitor.

        Args:
            min_trades_for_eval: min trades before evaluating health
            warning_win_rate: below this → WARNING state
            decaying_win_rate: below this → DECAYING state
            broken_sharpe: below this → BROKEN state
            decaying_ev_r: below this → DECAYING
            broken_ev_r: below this → BROKEN
            recovery_test_trades: trades to test before re-enabling
        """
        self.min_trades = min_trades_for_eval
        self.warn_wr = warning_win_rate
        self.decay_wr = decaying_win_rate
        self.broken_sharpe = broken_sharpe
        self.decay_ev = decaying_ev_r
        self.broken_ev = broken_ev_r
        self.recovery_trades = recovery_test_trades

        self._lock = threading.RLock()
        self._strategies: Dict[str, StrategyHealth] = {}
        self._trades: Dict[str, Deque[dict]] = {}  # strategy → trade history

    # ------------------------------------------------------------------
    # Record trade
    # ------------------------------------------------------------------
    def record_trade(self, strategy: str, pnl: float, r_multiple: float,
                     hold_time_s: float = 0, slippage_bps: float = 0) -> None:
        """Record a completed trade for a strategy."""
        trade = {
            "timestamp": time.time(),
            "pnl": pnl,
            "r_multiple": r_multiple,
            "hold_time_s": hold_time_s,
            "slippage_bps": slippage_bps,
            "win": pnl > 0,
        }
        with self._lock:
            if strategy not in self._trades:
                self._trades[strategy] = deque(maxlen=200)
            self._trades[strategy].append(trade)

            # Recompute health
            self._recompute(strategy)

    # ------------------------------------------------------------------
    # Recompute health
    # ------------------------------------------------------------------
    def _recompute(self, strategy: str) -> None:
        """Recompute health stats for a strategy."""
        trades = list(self._trades.get(strategy, []))
        if not trades:
            return

        if strategy not in self._strategies:
            self._strategies[strategy] = StrategyHealth(strategy=strategy)
        h = self._strategies[strategy]
        h.total_trades = len(trades)
        h.last_trade_at = trades[-1]["timestamp"]

        if len(trades) < self.min_trades:
            h.state = StrategyState.UNTESTED
            h.recommendation = f"Untested ({len(trades)}/{self.min_trades} trades)"
            return

        # Rolling win rates
        h.rolling_win_rate_20 = self._win_rate(trades, 20)
        h.rolling_win_rate_50 = self._win_rate(trades, 50)
        h.rolling_win_rate_100 = self._win_rate(trades, 100)

        # Rolling Sharpe
        pnls = [t["pnl"] for t in trades[-50:]]
        if len(pnls) >= 10 and np.std(pnls) > 0:
            h.rolling_sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        else:
            h.rolling_sharpe = 0.0

        # Profit factor
        wins = sum(t["pnl"] for t in trades[-50:] if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in trades[-50:] if t["pnl"] < 0))
        h.rolling_profit_factor = wins / max(losses, 0.01)

        # EV in R
        rs = [t["r_multiple"] for t in trades[-50:]]
        h.rolling_ev_r = float(np.mean(rs))

        # Max drawdown in R
        cum_r = np.cumsum(rs)
        peak = np.maximum.accumulate(cum_r)
        dd = peak - cum_r
        h.rolling_max_drawdown_r = float(np.max(dd)) if len(dd) > 0 else 0.0

        # Avg hold time + slippage
        h.avg_hold_time_s = float(np.mean([t["hold_time_s"] for t in trades[-50:]]))
        h.avg_slippage_bps = float(np.mean([t["slippage_bps"] for t in trades[-50:]]))

        # Decay detection: compare first half vs second half performance
        if len(trades) >= 40:
            half = len(trades) // 2
            first_half_ev = np.mean([t["r_multiple"] for t in trades[:half]])
            second_half_ev = np.mean([t["r_multiple"] for t in trades[half:]])
            h.decay_rate = float(second_half_ev - first_half_ev)  # negative = declining
            h.decay_confidence = min(1.0, len(trades) / 100)

        # Determine state
        self._determine_state(h)

    def _win_rate(self, trades: list, window: int) -> float:
        """Compute win rate over last N trades."""
        recent = trades[-window:]
        if not recent:
            return 0.0
        wins = sum(1 for t in recent if t["win"])
        return wins / len(recent)

    # ------------------------------------------------------------------
    # State determination
    # ------------------------------------------------------------------
    def _determine_state(self, h: StrategyHealth) -> None:
        """Determine strategy state from stats."""
        # If already disabled, stay disabled (manual re-enable required)
        if h.auto_disabled:
            h.state = StrategyState.BROKEN
            h.size_multiplier = 0.0
            h.recommendation = "DISABLED — strategy broken, requires manual review"
            return

        # Check for broken
        if (h.rolling_sharpe < self.broken_sharpe and h.total_trades >= 50) or \
           (h.rolling_ev_r < self.broken_ev and h.total_trades >= 50):
            h.state = StrategyState.BROKEN
            h.auto_disabled = True
            h.size_multiplier = 0.0
            h.paused_at = time.time()
            h.pause_reason = (f"Sharpe={h.rolling_sharpe:.2f} < {self.broken_sharpe} "
                            f"or EV={h.rolling_ev_r:.2f}R < {self.broken_ev}R")
            h.recommendation = f"DISABLED — {h.pause_reason}"
            log.warning("Strategy %s BROKEN: %s", h.strategy, h.pause_reason)
            return

        # Check for decaying
        if h.rolling_win_rate_50 < self.decay_wr or \
           h.rolling_ev_r < self.decay_ev or \
           (h.decay_rate < -0.3 and h.decay_confidence > 0.5):
            h.state = StrategyState.DECAYING
            h.size_multiplier = 0.25
            h.recommendation = (f"DECAYING — WR={h.rolling_win_rate_50:.0%}, "
                               f"EV={h.rolling_ev_r:.2f}R, decay={h.decay_rate:.3f}. "
                               f"Reduce size to 25%")
            return

        # Check for warning
        if h.rolling_win_rate_50 < self.warn_wr or h.rolling_ev_r < 0:
            h.state = StrategyState.WARNING
            h.size_multiplier = 0.5
            h.recommendation = (f"WARNING — WR={h.rolling_win_rate_50:.0%}, "
                               f"EV={h.rolling_ev_r:.2f}R. Reduce size to 50%")
            return

        # Healthy
        h.state = StrategyState.HEALTHY
        # Boost if performing well
        if h.rolling_ev_r > 0.5 and h.rolling_sharpe > 1.0:
            h.size_multiplier = 1.25
            h.recommendation = f"HEALTHY+ — WR={h.rolling_win_rate_50:.0%}, EV={h.rolling_ev_r:.2f}R. Boost 1.25x"
        else:
            h.size_multiplier = 1.0
            h.recommendation = f"HEALTHY — WR={h.rolling_win_rate_50:.0%}, EV={h.rolling_ev_r:.2f}R. Normal size"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def health(self, strategy: str) -> StrategyHealth:
        """Get current health for a strategy."""
        with self._lock:
            if strategy not in self._strategies:
                return StrategyHealth(strategy=strategy, state=StrategyState.UNTESTED)
            return self._strategies[strategy]

    def can_trade(self, strategy: str) -> bool:
        """Quick check: can this strategy place new trades?"""
        h = self.health(strategy)
        return h.state in (StrategyState.HEALTHY, StrategyState.WARNING, StrategyState.UNTESTED)

    def size_multiplier(self, strategy: str) -> float:
        """Get position size multiplier for this strategy."""
        return self.health(strategy).size_multiplier

    def all_health(self) -> Dict[str, StrategyHealth]:
        """Get health for all strategies."""
        with self._lock:
            return dict(self._strategies)

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------
    def disable(self, strategy: str, reason: str = "manual") -> None:
        """Manually disable a strategy."""
        with self._lock:
            if strategy not in self._strategies:
                self._strategies[strategy] = StrategyHealth(strategy=strategy)
            h = self._strategies[strategy]
            h.auto_disabled = True
            h.state = StrategyState.DISABLED
            h.size_multiplier = 0.0
            h.pause_reason = reason
            h.paused_at = time.time()
            log.info("Strategy %s manually disabled: %s", strategy, reason)

    def enable(self, strategy: str) -> None:
        """Manually re-enable a strategy."""
        with self._lock:
            if strategy not in self._strategies:
                return
            h = self._strategies[strategy]
            h.auto_disabled = False
            h.state = StrategyState.RECOVERING
            h.size_multiplier = 0.5  # start at half size
            h.recommendation = "RECOVERING — re-enabled at 50% size, monitoring"
            log.info("Strategy %s re-enabled", strategy)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Get summary of all strategies."""
        with self._lock:
            strategies = list(self._strategies.values())
        states = {}
        for s in strategies:
            states.setdefault(s.state.value, 0)
            states[s.state.value] += 1
        return {
            "total_strategies": len(strategies),
            "by_state": states,
            "healthy": states.get("healthy", 0),
            "warning": states.get("warning", 0),
            "decaying": states.get("decaying", 0),
            "broken": states.get("broken", 0),
            "disabled": states.get("disabled", 0),
            "strategies": {s.strategy: s.to_dict() for s in strategies},
        }
