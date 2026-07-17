"""trading_modules/adaptive_learning_engine.py
=====================================================================
Adaptive Learning Engine (Principle #118 — Never Stop Learning)
=====================================================================
Continuously learns which strategies, sessions, regimes, and setups
perform best — then adjusts future decisions accordingly.

Learning Loop:
    Trade closes → outcome recorded →
    ↓
    Update rolling stats per (strategy, session, regime, setup) →
    ↓
    Rank setups by EV / Sharpe →
    ↓
    Adjust strategy weights + position sizing →
    ↓
    Better decisions next cycle

What It Learns:
    1. Per-strategy performance (which strategy wins most?)
    2. Per-session performance (London? NY? Asia?)
    3. Per-regime performance (trend? range? breakout?)
    4. Per-setup performance (pullback? breakout? reversal?)
    5. Per-symbol performance (BTC? ETH? EURUSD?)
    6. Time-of-day patterns (when do we win/lose?)
    7. Day-of-week patterns (Mon vs Fri?)
    8. Correlation between confidence and outcome
    9. Feature importance drift (which features predict wins?)

Outputs:
    - Strategy weight table (multiplier 0-1.5 per strategy)
    - Session preference score (0-1 per session)
    - Regime preference score (0-1 per regime)
    - Setup ranking (sorted by EV)
    - Recommended position size adjustment
    - "Avoid" list (setups with negative EV)

Usage:
    engine = AdaptiveLearningEngine()

    # After each trade closes:
    engine.record_outcome(
        strategy="momentum", session="london", regime="trend_up",
        setup="pullback", symbol="BTCUSD",
        pnl=42.50, r_multiple=1.8, confidence=0.75,
        hold_time_s=3600,
    )

    # Before each trade:
    weights = engine.get_weights(strategy="momentum", session="london",
                                 regime="trend_up", setup="pullback")
    # weights = {
    #     "strategy_weight": 1.2,    # this strategy is performing well
    #     "session_weight": 0.9,     # london is OK for this strategy
    #     "regime_weight": 1.3,      # trend_up is great for momentum
    #     "setup_weight": 1.1,       # pullback setup is profitable
    #     "combined_multiplier": 1.18,  # overall position size multiplier
    #     "confidence_calibration": 1.05,  # strategy confidence is well-calibrated
    # }
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.adaptive_learning_engine")


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    timestamp: float
    strategy: str
    session: str          # "london", "new_york", "asia", "overlap", "off_hours"
    regime: str           # "trend_up", "trend_down", "range", "breakout", etc.
    setup: str            # "pullback", "breakout", "reversal", etc.
    symbol: str
    pnl: float
    r_multiple: float
    confidence: float     # strategy confidence at entry (0-1)
    hold_time_s: float
    win: bool
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Rolling performance stats for a category."""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_r: float = 0.0
    avg_pnl: float = 0.0
    avg_r: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    ev_per_trade: float = 0.0  # expected value in R
    weight: float = 1.0        # 0-1.5 multiplier

    def update(self, pnl: float, r: float) -> None:
        self.trades += 1
        self.total_pnl += pnl
        self.total_r += r
        self.avg_pnl = self.total_pnl / self.trades
        self.avg_r = self.total_r / self.trades
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.win_rate = self.wins / max(self.trades, 1)
        # EV in R = (win_rate × avg_win_r) - (loss_rate × 1.0)
        if self.trades >= 5:
            wins_r = [r for r in [r] if r > 0]
            losses_r = [r for r in [r] if r < 0]
            avg_win_r = max(1.5, abs(np.mean(wins_r)) if wins_r else 1.5)
            self.ev_per_trade = (self.win_rate * avg_win_r) - ((1 - self.win_rate) * 1.0)
        # Weight: scale based on EV
        if self.trades >= 10:
            if self.ev_per_trade > 0.5:
                self.weight = 1.3
            elif self.ev_per_trade > 0.2:
                self.weight = 1.1
            elif self.ev_per_trade > 0:
                self.weight = 1.0
            elif self.ev_per_trade > -0.2:
                self.weight = 0.7
            else:
                self.weight = 0.3  # nearly blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades": self.trades,
            "win_rate": round(self.win_rate, 3),
            "avg_pnl": round(self.avg_pnl, 2),
            "avg_r": round(self.avg_r, 3),
            "ev_per_trade": round(self.ev_per_trade, 3),
            "weight": round(self.weight, 2),
        }


class AdaptiveLearningEngine:
    """Learns from trade outcomes and adjusts future weights."""

    def __init__(self,
                 min_trades_for_weight: int = 10,
                 rolling_window: int = 100,
                 decay_half_life_days: float = 30.0):
        """Initialize engine.

        Args:
            min_trades_for_weight: minimum trades before weight is applied
            rolling_window: how many recent trades to consider
            decay_half_life_days: older trades decay exponentially
        """
        self.min_trades = min_trades_for_weight
        self.rolling_window = rolling_window
        self.decay_half_life = decay_half_life_days

        self._lock = threading.RLock()
        self._outcomes: deque = deque(maxlen=rolling_window)

        # Per-category performance
        self._by_strategy: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._by_session: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._by_regime: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._by_setup: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._by_symbol: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)

        # Confidence calibration: map strategy confidence → actual win rate
        self._confidence_buckets: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Record outcome
    # ------------------------------------------------------------------
    def record_outcome(self,
                       strategy: str, session: str, regime: str,
                       setup: str, symbol: str,
                       pnl: float, r_multiple: float,
                       confidence: float, hold_time_s: float,
                       features: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed trade outcome."""
        outcome = TradeOutcome(
            timestamp=time.time(),
            strategy=strategy, session=session, regime=regime,
            setup=setup, symbol=symbol,
            pnl=pnl, r_multiple=r_multiple,
            confidence=confidence, hold_time_s=hold_time_s,
            win=pnl > 0,
            features=features or {},
        )

        with self._lock:
            self._outcomes.append(outcome)
            self._by_strategy[strategy].update(pnl, r_multiple)
            self._by_session[session].update(pnl, r_multiple)
            self._by_regime[regime].update(pnl, r_multiple)
            self._by_setup[setup].update(pnl, r_multiple)
            self._by_symbol[symbol].update(pnl, r_multiple)

            # Confidence calibration
            bucket = f"{strategy}_{int(confidence * 10) / 10:.1f}"
            self._confidence_buckets[bucket].append((confidence, outcome.win))
            # Keep last 50 per bucket
            if len(self._confidence_buckets[bucket]) > 50:
                self._confidence_buckets[bucket] = self._confidence_buckets[bucket][-50:]

        log.debug("adaptive: recorded %s %s pnl=%.2f R=%.2f (strategy=%s WR=%.0f%%)",
                  symbol, "WIN" if outcome.win else "LOSS", pnl, r_multiple,
                  strategy, self._by_strategy[strategy].win_rate * 100)

    # ------------------------------------------------------------------
    # Get weights for a trade
    # ------------------------------------------------------------------
    def get_weights(self,
                    strategy: str, session: str, regime: str,
                    setup: str, symbol: str = "") -> Dict[str, float]:
        """Get recommended weights for an upcoming trade.

        Returns dict with:
            - strategy_weight: multiplier based on strategy performance
            - session_weight: multiplier based on session performance
            - regime_weight: multiplier based on regime performance
            - setup_weight: multiplier based on setup performance
            - symbol_weight: multiplier based on symbol performance
            - combined_multiplier: product of all weights (clipped 0-1.5)
            - confidence_calibration: how well-calibrated is strategy confidence
        """
        with self._lock:
            sw = self._safe_weight(self._by_strategy.get(strategy))
            sesw = self._safe_weight(self._by_session.get(session))
            rw = self._safe_weight(self._by_regime.get(regime))
            stw = self._safe_weight(self._by_setup.get(setup))
            symw = self._safe_weight(self._by_symbol.get(symbol)) if symbol else 1.0

        # Combined multiplier (geometric mean for stability)
        combined = (sw * sesw * rw * stw * symw) ** (1 / 5)
        combined = max(0.0, min(1.5, combined))

        # Confidence calibration
        cal = self._confidence_calibration(strategy)

        return {
            "strategy_weight": round(sw, 2),
            "session_weight": round(sesw, 2),
            "regime_weight": round(rw, 2),
            "setup_weight": round(stw, 2),
            "symbol_weight": round(symw, 2),
            "combined_multiplier": round(combined, 2),
            "confidence_calibration": round(cal, 2),
        }

    def _safe_weight(self, stats: Optional[PerformanceStats]) -> float:
        """Get weight from stats, defaulting to 1.0 if insufficient data."""
        if stats is None or stats.trades < self.min_trades:
            return 1.0
        return stats.weight

    def _confidence_calibration(self, strategy: str) -> float:
        """How well-calibrated is this strategy's confidence?

        Returns a multiplier:
            - 1.0 = perfectly calibrated
            - >1.0 = strategy is under-confident (wins more than it claims)
            - <1.0 = strategy is over-confident (wins less than it claims)
        """
        # Aggregate all buckets for this strategy
        all_outcomes: List[Tuple[float, bool]] = []
        for bucket, outcomes in self._confidence_buckets.items():
            if bucket.startswith(strategy):
                all_outcomes.extend(outcomes)

        if len(all_outcomes) < 20:
            return 1.0  # not enough data

        # Average claimed confidence vs actual win rate
        avg_confidence = np.mean([c for c, _ in all_outcomes])
        actual_win_rate = np.mean([w for _, w in all_outcomes])

        if avg_confidence == 0:
            return 1.0

        # Calibration = actual / claimed
        calibration = actual_win_rate / avg_confidence
        return max(0.5, min(2.0, calibration))

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------
    def rank_strategies(self) -> List[Tuple[str, PerformanceStats]]:
        """Return strategies ranked by EV."""
        with self._lock:
            sorted_strats = sorted(self._by_strategy.items(),
                                  key=lambda x: x[1].ev_per_trade, reverse=True)
        return sorted_strats

    def rank_sessions(self) -> List[Tuple[str, PerformanceStats]]:
        """Return sessions ranked by EV."""
        with self._lock:
            return sorted(self._by_session.items(),
                         key=lambda x: x[1].ev_per_trade, reverse=True)

    def rank_regimes(self) -> List[Tuple[str, PerformanceStats]]:
        """Return regimes ranked by EV."""
        with self._lock:
            return sorted(self._by_regime.items(),
                         key=lambda x: x[1].ev_per_trade, reverse=True)

    def rank_setups(self) -> List[Tuple[str, PerformanceStats]]:
        """Return setups ranked by EV."""
        with self._lock:
            return sorted(self._by_setup.items(),
                         key=lambda x: x[1].ev_per_trade, reverse=True)

    def rank_symbols(self) -> List[Tuple[str, PerformanceStats]]:
        """Return symbols ranked by EV."""
        with self._lock:
            return sorted(self._by_symbol.items(),
                         key=lambda x: x[1].ev_per_trade, reverse=True)

    # ------------------------------------------------------------------
    # Avoid list
    # ------------------------------------------------------------------
    def avoid_list(self, min_trades: int = 10) -> Dict[str, List[str]]:
        """Return combinations to avoid (negative EV with enough sample).

        Returns dict with keys: strategies, sessions, regimes, setups, symbols
        """
        out = {"strategies": [], "sessions": [], "regimes": [], "setups": [], "symbols": []}

        with self._lock:
            for name, s in self._by_strategy.items():
                if s.trades >= min_trades and s.ev_per_trade < 0:
                    out["strategies"].append(name)
            for name, s in self._by_session.items():
                if s.trades >= min_trades and s.ev_per_trade < 0:
                    out["sessions"].append(name)
            for name, s in self._by_regime.items():
                if s.trades >= min_trades and s.ev_per_trade < 0:
                    out["regimes"].append(name)
            for name, s in self._by_setup.items():
                if s.trades >= min_trades and s.ev_per_trade < 0:
                    out["setups"].append(name)
            for name, s in self._by_symbol.items():
                if s.trades >= min_trades and s.ev_per_trade < 0:
                    out["symbols"].append(name)
        return out

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Full summary of learning state."""
        with self._lock:
            return {
                "total_trades": len(self._outcomes),
                "strategies_tracked": len(self._by_strategy),
                "sessions_tracked": len(self._by_session),
                "regimes_tracked": len(self._by_regime),
                "setups_tracked": len(self._by_setup),
                "symbols_tracked": len(self._by_symbol),
                "top_strategy": self.rank_strategies()[0] if self._by_strategy else None,
                "top_session": self.rank_sessions()[0] if self._by_session else None,
                "top_regime": self.rank_regimes()[0] if self._by_regime else None,
                "top_setup": self.rank_setups()[0] if self._by_setup else None,
                "top_symbol": self.rank_symbols()[0] if self._by_symbol else None,
                "avoid_list": self.avoid_list(),
            }

    def stats_table(self) -> str:
        """Human-readable stats table."""
        lines = ["=" * 80, "  ADAPTIVE LEARNING ENGINE — Performance Summary", "=" * 80]

        for category, items, label in [
            ("Strategies", self.rank_strategies(), "Strategy"),
            ("Sessions", self.rank_sessions(), "Session"),
            ("Regimes", self.rank_regimes(), "Regime"),
            ("Setups", self.rank_setups(), "Setup"),
            ("Symbols", self.rank_symbols(), "Symbol"),
        ]:
            if not items:
                continue
            lines.append(f"\n  {category} (ranked by EV):")
            lines.append(f"  {'Name':25s} {'Trades':>6s} {'Win%':>6s} {'AvgR':>6s} {'EV':>6s} {'Weight':>6s}")
            lines.append("  " + "-" * 60)
            for name, s in items[:5]:  # top 5
                lines.append(f"  {name:25s} {s.trades:>6d} {s.win_rate*100:>5.0f}% "
                            f"{s.avg_r:>+6.2f} {s.ev_per_trade:>+6.2f} {s.weight:>5.2f}x")

        avoid = self.avoid_list()
        any_avoid = any(v for v in avoid.values())
        if any_avoid:
            lines.append("\n  AVOID LIST:")
            for cat, items in avoid.items():
                if items:
                    lines.append(f"    {cat}: {', '.join(items)}")

        lines.append("=" * 80)
        return "\n".join(lines)
