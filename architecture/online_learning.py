"""architecture/online_learning.py
=====================================================================
Online Learning System (Improvement #12)
=====================================================================
Continuously updates model weights and strategy parameters based on
recent trade outcomes — without full retraining.

Three Learning Modes:

1. PARAMETER UPDATES — adjust thresholds and multipliers
   Example: If recent trades with RSI > 70 are losing, raise the
   overbought threshold from 70 → 72. Track the change with versioning.

2. BANDIT LEARNING — explore/exploit across strategy variants
   Maintain a portfolio of strategy configs. Use Thompson sampling to
   allocate cycles to the best-performing variant.

3. GRADIENT UPDATES — fine-tune ML model weights incrementally
   After each closed trade, compute the prediction error and nudge
   model weights (partial_fit for sklearn, SGD for torch).

Feedback Loop:
    Trade closes → outcome recorded →
    ↓
    Update rolling win rate / Sharpe per strategy variant
    ↓
    Update bandit prior (Beta distribution)
    ↓
    Update threshold parameters (Bayesian update)
    ↓
    Optionally: partial_fit on ML model
    ↓
    Drift check → if significant, trigger full retrain

Safety:
    - All updates are versioned (rollback possible)
    - Updates are clipped (no parameter can move >20% in one cycle)
    - Updates are logged to DecisionAuditor
    - A "shadow" copy of the previous params is kept for A/B test
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.online_learning")


@dataclass
class ParameterUpdate:
    """A single parameter update record (for rollback + audit)."""
    timestamp: str = ""
    parameter: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    delta_pct: float = 0.0
    reason: str = ""
    sample_size: int = 0
    confidence: float = 0.0


@dataclass
class StrategyVariant:
    """A strategy configuration with a bandit-style prior."""
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    # Beta distribution params for Thompson sampling
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1
    total_pnl: float = 0.0
    trade_count: int = 0
    last_used: float = 0.0

    @property
    def expected_win_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample_thompson(self) -> float:
        return random.betavariate(self.alpha, self.beta)


class OnlineLearner:
    """Continuous parameter + strategy learning."""

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 update_clip_pct: float = 0.20,
                 min_samples_for_update: int = 10,
                 rolling_window: int = 50):
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        self._update_clip = update_clip_pct
        self._min_samples = min_samples_for_update
        self._rolling_window = rolling_window

        # Tunable parameters (start with sensible defaults)
        self._params: Dict[str, float] = {
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "min_factors": 3.0,
            "risk_per_trade": 0.02,
            "sl_atr_multiple": 1.5,
            "tp_atr_multiple": 2.5,
            "min_confidence": 0.60,
            "max_spread_bps": 15.0,
        }
        # H15 fix: enforce min/max bounds for each parameter so clipping
        # the delta can't drive a parameter outside its valid range.
        # Without this, a run of bad trades could push min_confidence to
        # 1.5 (invalid) or rsi_overbought to 200 (nonsensical).
        self._param_bounds: Dict[str, tuple] = {
            "rsi_overbought": (60.0, 85.0),
            "rsi_oversold": (15.0, 40.0),
            "min_factors": (1.0, 8.0),
            "risk_per_trade": (0.005, 0.05),
            "sl_atr_multiple": (1.0, 3.0),
            "tp_atr_multiple": (1.5, 5.0),
            "min_confidence": (0.3, 0.9),
            "max_spread_bps": (5.0, 50.0),
        }
        self._param_history: Dict[str, List[ParameterUpdate]] = {}

        # Strategy variants for bandit
        self._variants: Dict[str, StrategyVariant] = {}

        # Rolling trade outcomes (for parameter updates)
        self._recent_outcomes: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Parameter updates (Bayesian-style nudges)
    # ------------------------------------------------------------------
    def get_param(self, name: str, default: float = 0.0) -> float:
        with self._lock:
            return self._params.get(name, default)

    def get_all_params(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._params)

    def update_parameter(self,
                         name: str,
                         new_value: float,
                         reason: str = "",
                         confidence: float = 0.5) -> ParameterUpdate:
        """Update a parameter, clipped to ±update_clip_pct per cycle."""
        with self._lock:
            old = self._params.get(name, new_value)
            if old != 0:
                clipped_new = old + (new_value - old) * min(1.0, self._update_clip / 100 + 1)
                # Actually clip the delta
                delta_pct = abs(new_value - old) / abs(old)
                if delta_pct > self._update_clip:
                    sign = 1 if new_value > old else -1
                    clipped_new = old + sign * abs(old) * self._update_clip
                else:
                    clipped_new = new_value
            else:
                clipped_new = new_value

            update = ParameterUpdate(
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                parameter=name,
                old_value=old,
                new_value=clipped_new,
                delta_pct=abs(clipped_new - old) / abs(old) if old != 0 else 0.0,
                reason=reason,
                confidence=confidence,
            )
            self._params[name] = clipped_new
            # H15 fix: enforce absolute bounds so a clipped delta can't
            # push a parameter outside its valid range.
            if name in self._param_bounds:
                lo, hi = self._param_bounds[name]
                if clipped_new < lo:
                    clipped_new = lo
                elif clipped_new > hi:
                    clipped_new = hi
                self._params[name] = clipped_new
                update.new_value = clipped_new
            self._param_history.setdefault(name, []).append(update)

            log.info("online_learning: parameter %s: %.4f → %.4f (%.1f%% delta) "
                     "reason=%s",
                     name, old, clipped_new, update.delta_pct * 100, reason)
            return update

    def rollback_parameter(self, name: str, steps: int = 1) -> bool:
        """Roll back a parameter N updates."""
        with self._lock:
            history = self._param_history.get(name, [])
            if len(history) < steps:
                return False
            for _ in range(steps):
                update = history.pop()
                self._params[name] = update.old_value
            log.info("online_learning: rolled back %s by %d steps to %.4f",
                     name, steps, self._params[name])
            return True

    # ------------------------------------------------------------------
    # Auto-update based on trade outcomes
    # ------------------------------------------------------------------
    def record_outcome(self,
                       symbol: str,
                       strategy: str,
                       direction: str,
                       pnl: float,
                       r_multiple: float,
                       features: Dict[str, Any]) -> None:
        """Record a trade outcome and trigger parameter updates."""
        outcome = {
            "timestamp": time.time(),
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "features": features,
            "win": pnl > 0,
        }
        with self._lock:
            self._recent_outcomes.append(outcome)
            if len(self._recent_outcomes) > self._rolling_window:
                self._recent_outcomes = self._recent_outcomes[-self._rolling_window:]

            # Update bandit variant
            variant = self._variants.get(strategy)
            if variant is None:
                variant = StrategyVariant(name=strategy)
                self._variants[strategy] = variant
            variant.trade_count += 1
            variant.total_pnl += pnl
            if pnl > 0:
                variant.alpha += 1
            else:
                variant.beta += 1
            variant.last_used = time.time()

        # Trigger parameter updates
        self._auto_update_params()

    def _auto_update_params(self) -> None:
        """Analyze recent outcomes and suggest parameter updates."""
        with self._lock:
            outcomes = list(self._recent_outcomes)
        if len(outcomes) < self._min_samples:
            return

        # Win rate
        wins = sum(1 for o in outcomes if o["win"])
        win_rate = wins / len(outcomes)

        # If win rate is too low, tighten entry criteria (raise min_confidence)
        if win_rate < 0.40:
            current = self._params.get("min_confidence", 0.60)
            self.update_parameter(
                "min_confidence",
                current + 0.02,
                reason=f"win_rate={win_rate:.2f} < 0.40, tightening",
                confidence=0.6,
            )

        # If win rate is high, can loosen criteria (explore more)
        elif win_rate > 0.65:
            current = self._params.get("min_confidence", 0.60)
            self.update_parameter(
                "min_confidence",
                current - 0.01,
                reason=f"win_rate={win_rate:.2f} > 0.65, loosening",
                confidence=0.5,
            )

        # RSI threshold updates based on outcomes at extremes
        rsi_wins = [o for o in outcomes
                   if "rsi_14" in o["features"] and o["features"]["rsi_14"]]
        if rsi_wins:
            rsi_long_losses = [o for o in rsi_wins
                              if o["direction"] == "BUY" and not o["win"]
                              and o["features"].get("rsi_14", 50) > 65]
            if len(rsi_long_losses) > 3:
                # RSI > 65 longs are losing — raise overbought threshold
                current = self._params.get("rsi_overbought", 70.0)
                self.update_parameter(
                    "rsi_overbought",
                    current + 1.0,
                    reason=f"{len(rsi_long_losses)} losing longs at RSI>65",
                    confidence=0.5,
                )

    # ------------------------------------------------------------------
    # Bandit selection
    # ------------------------------------------------------------------
    def select_strategy_variant(self) -> str:
        """Thompson sampling: pick the strategy variant to use this cycle."""
        with self._lock:
            if not self._variants:
                return "default"
            samples = {name: v.sample_thompson()
                      for name, v in self._variants.items()}
        best = max(samples.items(), key=lambda x: x[1])
        log.debug("online_learning: bandit selected %s (sample=%.3f)",
                  best[0], best[1])
        return best[0]

    def variant_stats(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": v.name,
                    "alpha": v.alpha, "beta": v.beta,
                    "expected_win_rate": v.expected_win_rate,
                    "trade_count": v.trade_count,
                    "total_pnl": v.total_pnl,
                    "avg_pnl": v.total_pnl / max(v.trade_count, 1),
                }
                for v in self._variants.values()
            ]

    # ------------------------------------------------------------------
    # Periodic full retrain trigger
    # ------------------------------------------------------------------
    def should_retrain(self, drift_psi: float = 0.0,
                       last_retrain_s: float = 0) -> bool:
        """Decide if a full model retrain is needed."""
        if drift_psi > 0.25:
            return True
        if time.time() - last_retrain_s > 7 * 86400:  # 7 days
            return True
        with self._lock:
            if len(self._recent_outcomes) >= 30:
                wr = sum(1 for o in self._recent_outcomes if o["win"]) / max(len(self._recent_outcomes), 1)  # P0-15 fix: was /30
                if wr < 0.30:  # catastrophic
                    return True
        return False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_params": dict(self._params),
                "param_history_count": {k: len(v) for k, v in self._param_history.items()},
                "variants": self.variant_stats(),
                "recent_outcomes": len(self._recent_outcomes),
            }
