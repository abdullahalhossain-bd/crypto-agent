"""trading_modules/adaptive_strategy_router.py
=====================================================================
Adaptive Strategy Router (Principle #195)
=====================================================================
Automatically switches the active trading strategy based on the current
market regime. Different regimes require different approaches.

Regime → Strategy Mapping:
    TREND_UP       → Trend Following (ride the trend)
    TREND_DOWN     → Trend Following (short side)
    RANGE          → Mean Reversion (fade extremes)
    BREAKOUT       → Breakout Strategy (catch the move)
    CRISIS         → Capital Protection (minimal/defensive)
    HIGH_VOL       → Reduce Size + Scalping (quick in/out)
    LOW_VOL        → Range Strategy + Accumulation
    NEWS           → Low Risk Mode (wait or minimal)
    RECOVERY       → Accumulation (gradual longs)
    UNKNOWN        → Observe & Wait (no new trades)

Strategy Switching Logic:
    1. Detect current regime (from MarketCycleEngine + MarketContextEngine)
    2. Look up recommended strategy for this regime
    3. If current strategy ≠ recommended, switch
    4. Switch = gracefully close current strategy's positions + activate new
    5. Log every switch for audit trail

Usage:
    router = AdaptiveStrategyRouter()

    # Register strategies
    router.register_strategy("trend_following", TrendStrategy())
    router.register_strategy("mean_reversion", MeanRevStrategy())
    router.register_strategy("breakout", BreakoutStrategy())

    # Each cycle:
    recommended = router.route(regime="range", volatility_regime="normal")
    # recommended = {
    #     "strategy": "mean_reversion",
    #     "action": "switch",
    #     "reason": "Range regime → mean reversion optimal",
    #     "position_size_multiplier": 0.8,
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.adaptive_strategy_router")


class StrategyType(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    CAPITAL_PROTECTION = "capital_protection"
    ACCUMULATION = "accumulation"
    OBSERVE_WAIT = "observe_wait"


# Regime → Strategy mapping
REGIME_STRATEGY_MAP: Dict[str, StrategyType] = {
    "trend_up": StrategyType.TREND_FOLLOWING,
    "trend_down": StrategyType.TREND_FOLLOWING,
    "range": StrategyType.MEAN_REVERSION,
    "breakout": StrategyType.BREAKOUT,
    "crisis": StrategyType.CAPITAL_PROTECTION,
    "recovery": StrategyType.ACCUMULATION,
    "consolidation": StrategyType.MEAN_REVERSION,
    "expansion": StrategyType.TREND_FOLLOWING,
    "peak": StrategyType.CAPITAL_PROTECTION,
    "decline": StrategyType.CAPITAL_PROTECTION,
    "unknown": StrategyType.OBSERVE_WAIT,
}


# Volatility regime adjustments
VOLATILITY_ADJUSTMENTS: Dict[str, float] = {
    "low": 1.0,
    "normal": 1.0,
    "high": 0.5,     # reduce size 50%
    "extreme": 0.2,  # reduce size 80%
}


@dataclass
class StrategySwitch:
    """Record of a strategy switch."""
    timestamp: str = ""
    from_strategy: str = ""
    to_strategy: str = ""
    reason: str = ""
    regime: str = ""
    position_size_mult: float = 1.0


@dataclass
class RouteResult:
    """Strategy routing result."""
    strategy: StrategyType = StrategyType.OBSERVE_WAIT
    action: str = "hold"           # hold, switch, activate, deactivate
    reason: str = ""
    regime: str = "unknown"
    volatility_regime: str = "normal"
    position_size_multiplier: float = 1.0
    previous_strategy: Optional[StrategyType] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "action": self.action,
            "reason": self.reason,
            "regime": self.regime,
            "volatility_regime": self.volatility_regime,
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "previous_strategy": self.previous_strategy.value if self.previous_strategy else None,
            "description": self.description,
        }


class AdaptiveStrategyRouter:
    """Routes to the optimal strategy based on market regime."""

    def __init__(self,
                 default_strategy: StrategyType = StrategyType.TREND_FOLLOWING,
                 min_switch_interval_bars: int = 10):
        """Initialize router.

        Args:
            default_strategy: starting strategy
            min_switch_interval_bars: minimum bars between switches (prevents flapping)
        """
        self.current_strategy = default_strategy
        self.min_switch_interval = min_switch_interval_bars
        self._strategies: Dict[str, Any] = {}
        self._switches: List[StrategySwitch] = []
        self._bars_since_switch = 0

    def register_strategy(self, name: str, strategy: Any) -> None:
        """Register a strategy implementation."""
        self._strategies[name] = strategy
        log.info("router: registered strategy '%s'", name)

    # ------------------------------------------------------------------
    # Route to optimal strategy
    # ------------------------------------------------------------------
    def route(self,
              regime: str = "unknown",
              volatility_regime: str = "normal",
              news_pending: bool = False,
              market_cycle: str = "unknown") -> RouteResult:
        """Determine the optimal strategy for current conditions.

        Args:
            regime: current market regime
            volatility_regime: low/normal/high/extreme
            news_pending: is high-impact news pending?
            market_cycle: current market cycle phase

        Returns:
            RouteResult with recommended strategy + action
        """
        self._bars_since_switch += 1

        result = RouteResult(
            regime=regime,
            volatility_regime=volatility_regime,
            previous_strategy=self.current_strategy,
        )

        # === News override → low risk mode ===
        if news_pending:
            recommended = StrategyType.CAPITAL_PROTECTION
            result.reason = "News pending → capital protection mode"
        # === Crisis override → capital protection ===
        elif regime == "crisis" or volatility_regime == "extreme":
            recommended = StrategyType.CAPITAL_PROTECTION
            result.reason = f"Crisis/extreme vol → capital protection"
        # === Use regime map ===
        else:
            recommended = REGIME_STRATEGY_MAP.get(regime, StrategyType.OBSERVE_WAIT)
            result.reason = f"{regime} regime → {recommended.value}"

        # === Position size adjustment ===
        result.position_size_multiplier = VOLATILITY_ADJUSTMENTS.get(volatility_regime, 1.0)
        if recommended == StrategyType.CAPITAL_PROTECTION:
            result.position_size_multiplier *= 0.3  # extra reduction
        if recommended == StrategyType.OBSERVE_WAIT:
            result.position_size_multiplier = 0.0  # no new trades

        # === Determine action ===
        if recommended != self.current_strategy:
            if self._bars_since_switch < self.min_switch_interval:
                result.action = "hold"  # too soon to switch
                result.reason = (
                    f"Want to switch to {recommended.value} but min interval not met "
                    f"({self._bars_since_switch}/{self.min_switch_interval} bars)"
                )
                result.strategy = self.current_strategy
            else:
                result.action = "switch"
                result.strategy = recommended
                self._switch(recommended, result.reason, regime,
                            result.position_size_multiplier)
        else:
            result.action = "hold"
            result.strategy = self.current_strategy

        result.description = self._describe(result)
        return result

    def _switch(self, new_strategy: StrategyType, reason: str,
                regime: str, size_mult: float) -> None:
        """Perform the strategy switch."""
        switch = StrategySwitch(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            from_strategy=self.current_strategy.value,
            to_strategy=new_strategy.value,
            reason=reason,
            regime=regime,
            position_size_mult=size_mult,
        )
        self._switches.append(switch)
        self.current_strategy = new_strategy
        self._bars_since_switch = 0
        log.info("router: switched %s → %s (%s)",
                switch.from_strategy, switch.to_strategy, reason)

    def _describe(self, r: RouteResult) -> str:
        """Generate description."""
        return (
            f"Strategy: {r.strategy.value} (action: {r.action}). "
            f"Regime: {r.regime}, vol: {r.volatility_regime}. "
            f"Size: {r.position_size_multiplier:.0%}x. "
            f"Reason: {r.reason}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def current(self) -> StrategyType:
        """Get current active strategy."""
        return self.current_strategy

    def switch_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get switch history."""
        return [
            {
                "timestamp": s.timestamp,
                "from": s.from_strategy,
                "to": s.to_strategy,
                "reason": s.reason,
                "regime": s.regime,
                "size_mult": round(s.position_size_mult, 2),
            }
            for s in self._switches[-last_n:]
        ]

    def stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        from collections import Counter
        strategy_counts = Counter(s.to_strategy for s in self._switches)
        return {
            "current_strategy": self.current_strategy.value,
            "total_switches": len(self._switches),
            "bars_since_switch": self._bars_since_switch,
            "strategy_frequency": dict(strategy_counts),
            "registered_strategies": list(self._strategies.keys()),
        }
