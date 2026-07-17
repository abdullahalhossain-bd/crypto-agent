"""trading_modules/portfolio_allocation_optimizer.py
=====================================================================
Portfolio Allocation Optimizer (Principle #172)
=====================================================================
Optimizes capital allocation across multiple strategy types:
    TREND FOLLOWING   — captures sustained directional moves
    BREAKOUT          — catches range expansions
    MEAN REVERSION    — fades extremes in ranges
    SCALPING          — quick in-and-out for small gains
    CASH              — opportunity reserve

Default Allocation (changes with market cycle):
    | Phase          | Trend | Breakout | MeanRev | Scalp | Cash |
    |----------------|-------|----------|---------|-------|------|
    | EXPANSION      | 40%   | 25%      | 10%     | 15%   | 10%  |
    | PEAK           | 20%   | 10%      | 25%     | 15%   | 30%  |
    | CONSOLIDATION  | 15%   | 10%      | 40%     | 20%   | 15%  |
    | DECLINE        | 10%   | 5%       | 15%     | 10%   | 60%  |
    | RECOVERY       | 25%   | 20%      | 15%     | 15%   | 25%  |
    | UNKNOWN        | 25%   | 20%      | 20%     | 15%   | 20%  |

Allocation also adjusts based on:
    - Per-strategy recent performance (boost winners, cut losers)
    - Correlation between strategies (diversify)
    - Risk budget remaining
    - Volatility regime

Usage:
    opt = PortfolioAllocationOptimizer(equity=10000)

    # Set strategy performance
    opt.set_strategy_performance("trend", win_rate=0.62, avg_r=0.5, sharpe=1.5)
    opt.set_strategy_performance("breakout", win_rate=0.45, avg_r=0.3, sharpe=1.0)

    # Get optimal allocation
    allocation = opt.optimize(market_cycle="expansion", risk_budget_remaining=0.8)
    # allocation = {
    #     "trend": 0.40, "breakout": 0.25, "mean_reversion": 0.10,
    #     "scalping": 0.15, "cash": 0.10,
    #     "total_deployed": 0.90,
    #     "diversification_score": 0.82,
    # }
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.portfolio_allocation_optimizer")


# Default allocations per market cycle phase
DEFAULT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    "expansion":      {"trend": 0.40, "breakout": 0.25, "mean_reversion": 0.10, "scalping": 0.15, "cash": 0.10},
    "peak":           {"trend": 0.20, "breakout": 0.10, "mean_reversion": 0.25, "scalping": 0.15, "cash": 0.30},
    "consolidation":  {"trend": 0.15, "breakout": 0.10, "mean_reversion": 0.40, "scalping": 0.20, "cash": 0.15},
    "decline":        {"trend": 0.10, "breakout": 0.05, "mean_reversion": 0.15, "scalping": 0.10, "cash": 0.60},
    "recovery":       {"trend": 0.25, "breakout": 0.20, "mean_reversion": 0.15, "scalping": 0.15, "cash": 0.25},
    "unknown":        {"trend": 0.25, "breakout": 0.20, "mean_reversion": 0.20, "scalping": 0.15, "cash": 0.20},
}

STRATEGY_TYPES = ["trend", "breakout", "mean_reversion", "scalping", "cash"]


@dataclass
class AllocationResult:
    """Portfolio allocation optimization result."""
    allocation: Dict[str, float] = field(default_factory=dict)  # strategy → fraction
    total_deployed: float = 0.0       # fraction deployed (1 - cash)
    cash_reserve: float = 0.0
    diversification_score: float = 0.0  # 0-1
    allocation_usd: Dict[str, float] = field(default_factory=dict)
    # Adjustments
    performance_adjustments: Dict[str, float] = field(default_factory=dict)
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    # Metadata
    market_cycle: str = "unknown"
    risk_budget_remaining: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation": {k: round(v, 3) for k, v in self.allocation.items()},
            "total_deployed": round(self.total_deployed, 3),
            "cash_reserve": round(self.cash_reserve, 3),
            "diversification_score": round(self.diversification_score, 3),
            "allocation_usd": {k: round(v, 2) for k, v in self.allocation_usd.items()},
            "performance_adjustments": {k: round(v, 3) for k, v in self.performance_adjustments.items()},
            "risk_adjustments": {k: round(v, 3) for k, v in self.risk_adjustments.items()},
            "market_cycle": self.market_cycle,
            "risk_budget_remaining": round(self.risk_budget_remaining, 3),
            "reason": self.reason,
        }


class PortfolioAllocationOptimizer:
    """Optimizes capital allocation across strategy types."""

    def __init__(self,
                 equity: float = 10000.0,
                 min_allocation: float = 0.05,
                 max_allocation: float = 0.50,
                 performance_weight: float = 0.3,
                 risk_weight: float = 0.2):
        """Initialize allocator.

        Args:
            equity: total account equity
            min_allocation: minimum per strategy (5%)
            max_allocation: maximum per strategy (50%)
            performance_weight: how much to weight recent performance
            risk_weight: how much to weight risk budget
        """
        self.equity = equity
        self.min_alloc = min_allocation
        self.max_alloc = max_allocation
        self.perf_weight = performance_weight
        self.risk_weight = risk_weight
        self._strategy_perf: Dict[str, dict] = {}

    def set_strategy_performance(self, strategy: str,
                                  win_rate: float, avg_r: float,
                                  sharpe: float, sample_size: int = 0) -> None:
        """Set recent performance for a strategy."""
        self._strategy_perf[strategy] = {
            "win_rate": win_rate,
            "avg_r": avg_r,
            "sharpe": sharpe,
            "sample_size": sample_size,
        }

    def update_equity(self, equity: float) -> None:
        self.equity = equity

    # ------------------------------------------------------------------
    # Optimize allocation
    # ------------------------------------------------------------------
    def optimize(self,
                 market_cycle: str = "unknown",
                 risk_budget_remaining: float = 1.0,
                 volatility_regime: str = "normal") -> AllocationResult:
        """Optimize allocation across strategy types.

        Args:
            market_cycle: current market cycle phase
            risk_budget_remaining: fraction of risk budget remaining (0-1)
            volatility_regime: low/normal/high/extreme

        Returns:
            AllocationResult with optimal split
        """
        result = AllocationResult(
            market_cycle=market_cycle,
            risk_budget_remaining=risk_budget_remaining,
        )

        # === Start with default allocation for this cycle ===
        base = DEFAULT_ALLOCATIONS.get(market_cycle, DEFAULT_ALLOCATIONS["unknown"]).copy()
        result.allocation = base.copy()

        # === Performance adjustments ===
        for strategy in STRATEGY_TYPES:
            if strategy == "cash":
                continue
            perf = self._strategy_perf.get(strategy)
            if perf and perf["sample_size"] >= 10:
                # Boost winners, cut losers
                if perf["avg_r"] > 0.3:
                    adj = 1.2  # 20% boost
                elif perf["avg_r"] > 0:
                    adj = 1.0  # no change
                elif perf["avg_r"] > -0.2:
                    adj = 0.7  # 30% cut
                else:
                    adj = 0.3  # 70% cut
                result.performance_adjustments[strategy] = adj
                result.allocation[strategy] *= adj

        # === Risk budget adjustments ===
        # If risk budget low, shift to cash
        if risk_budget_remaining < 0.3:
            risk_mult = 0.5  # deploy only 50% of normal
            for s in STRATEGY_TYPES:
                if s != "cash":
                    result.allocation[s] *= risk_mult
                    result.risk_adjustments[s] = risk_mult
            result.allocation["cash"] = 1.0 - sum(
                result.allocation[s] for s in STRATEGY_TYPES if s != "cash"
            )
            result.risk_adjustments["cash"] = 1.0

        # === Volatility adjustments ===
        if volatility_regime == "extreme":
            # Shift heavily to cash
            for s in STRATEGY_TYPES:
                if s != "cash":
                    result.allocation[s] *= 0.3
            result.allocation["cash"] = 1.0 - sum(
                result.allocation[s] for s in STRATEGY_TYPES if s != "cash"
            )
        elif volatility_regime == "high":
            for s in STRATEGY_TYPES:
                if s != "cash":
                    result.allocation[s] *= 0.7

        # === Enforce min/max + normalize ===
        total = sum(result.allocation.values())
        if total > 0:
            for s in result.allocation:
                result.allocation[s] /= total
                result.allocation[s] = max(self.min_alloc, min(self.max_alloc, result.allocation[s]))
            # Re-normalize after clamping
            total = sum(result.allocation.values())
            for s in result.allocation:
                result.allocation[s] /= total

        # === Compute metrics ===
        result.total_deployed = sum(
            result.allocation[s] for s in STRATEGY_TYPES if s != "cash"
        )
        result.cash_reserve = result.allocation.get("cash", 0.0)
        result.diversification_score = self._compute_diversification(result.allocation)
        result.allocation_usd = {
            s: result.allocation[s] * self.equity for s in STRATEGY_TYPES
        }
        result.reason = self._explain(result)

        return result

    def _compute_diversification(self, allocation: Dict[str, float]) -> float:
        """Compute diversification score (0-1).

        Based on Herfindahl index (lower = more diversified).
        """
        weights = [allocation.get(s, 0) for s in STRATEGY_TYPES if s != "cash"]
        total = sum(weights)
        if total == 0:
            return 0.0
        norm = [w / total for w in weights]
        herfindahl = sum(w * w for w in norm)
        return max(0.0, 1.0 - herfindahl)

    def _explain(self, r: AllocationResult) -> str:
        """Generate explanation for the allocation."""
        parts = [f"Market cycle: {r.market_cycle}"]
        if r.performance_adjustments:
            boosted = [s for s, adj in r.performance_adjustments.items() if adj > 1]
            cut = [s for s, adj in r.performance_adjustments.items() if adj < 1]
            if boosted:
                parts.append(f"boosted: {', '.join(boosted)}")
            if cut:
                parts.append(f"reduced: {', '.join(cut)}")
        if r.risk_budget_remaining < 0.3:
            parts.append(f"low risk budget ({r.risk_budget_remaining:.0%}) → cash heavy")
        parts.append(f"diversification: {r.diversification_score:.2f}")
        return "; ".join(parts)
