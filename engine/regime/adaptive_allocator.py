"""engine.regime.adaptive_allocator
=====================================================================
Day 25 — Adaptive allocator.

Modulates position size and strategy enable/disable based on the
current `RegimeState` and each strategy's `regime_affinity`.

Rules:
  - If regime confidence < threshold → scale down everything
  - Per strategy: size_multiplier = affinity[regime] * regime_confidence
  - If size_multiplier < min_multiplier → strategy is "disabled"
    (its signals are vetoed by the portfolio manager)
  - In "high_vol" regime, also apply a global 0.5x de-risking

Output: a per-strategy allocation plan that the portfolio manager
uses to scale `TargetAllocation.lots`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from engine.regime.regime_classifier import RegimeState
from engine.strategies.base import StrategyMetadata
from utils.logger import get_logger

log = get_logger("engine.regime.allocator")


@dataclass
class StrategyAllocationPlan:
    """Per-strategy decision for the current bar."""
    strategy_name: str
    enabled: bool
    size_multiplier: float
    regime: str
    regime_confidence: float
    affinity: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ----------------------------------------------------------------------
class AdaptiveAllocator:
    def __init__(self,
                 min_regime_confidence: float = 0.3,
                 min_multiplier: float = 0.1,
                 high_vol_global_factor: float = 0.5) -> None:
        self.min_regime_confidence = float(min_regime_confidence)
        self.min_multiplier = float(min_multiplier)
        self.high_vol_global_factor = float(high_vol_global_factor)

    # ----------------------------------------------------------------
    def allocate(
        self,
        regime: RegimeState,
        strategies: dict[str, StrategyMetadata],
    ) -> dict[str, StrategyAllocationPlan]:
        """Return a per-strategy allocation plan for the current bar."""
        out: dict[str, StrategyAllocationPlan] = {}
        global_factor = (self.high_vol_global_factor
                         if regime.label == "high_vol" else 1.0)

        # If regime confidence is too low, just enable everything at
        # baseline size (don't override strategy affinity with bad data)
        if regime.confidence < self.min_regime_confidence:
            for name, meta in strategies.items():
                out[name] = StrategyAllocationPlan(
                    strategy_name=name, enabled=True,
                    size_multiplier=global_factor,
                    regime=regime.label,
                    regime_confidence=regime.confidence,
                    affinity=1.0,
                    reason="low regime confidence — neutral",
                )
            return out

        for name, meta in strategies.items():
            affinity = float(meta.regime_affinity.get(regime.label, 0.5))
            mult = affinity * regime.confidence * global_factor
            mult = max(0.0, min(1.0, mult))
            enabled = mult >= self.min_multiplier
            reason = (f"affinity={affinity:.2f} * conf={regime.confidence:.2f}"
                      f" * global={global_factor:.2f}")
            out[name] = StrategyAllocationPlan(
                strategy_name=name,
                enabled=enabled,
                size_multiplier=float(mult),
                regime=regime.label,
                regime_confidence=regime.confidence,
                affinity=affinity,
                reason=reason,
            )
        log.debug("ALLOCATOR regime=%s plans=%s", regime.label,
                  {n: f"{p.size_multiplier:.2f}{'✓' if p.enabled else '✗'}"
                   for n, p in out.items()})
        return out
