"""factory.auto_retirement
=====================================================================
Day 67-70 — Auto-Retirement System.

Continuously monitors every production strategy and decides whether
to retire it. Three triggers:

  1. PERFORMANCE DECAY     : decay_score < threshold over N cycles
  2. DRAWDOWN BREACH       : rolling drawdown > strategy-specific limit
  3. REGIME MISMATCH       : current regime has affinity < threshold
                             AND has been so for > M cycles

When triggered, the strategy is:
  - Marked as "retired" in the version store
  - Removed from the strategy pool (next cycle)
  - Position-sized to 0 (existing positions are managed normally)
  - Operator notified via the alert system

Auto-retirement is REVERSIBLE — operator can promote a retired
strategy back if the regime changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from factory.decay_detector import DecayDetector, DecayReport
from factory.strategy_versioning import StrategyVersionStore
from utils.logger import get_logger

log = get_logger("factory.retirement")


@dataclass
class RetirementDecision:
    strategy_name: str
    version: str
    retire: bool
    reason: str
    trigger: str                 # decay | drawdown | regime | manual
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "version": self.version,
            "retire": self.retire,
            "reason": self.reason,
            "trigger": self.trigger,
            "evidence": dict(self.evidence),
            "ts": self.ts,
        }


# ----------------------------------------------------------------------
class AutoRetirement:
    def __init__(
        self,
        version_store: StrategyVersionStore,
        decay_detector: DecayDetector,
        max_drawdown_pct: float = 0.10,
        min_regime_affinity: float = 0.2,
        regime_mismatch_cycles: int = 50,
    ) -> None:
        self.version_store = version_store
        self.decay = decay_detector
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_regime_affinity = float(min_regime_affinity)
        self.regime_mismatch_cycles = int(regime_mismatch_cycles)
        # Track consecutive regime-mismatch cycles per strategy
        self._regime_mismatch: dict[str, int] = {}
        # Track rolling drawdown per strategy
        self._rolling_dd: dict[str, float] = {}

    # ----------------------------------------------------------------
    def evaluate(
        self,
        strategy_name: str,
        current_regime: str = "",
        regime_affinity: Optional[dict[str, float]] = None,
    ) -> RetirementDecision:
        """Decide whether to retire a single strategy."""
        version = self.version_store.get(strategy_name)
        version_str = version.version if version else "unknown"
        trigger = ""
        reason = ""
        retire = False
        evidence: dict[str, Any] = {}

        # 1. Decay check
        decay_report = self.decay.evaluate(strategy_name)
        if decay_report is not None:
            evidence["decay"] = decay_report.to_dict()
            if decay_report.recommendation == "retire":
                retire = True
                trigger = "decay"
                reason = (f"decay_score={decay_report.decay_score:.2f} "
                          f"< {self.decay.retirement_threshold}")

        # 2. Drawdown check
        dd = self._rolling_dd.get(strategy_name, 0.0)
        evidence["drawdown_pct"] = dd
        if not retire and dd > self.max_drawdown_pct:
            retire = True
            trigger = "drawdown"
            reason = (f"rolling drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}")

        # 3. Regime mismatch check
        if regime_affinity and current_regime:
            affinity = regime_affinity.get(current_regime, 0.5)
            if affinity < self.min_regime_affinity:
                self._regime_mismatch[strategy_name] = (
                    self._regime_mismatch.get(strategy_name, 0) + 1
                )
            else:
                self._regime_mismatch[strategy_name] = 0
            mismatch_cycles = self._regime_mismatch.get(strategy_name, 0)
            evidence["regime_mismatch_cycles"] = mismatch_cycles
            evidence["current_regime"] = current_regime
            evidence["regime_affinity"] = affinity
            if (not retire
                    and mismatch_cycles >= self.regime_mismatch_cycles):
                retire = True
                trigger = "regime"
                reason = (f"regime mismatch for {mismatch_cycles} cycles "
                          f"(affinity={affinity:.2f} in {current_regime})")
        else:
            evidence["regime_mismatch_cycles"] = self._regime_mismatch.get(strategy_name, 0)

        decision = RetirementDecision(
            strategy_name=strategy_name,
            version=version_str,
            retire=retire,
            reason=reason or "healthy",
            trigger=trigger or "none",
            evidence=evidence,
            ts=time.time(),
        )

        # If retiring, mark in version store
        if retire and version is not None:
            self.version_store.retire(strategy_name, version_str,
                                       reason=reason)
            log.warning("AUTO-RETIRE %s v%s trigger=%s reason=%s",
                        strategy_name, version_str, trigger, reason)
        return decision

    # ----------------------------------------------------------------
    def update_drawdown(self, strategy_name: str, drawdown_pct: float) -> None:
        self._rolling_dd[strategy_name] = float(drawdown_pct)

    def reset_mismatch(self, strategy_name: str) -> None:
        self._regime_mismatch.pop(strategy_name, None)

    # ----------------------------------------------------------------
    def evaluate_all(
        self,
        strategy_names: list[str],
        current_regime: str = "",
        regime_affinities: Optional[dict[str, dict[str, float]]] = None,
    ) -> list[RetirementDecision]:
        out = []
        for name in strategy_names:
            affinity = (regime_affinities or {}).get(name)
            out.append(self.evaluate(name, current_regime, affinity))
        return out
