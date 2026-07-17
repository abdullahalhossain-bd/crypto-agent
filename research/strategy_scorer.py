"""research.strategy_scorer
=====================================================================
Day 53-55 — Strategy Scoring System.

Every approved strategy gets a multi-dimensional score:

  - Sharpe (net of fees)        : risk-adjusted return
  - Drawdown                    : worst peak-to-trough
  - Stability                   : Sharpe consistency across folds
  - Regime robustness           : profitable in N of 3 regimes
  - Execution sensitivity       : how much edge survives costs
  - Capacity                    : lot-size scaling assumption

The composite score weights each dimension and is used by the
strategy factory (Phase 3) to decide which strategies are deployed
and how much capital they receive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger("research.scorer")


@dataclass
class StrategyScore:
    """Multi-dimensional score for a strategy."""
    strategy_name: str
    sharpe: float
    drawdown: float
    stability: float          # 1 - normalised std of fold sharpes
    regime_robustness: float  # fraction of profitable regimes
    execution_sensitivity: float  # net_sharpe / gross_sharpe
    capacity_score: float = 1.0   # 1.0 = unaffected by scale (default optimistic)
    composite: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "sharpe": self.sharpe,
            "drawdown": self.drawdown,
            "stability": self.stability,
            "regime_robustness": self.regime_robustness,
            "execution_sensitivity": self.execution_sensitivity,
            "capacity_score": self.capacity_score,
            "composite": self.composite,
            "weights": dict(self.weights),
            "metadata": dict(self.metadata),
        }

    def grade(self) -> str:
        """Letter grade A-F for quick classification."""
        if self.composite >= 0.8:
            return "A"
        if self.composite >= 0.65:
            return "B"
        if self.composite >= 0.5:
            return "C"
        if self.composite >= 0.35:
            return "D"
        return "F"


# ----------------------------------------------------------------------
class StrategyScorer:
    DEFAULT_WEIGHTS: dict[str, float] = {
        "sharpe": 0.30,
        "drawdown": 0.15,
        "stability": 0.20,
        "regime_robustness": 0.20,
        "execution_sensitivity": 0.10,
        "capacity_score": 0.05,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalise weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ----------------------------------------------------------------
    def score(
        self,
        strategy_name: str,
        sharpe: float,
        drawdown: float,
        fold_sharpes: list[float],
        regime_pass_count: int,
        gross_sharpe: float,
        net_sharpe: float,
        capacity_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyScore:
        """Compute the composite score."""
        # Normalise each dimension to [0, 1]
        norm_sharpe = float(min(1.0, max(0.0, sharpe / 3.0)))  # 3.0 Sharpe = max
        norm_drawdown = float(max(0.0, 1.0 - drawdown / 0.2))   # 20% DD = 0
        if fold_sharpes and len(fold_sharpes) > 1:
            arr = np.array(fold_sharpes)
            mean_s = float(arr.mean())
            std_s = float(arr.std())
            # Stability: mean / (mean + std), bounded [0,1]
            stability = float(mean_s / (mean_s + std_s)) if (mean_s + std_s) > 0 else 0.0
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 0.5
        regime_robustness = float(regime_pass_count / 3.0)
        exec_sens = float(net_sharpe / gross_sharpe) if gross_sharpe > 0 else 0.0
        exec_sens = max(0.0, min(1.0, exec_sens))
        capacity = float(max(0.0, min(1.0, capacity_score)))

        w = self.weights
        composite = (
            w["sharpe"] * norm_sharpe
            + w["drawdown"] * norm_drawdown
            + w["stability"] * stability
            + w["regime_robustness"] * regime_robustness
            + w["execution_sensitivity"] * exec_sens
            + w["capacity_score"] * capacity
        )
        s = StrategyScore(
            strategy_name=strategy_name,
            sharpe=float(sharpe),
            drawdown=float(drawdown),
            stability=float(stability),
            regime_robustness=float(regime_robustness),
            execution_sensitivity=float(exec_sens),
            capacity_score=capacity,
            composite=float(composite),
            weights=dict(w),
            metadata=metadata or {
                "norm_sharpe": norm_sharpe,
                "norm_drawdown": norm_drawdown,
                "fold_sharpes": list(fold_sharpes) if fold_sharpes else [],
            },
        )
        log.info("SCORE %s grade=%s composite=%.3f sharpe=%.2f dd=%.2f "
                 "stab=%.2f regime=%.2f exec=%.2f",
                 strategy_name, s.grade(), composite, sharpe, drawdown,
                 stability, regime_robustness, exec_sens)
        return s

    # ----------------------------------------------------------------
    def rank(self, scores: list[StrategyScore]) -> list[StrategyScore]:
        """Return scores sorted by composite (descending)."""
        return sorted(scores, key=lambda s: s.composite, reverse=True)

    def filter_deployable(self, scores: list[StrategyScore],
                          min_grade: str = "C") -> list[StrategyScore]:
        """Return only scores whose grade >= min_grade."""
        order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        threshold = order.get(min_grade, 3)
        return [s for s in scores if order.get(s.grade(), 0) >= threshold]
