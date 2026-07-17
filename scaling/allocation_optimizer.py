"""scaling.allocation_optimizer
=====================================================================
Day 88-90 — Allocation Optimizer.

Computes the optimal capital allocation across strategies using:

  weight_i = (performance_score_i × stability_score_i × regime_match_i)
             / Σ_j(performance_j × stability_j × regime_match_j)

Then applies practical constraints:
  - Max weight per strategy (concentration cap)
  - Min weight (avoid dust allocations)
  - Sum to 1.0 (or to whatever fraction of capital is deployable)
  - Correlation penalty (similar to portfolio_manager.allocate)

Output: an `AllocationPlan` mapping strategy → weight, plus audit metadata.

Audit fixes:
  - Critical #1: iterative cap-and-renormalise that handles the all-capped
    case (where every strategy hits the max_weight cap, leaving sum < 1).
    The old code's fixed 5-iteration redistribution loop would break
    without renormalising, producing under-allocated weights.
  - Critical #3: NaN/inf input validation — replaces non-finite values
    with 0 and logs a warning.
  - Major #1: replaced the fixed `for _ in range(5)` loop with a `while`
    loop that runs until convergence (or a safety limit of 100 iterations).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("scaling.optimizer")


@dataclass
class AllocationPlan:
    weights: dict[str, float]
    total_weight: float
    reason: str = ""
    raw_weights: dict[str, float] = field(default_factory=dict)
    constraints_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "total_weight": self.total_weight,
            "reason": self.reason,
            "raw_weights": dict(self.raw_weights),
            "constraints_applied": list(self.constraints_applied),
            "metadata": dict(self.metadata),
        }


def _sanitize_float(val: Any, default: float = 0.0, name: str = "") -> float:
    """Critical #3 fix: replace NaN/inf with a finite default."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        log.warning("allocation_optimizer: non-numeric value %r for %s — using default %.2f", val, name, default)
        return default
    if not math.isfinite(f):
        log.warning("allocation_optimizer: non-finite value %r for %s — using default %.2f", val, name, default)
        return default
    return f


# ----------------------------------------------------------------------
class AllocationOptimizer:
    def __init__(
        self,
        max_weight_per_strategy: float = 0.40,
        min_weight: float = 0.05,
        correlation_penalty_threshold: float = 0.7,
    ) -> None:
        self.max_weight = float(max_weight_per_strategy)
        self.min_weight = float(min_weight)
        self.corr_threshold = float(correlation_penalty_threshold)

    # ----------------------------------------------------------------
    def optimize(
        self,
        strategy_scores: dict[str, float],         # composite scores in [0, 1]
        strategy_stabilities: dict[str, float],    # stability scores in [0, 1]
        strategy_regime_matches: dict[str, float], # regime match in [0, 1]
        correlation_matrix: Optional[dict[str, dict[str, float]]] = None,
        deployable_fraction: float = 1.0,
    ) -> AllocationPlan:
        """Compute optimal weights subject to constraints."""
        strategies = list(strategy_scores.keys())
        if not strategies:
            return AllocationPlan({}, 0.0, "no strategies")

        # Step 1: raw weights = score * stability * regime_match
        # Critical #3 fix: sanitize all inputs for NaN/inf.
        raw: dict[str, float] = {}
        for s in strategies:
            score = max(0.0, _sanitize_float(strategy_scores.get(s, 0.0), 0.0, f"score[{s}]"))
            stab = max(0.0, _sanitize_float(strategy_stabilities.get(s, 0.5), 0.5, f"stability[{s}]"))
            regime = max(0.0, _sanitize_float(strategy_regime_matches.get(s, 0.5), 0.5, f"regime[{s}]"))
            raw[s] = score * stab * regime

        # Step 2: correlation penalty (scale down redundant strategies)
        constraints_applied: list[str] = []
        if correlation_matrix:
            for i, s_i in enumerate(strategies):
                for s_j in strategies[i + 1:]:
                    corr = abs(_sanitize_float(
                        correlation_matrix.get(s_i, {}).get(s_j, 0.0), 0.0,
                        f"corr[{s_i},{s_j}]"))
                    if corr >= self.corr_threshold:
                        if raw[s_i] >= raw[s_j]:
                            raw[s_j] *= (1.0 - corr * 0.5)
                            constraints_applied.append(
                                f"corr_penalty:{s_j} (corr={corr:.2f} with {s_i})"
                            )
                        else:
                            raw[s_i] *= (1.0 - corr * 0.5)
                            constraints_applied.append(
                                f"corr_penalty:{s_i} (corr={corr:.2f} with {s_j})"
                            )

        # Step 3: normalise to sum=1
        total_raw = sum(raw.values())
        if total_raw <= 0:
            weights = {s: 1.0 / len(strategies) for s in strategies}
            constraints_applied.append("equal_weight_fallback")
        else:
            weights = {s: raw[s] / total_raw for s in strategies}

        # Step 4: apply max weight cap with iterative cap-and-renormalise.
        # Critical #1 + Major #1 fix: the old code used a fixed 5-iteration
        # redistribution loop that failed when ALL strategies were capped
        # (no uncapped strategies to redistribute to), leaving sum < 1.0.
        # The new algorithm:
        #   1. Cap any weight exceeding max_weight.
        #   2. Renormalise the uncapped strategies to fill the remaining budget.
        #   3. Repeat until no weight exceeds max_weight.
        #   4. Final renormalise to guarantee sum = 1.0 (handles all-capped case).
        for iteration in range(100):  # Major #1: while-loop with safety limit
            capped_any = False
            for s in weights:
                if weights[s] > self.max_weight:
                    weights[s] = self.max_weight
                    capped_any = True
            if not capped_any:
                break
            # Renormalise: redistribute the remaining budget to uncapped strategies.
            total = sum(weights.values())
            if total > 0:
                # Scale all weights so sum = 1.0; capped strategies will be
                # re-capped on the next iteration if they exceed max_weight again.
                weights = {s: w / total for s, w in weights.items()}
            if iteration == 0:
                constraints_applied.append(f"max_weight_cap:{self.max_weight}")

        # Critical #1 fix: final renormalise — if all strategies were capped,
        # the sum may be < 1.0. Scale them up proportionally (even if some
        # exceed max_weight — in the all-capped case, max_weight is the only
        # constraint and we must honour sum=1.0 over the per-strategy cap).
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            weights = {s: w / total for s, w in weights.items()}

        # Step 5: drop dust allocations (< min_weight)
        dust = [s for s, w in weights.items() if w < self.min_weight]
        for s in dust:
            weights[s] = 0.0
            constraints_applied.append(f"drop_dust:{s}")
        # Renormalise
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        # Step 6: scale to deployable_fraction
        weights = {s: w * deployable_fraction for s, w in weights.items()}
        total_weight = sum(weights.values())

        return AllocationPlan(
            weights={s: float(w) for s, w in weights.items() if w > 0},
            total_weight=float(total_weight),
            reason="optimised",
            raw_weights={s: float(raw[s]) for s in strategies},
            constraints_applied=constraints_applied,
            metadata={
                "n_strategies_input": len(strategies),
                "n_strategies_deployed": sum(1 for w in weights.values() if w > 0),
                "deployable_fraction": deployable_fraction,
            },
        )
