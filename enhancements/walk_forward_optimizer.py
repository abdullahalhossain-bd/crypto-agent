"""enhancements.walk_forward_optimizer
=====================================================================
Day 155-157 — Walk-forward parameter optimizer.

For each strategy, searches the parameter space using walk-forward
cross-validation. Different from ml.trainer (which trains ML models)
— this optimises STRATEGY PARAMETERS (e.g. SMA fast/slow periods).

Process:
  1. Define parameter grid
  2. For each parameter combination:
     a. Train on first 70% of data
     b. Test on next 30%
     c. Slide forward, repeat
     d. Compute average out-of-sample Sharpe
  3. Return the parameter set with highest avg OOS Sharpe
  4. Verify it's not overfit (OOS Sharpe within 50% of in-sample)

Anti-overfitting:
  - Reject if IS Sharpe / OOS Sharpe > 2.0 (overfit)
  - Reject if OOS Sharpe variance across folds > 1.0 (unstable)
  - Reject if any fold has negative Sharpe
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("enhancements.wf_optimizer")


@dataclass
class OptimizationResult:
    strategy_name: str
    best_params: dict[str, Any]
    best_oos_sharpe: float
    best_is_sharpe: float
    overfit_ratio: float
    fold_sharpes: list[float]
    n_combinations_tested: int
    all_results: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "best_params": dict(self.best_params),
            "best_oos_sharpe": self.best_oos_sharpe,
            "best_is_sharpe": self.best_is_sharpe,
            "overfit_ratio": self.overfit_ratio,
            "fold_sharpes": list(self.fold_sharpes),
            "n_combinations_tested": self.n_combinations_tested,
            "passed": self.passed,
            "reason": self.reason,
            "all_results": list(self.all_results),
        }


# ----------------------------------------------------------------------
class WalkForwardOptimizer:
    def __init__(self,
                 n_folds: int = 5,
                 train_ratio: float = 0.7,
                 min_oos_sharpe: float = 0.3,
                 max_overfit_ratio: float = 2.0,
                 max_fold_variance: float = 1.0) -> None:
        self.n_folds = int(n_folds)
        self.train_ratio = float(train_ratio)
        self.min_oos_sharpe = float(min_oos_sharpe)
        self.max_overfit_ratio = float(max_overfit_ratio)
        self.max_fold_variance = float(max_fold_variance)

    # ----------------------------------------------------------------
    def optimize(
        self,
        strategy_name: str,
        param_grid: dict[str, list[Any]],
        backtest_func: Callable[[dict[str, Any], pd.DataFrame], dict[str, float]],
        df: pd.DataFrame,
    ) -> OptimizationResult:
        """Optimize parameters.

        Args:
            strategy_name: name for labelling
            param_grid: e.g. {"sma_fast": [5, 10, 20], "sma_slow": [30, 50, 100]}
            backtest_func: function(params, df_slice) → {"sharpe": float, "pnl": list[float]}
            df: full OHLCV dataframe
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        log.info("WFO: testing %d parameter combinations for %s",
                 len(combinations), strategy_name)

        all_results: list[dict[str, Any]] = []
        best_oos = -999.0
        best_params: dict[str, Any] = {}
        best_is = 0.0
        best_folds: list[float] = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                oos_sharpe, is_sharpe, fold_sharpes = self._walk_forward_eval(
                    params, backtest_func, df,
                )
            except Exception as e:  # noqa: BLE001
                log.debug("WFO combo %s failed: %r", params, e)
                continue
            result = {
                "params": params,
                "oos_sharpe": oos_sharpe,
                "is_sharpe": is_sharpe,
                "fold_sharpes": list(fold_sharpes),
                "overfit_ratio": (is_sharpe / oos_sharpe) if oos_sharpe > 0 else 0.0,
            }
            all_results.append(result)
            if oos_sharpe > best_oos:
                best_oos = oos_sharpe
                best_params = params
                best_is = is_sharpe
                best_folds = list(fold_sharpes)

        # Anti-overfitting checks
        overfit = (best_is / best_oos) if best_oos > 0 else 0.0
        fold_var = float(np.std(best_folds)) if len(best_folds) > 1 else 0.0
        any_negative_fold = any(s < 0 for s in best_folds)

        passed = True
        reasons: list[str] = []
        if best_oos < self.min_oos_sharpe:
            passed = False
            reasons.append(f"OOS Sharpe {best_oos:.2f} < {self.min_oos_sharpe}")
        if overfit > self.max_overfit_ratio:
            passed = False
            reasons.append(f"overfit ratio {overfit:.2f} > {self.max_overfit_ratio}")
        if fold_var > self.max_fold_variance:
            passed = False
            reasons.append(f"fold variance {fold_var:.2f} > {self.max_fold_variance}")
        if any_negative_fold:
            passed = False
            reasons.append("at least one fold has negative Sharpe")

        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_oos_sharpe=float(best_oos),
            best_is_sharpe=float(best_is),
            overfit_ratio=float(overfit),
            fold_sharpes=best_folds,
            n_combinations_tested=len(all_results),
            all_results=all_results,
            passed=passed,
            reason="; ".join(reasons) if reasons else "passed all checks",
        )

    # ----------------------------------------------------------------
    def _walk_forward_eval(
        self,
        params: dict[str, Any],
        backtest_func: Callable,
        df: pd.DataFrame,
    ) -> tuple[float, float, list[float]]:
        """Run walk-forward CV for one parameter set."""
        n = len(df)
        fold_size = n // (self.n_folds + 1)
        if fold_size < 50:
            return (0.0, 0.0, [])
        oos_sharpes: list[float] = []
        is_sharpes: list[float] = []
        for k in range(self.n_folds):
            train_start = k * fold_size
            train_end = train_start + int(fold_size * self.train_ratio)
            test_start = train_end
            test_end = min(n, (k + 1) * fold_size)
            if test_end <= test_start:
                continue
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            try:
                train_result = backtest_func(params, train_df)
                test_result = backtest_func(params, test_df)
                is_sharpes.append(float(train_result.get("sharpe", 0.0)))
                oos_sharpes.append(float(test_result.get("sharpe", 0.0)))
            except Exception:  # noqa: BLE001
                continue
        if not oos_sharpes:
            return (0.0, 0.0, [])
        return (
            float(np.mean(oos_sharpes)),
            float(np.mean(is_sharpes)),
            oos_sharpes,
        )
