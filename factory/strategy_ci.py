"""factory.strategy_ci
=====================================================================
Day 56-60 — Strategy CI/CD pipeline.

Every candidate strategy must pass FOUR automated gates before it
can be promoted to production:

  1. UNIT TESTS      : deterministic signal correctness
  2. BACKTEST        : walk-forward Sharpe + DD within thresholds
  3. REGIME STABILITY: profitable in ≥ 2 of 3 regimes
  4. EXECUTION SIM   : net of slippage + commission still profitable

If any gate fails, the strategy is rejected with a reason. If all
pass, a new version is registered in the StrategyVersionStore.

This is conceptually identical to software CI/CD: code goes in,
automated tests run, artifact (strategy version) comes out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from factory.strategy_versioning import StrategyVersion, StrategyVersionStore
from research.evaluation_pipeline import EvaluationPipeline, EvaluationResult
from research.hypothesis_generator import StrategyHypothesis
from utils.logger import get_logger

log = get_logger("factory.ci")


@dataclass
class CIPipelineResult:
    strategy_name: str
    version: str
    passed: bool
    gate_failed: str = ""
    reason: str = ""
    duration_s: float = 0.0
    unit_tests: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    regime_stability: dict[str, Any] = field(default_factory=dict)
    execution_sim: dict[str, Any] = field(default_factory=dict)
    version_registered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "version": self.version,
            "passed": self.passed,
            "gate_failed": self.gate_failed,
            "reason": self.reason,
            "duration_s": self.duration_s,
            "unit_tests": dict(self.unit_tests),
            "backtest": dict(self.backtest),
            "regime_stability": dict(self.regime_stability),
            "execution_sim": dict(self.execution_sim),
            "version_registered": self.version_registered,
        }


# ----------------------------------------------------------------------
class StrategyCI:
    """Orchestrates the 4-gate CI pipeline."""

    def __init__(
        self,
        version_store: StrategyVersionStore,
        evaluation_pipeline: Optional[EvaluationPipeline] = None,
        min_sharpe: float = 0.5,
        min_win_rate: float = 0.45,
        max_drawdown_pct: float = 0.15,
        min_regime_pass_count: int = 2,
        min_net_sharpe: float = 0.2,
    ) -> None:
        self.version_store = version_store
        self.eval = evaluation_pipeline or EvaluationPipeline()
        self.min_sharpe = float(min_sharpe)
        self.min_win_rate = float(min_win_rate)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_regime_pass_count = int(min_regime_pass_count)
        self.min_net_sharpe = float(min_net_sharpe)

    # ----------------------------------------------------------------
    def run(
        self,
        strategy_name: str,
        hypothesis: StrategyHypothesis,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        unit_tests: Optional[list[Callable[[], bool]]] = None,
    ) -> CIPipelineResult:
        """Run all gates. Returns the final CI result."""
        start = time.time()
        result = CIPipelineResult(
            strategy_name=strategy_name,
            version="0.0.0",
            passed=False,
        )

        # GATE 1: unit tests
        ut_result = self._run_unit_tests(unit_tests or [])
        result.unit_tests = ut_result
        if not ut_result["passed"]:
            result.gate_failed = "unit_tests"
            result.reason = ut_result["reason"]
            result.duration_s = time.time() - start
            return result

        # GATE 2-5: evaluation pipeline (backtest + walk-forward + stress + shadow)
        eval_result = self.eval.evaluate(hypothesis, df, features_df)
        result.backtest = eval_result.metrics
        result.regime_stability = eval_result.stress_test
        result.execution_sim = eval_result.shadow
        if not eval_result.passed:
            result.gate_failed = eval_result.gate
            result.reason = eval_result.reason
            result.duration_s = time.time() - start
            return result

        # All gates passed → register a new version
        new_version = self.version_store.bump_version(strategy_name)
        version_obj = StrategyVersion(
            strategy_name=strategy_name,
            version=new_version,
            registered_at=time.time(),
            hypothesis=hypothesis.to_dict(),
            score=eval_result.final_score,
            metrics=eval_result.metrics,
            status="staging",
        )
        self.version_store.register(version_obj)
        result.version = new_version
        result.version_registered = True
        result.passed = True
        result.gate_failed = ""
        result.reason = "all gates passed"
        result.duration_s = time.time() - start
        log.info("CI PASS %s v%s (%.1fs) score=%.3f",
                 strategy_name, new_version, result.duration_s,
                 eval_result.final_score)
        return result

    # ----------------------------------------------------------------
    def _run_unit_tests(self, tests: list[Callable[[], bool]]) -> dict[str, Any]:
        if not tests:
            return {"passed": True, "n": 0, "reason": "no unit tests"}
        results: list[dict[str, Any]] = []
        all_passed = True
        for i, t in enumerate(tests):
            try:
                ok = bool(t())
                results.append({"test": i, "passed": ok})
                if not ok:
                    all_passed = False
            except Exception as e:  # noqa: BLE001
                results.append({"test": i, "passed": False, "error": repr(e)})
                all_passed = False
        return {
            "passed": all_passed,
            "n": len(tests),
            "results": results,
            "reason": "" if all_passed else "one or more unit tests failed",
        }
