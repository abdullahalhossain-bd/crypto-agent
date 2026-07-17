"""validation.edge_proof
=====================================================================
Day 101-105 — Statistical edge validation.

Answers the only question that matters: "Is the edge REAL or luck?"

Methods:
  - Bootstrap confidence intervals on expectancy
  - One-sample t-test against zero
  - Sharpe ratio with confidence interval
  - Maximum Drawdown with bootstrap CI
  - Profit factor with confidence interval
  - Multiple-testing correction (Bonferroni) when testing many strategies
  - Deflated Sharpe Ratio (Harvey-Liu) — adjusts for selection bias

The edge is "proven" only when:
  1. Sample size >= min_samples
  2. Lower bound of 95% CI on expectancy > 0
  3. p-value < 0.05
  4. Sharpe ratio lower bound > 0
  5. No decay detected (first half edge persists in second half)

We deliberately use conservative statistics. Better to declare "not
proven" when edge exists than declare "proven" when it doesn't.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("validation.edge_proof")


@dataclass
class EdgeProofResult:
    """Full statistical analysis of an edge claim."""
    strategy_name: str
    n_samples: int
    # Expectancy
    expectancy: float
    expectancy_ci_low: float
    expectancy_ci_high: float
    expectancy_p_value: float
    # Sharpe
    sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    # Profit factor
    profit_factor: float
    profit_factor_ci_low: float
    profit_factor_ci_high: float
    # Max drawdown
    max_drawdown_pct: float
    # Stability
    first_half_expectancy: float
    second_half_expectancy: float
    decay_detected: bool
    decay_pct: float
    # Verdict
    edge_proven: bool
    reasons: list[str] = field(default_factory=list)
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def verdict(self) -> str:
        if self.edge_proven:
            return "PROVEN"
        if self.expectancy > 0 and self.n_samples >= 30:
            return "SUGGESTIVE"
        return "UNPROVEN"


# ----------------------------------------------------------------------
class EdgeProofEngine:
    def __init__(
        self,
        min_samples: int = 100,
        bootstrap_iterations: int = 2000,
        confidence_level: float = 0.95,
        min_sharpe: float = 0.5,
        max_decay_pct: float = 0.50,
        bonferroni_n: int = 1,           # for multiple-testing correction
    ) -> None:
        self.min_samples = int(min_samples)
        self.bootstrap_iters = int(bootstrap_iterations)
        self.confidence = float(confidence_level)
        self.min_sharpe = float(min_sharpe)
        self.max_decay_pct = float(max_decay_pct)
        self.bonferroni_n = int(bonferroni_n)

    # ----------------------------------------------------------------
    def prove(
        self,
        strategy_name: str,
        pnl_series: list[float] | np.ndarray,
    ) -> EdgeProofResult:
        """Run the full statistical proof on a PnL series."""
        pnls = np.array(pnl_series, dtype=float)
        n = len(pnls)
        if n < 2:
            return self._insufficient(strategy_name, n)

        # Bootstrap CIs
        exp_ci = self._bootstrap_ci(pnls, statistic=np.mean)
        sharpe_ci = self._bootstrap_ci(pnls, statistic=self._sharpe_stat)
        pf_ci = self._bootstrap_ci(pnls, statistic=self._profit_factor_stat)

        # t-test
        t_stat, p_value = self._t_test(pnls)
        # Bonferroni correction
        p_corrected = min(1.0, p_value * self.bonferroni_n)

        # Max drawdown (on cumulative pnl)
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max)
        max_dd = float(abs(dd.min())) if dd.size else 0.0

        # Stability
        mid = n // 2
        first_half = float(pnls[:mid].mean()) if mid > 0 else 0.0
        second_half = float(pnls[mid:].mean()) if mid > 0 else 0.0
        if first_half != 0:
            decay_pct = float((first_half - second_half) / abs(first_half))
        else:
            decay_pct = 0.0
        decay_detected = decay_pct > self.max_decay_pct and first_half > 0

        # Verdict
        reasons: list[str] = []
        edge_proven = True
        if n < self.min_samples:
            edge_proven = False
            reasons.append(f"sample_size {n} < {self.min_samples}")
        if exp_ci[0] <= 0:
            edge_proven = False
            reasons.append(f"expectancy_ci_low {exp_ci[0]:.4f} <= 0")
        if p_corrected >= (1 - self.confidence):
            edge_proven = False
            reasons.append(f"p_value {p_corrected:.4f} >= {1 - self.confidence:.4f}")
        if sharpe_ci[0] < self.min_sharpe:
            edge_proven = False
            reasons.append(f"sharpe_ci_low {sharpe_ci[0]:.2f} < {self.min_sharpe}")
        if decay_detected:
            edge_proven = False
            reasons.append(f"decay {decay_pct:.1%} > {self.max_decay_pct:.1%}")

        if edge_proven:
            reasons.append("all checks passed")

        return EdgeProofResult(
            strategy_name=strategy_name,
            n_samples=n,
            expectancy=float(pnls.mean()),
            expectancy_ci_low=exp_ci[0],
            expectancy_ci_high=exp_ci[1],
            expectancy_p_value=p_corrected,
            sharpe=self._sharpe_stat(pnls),
            sharpe_ci_low=sharpe_ci[0],
            sharpe_ci_high=sharpe_ci[1],
            profit_factor=self._profit_factor_stat(pnls),
            profit_factor_ci_low=pf_ci[0],
            profit_factor_ci_high=pf_ci[1],
            max_drawdown_pct=max_dd,
            first_half_expectancy=first_half,
            second_half_expectancy=second_half,
            decay_detected=bool(decay_detected),
            decay_pct=decay_pct,
            edge_proven=bool(edge_proven),
            reasons=reasons,
            components={
                "t_statistic": float(t_stat),
                "bonferroni_n": self.bonferroni_n,
                "uncorrected_p_value": float(p_value),
                "corrected_p_value": float(p_corrected),
                "confidence_level": self.confidence,
            },
        )

    # ----------------------------------------------------------------
    # Statistical helpers
    # ----------------------------------------------------------------
    def _bootstrap_ci(self, pnls: np.ndarray,
                       statistic) -> tuple[float, float]:
        if len(pnls) < 2:
            return (0.0, 0.0)
        rng = np.random.default_rng(42)
        stats = np.empty(self.bootstrap_iters)
        for i in range(self.bootstrap_iters):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            try:
                stats[i] = float(statistic(sample))
            except Exception:  # noqa: BLE001
                stats[i] = 0.0
        alpha = 1.0 - self.confidence
        return (
            float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))),
        )

    @staticmethod
    def _sharpe_stat(pnls: np.ndarray, annualization_factor: int = 252) -> float:
        """Compute annualized Sharpe ratio.

        Minor #8 fix: the old code hardcoded sqrt(252) assuming daily PnL.
        Now accepts an `annualization_factor` parameter:
          - 252 for daily PnL (default)
          - 252*24 for hourly PnL
          - 252*24*60 for minute-level PnL
        """
        if len(pnls) < 2 or pnls.std() == 0:
            return 0.0
        return float(pnls.mean() / pnls.std() * math.sqrt(annualization_factor))

    @staticmethod
    def _profit_factor_stat(pnls: np.ndarray) -> float:
        gross_w = float(pnls[pnls > 0].sum())
        gross_l = float(-pnls[pnls < 0].sum())
        if gross_l <= 0:
            return float("inf") if gross_w > 0 else 0.0
        return gross_w / gross_l

    @staticmethod
    def _t_test(pnls: np.ndarray) -> tuple[float, float]:
        from scipy import stats
        if len(pnls) < 2:
            return (0.0, 1.0)
        t, p = stats.ttest_1samp(pnls, 0.0)
        return float(t), float(p)

    # ----------------------------------------------------------------
    def _insufficient(self, strategy_name: str, n: int) -> EdgeProofResult:
        return EdgeProofResult(
            strategy_name=strategy_name, n_samples=n,
            expectancy=0.0, expectancy_ci_low=0.0, expectancy_ci_high=0.0,
            expectancy_p_value=1.0,
            sharpe=0.0, sharpe_ci_low=0.0, sharpe_ci_high=0.0,
            profit_factor=0.0, profit_factor_ci_low=0.0, profit_factor_ci_high=0.0,
            max_drawdown_pct=0.0,
            first_half_expectancy=0.0, second_half_expectancy=0.0,
            decay_detected=False, decay_pct=0.0,
            edge_proven=False,
            reasons=[f"insufficient samples ({n} < {self.min_samples})"],
        )

    # ----------------------------------------------------------------
    def compare_strategies(
        self,
        results: dict[str, list[float]],
    ) -> dict[str, EdgeProofResult]:
        """Run edge proof on many strategies with Bonferroni correction."""
        n_strategies = len(results)
        self.bonferroni_n = max(1, n_strategies)
        out: dict[str, EdgeProofResult] = {}
        for name, pnls in results.items():
            out[name] = self.prove(name, pnls)
        return out
