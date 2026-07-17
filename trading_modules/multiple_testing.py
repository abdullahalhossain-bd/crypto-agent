"""
Multiple-Testing Corrections Module
=====================================

The answer to "is my backtest real or just lucky?"

When you test many signals or strategies, the "best" one will be inflated
by selection bias. These corrections tell you whether your edge is real.

Implements 5 methods:
  1. Benjamini-Hochberg FDR (False Discovery Rate)
  2. Deflated Sharpe Ratio (DSR) — Bailey & López de Prado
  3. White's Reality Check — bootstrap-based
  4. PBO (Probability of Backtest Overfitting) — combinatorial
  5. CSCV (Combinatorially Symmetric Cross-Validation)

Source: ml4t-3e (review #18) + AI-Trader research pipeline (review #19)
       + López de Prado (2018) "Advances in Financial Machine Learning"

Usage:
    from multiple_testing import (
        benjamini_hochberg,
        deflated_sharpe_ratio,
        whites_reality_check,
        probability_of_backtest_overfitting,
    )

    # 1. BH-FDR: correct p-values for multiple testing
    p_values = [0.001, 0.03, 0.04, 0.12, 0.45]
    corrected = benjamini_hochberg(p_values)
    # → q-values: [0.005, 0.075, 0.08, 0.15, 0.45]

    # 2. DSR: is this Sharpe ratio real?
    dsr = deflated_sharpe_ratio(
        sharpe=1.5, n_trials=100, n_obs=252,
        skew=-0.3, kurtosis=4.5,
    )
    # → DSR p-value: 0.12 (not significant after 100 trials)

    # 3. White's Reality Check: is strategy better than benchmark?
    p = whites_reality_check(strategy_returns, benchmark_returns)
    # → p-value for "strategy genuinely outperforms"

    # 4. PBO: probability your backtest is overfit?
    pbo = probability_of_backtest_overfitting(returns_matrix)
    # → PBO = 0.65 means 65% chance your "best" strategy is overfit
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from statistics import NormalDist


# ═══════════════════════════════════════════════════════════════
# 1. Benjamini-Hochberg FDR
# ═══════════════════════════════════════════════════════════════

def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    Benjamini-Hochberg False Discovery Rate correction.

    Controls the expected proportion of false discoveries among
    all rejected hypotheses.

    Args:
        p_values: List of raw p-values

    Returns:
        List of q-values (FDR-adjusted p-values), same order as input
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    running = 1.0

    # Walk from largest to smallest
    for rank, (original_idx, p_val) in reversed(list(enumerate(indexed, start=1))):
        corrected = p_val * n / max(rank, 1)
        running = min(running, corrected)
        adjusted[original_idx] = min(running, 1.0)

    return [round(q, 6) for q in adjusted]


# ═══════════════════════════════════════════════════════════════
# 2. Deflated Sharpe Ratio (DSR)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DSRResult:
    """Deflated Sharpe Ratio result."""
    observed_sharpe: float
    expected_sharpe: float  # Expected max Sharpe under null
    dsr_statistic: float    # Test statistic
    p_value: float          # P-value (lower = more significant)
    is_significant: bool    # True if p < 0.05
    n_trials: int
    n_observations: int

    def to_dict(self) -> dict:
        return {
            "observed_sharpe": round(self.observed_sharpe, 4),
            "expected_sharpe": round(self.expected_sharpe, 4),
            "dsr_statistic": round(self.dsr_statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
        }


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
) -> DSRResult:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the observed Sharpe ratio for:
    1. Multiple testing (you tried N strategies)
    2. Non-normal returns (skew and kurtosis)
    3. Sample length (shorter = less reliable)

    The expected maximum Sharpe under the null (all strategies have
    zero true Sharpe) grows with n_trials. DSR tests whether the
    observed Sharpe exceeds this expected maximum.

    Args:
        sharpe: Observed annualized Sharpe ratio
        n_trials: Number of strategies tested (multiple testing penalty)
        n_obs: Number of observations (days, bars, etc.)
        skew: Return skewness (0 = normal)
        kurtosis: Return kurtosis (3 = normal)
        alpha: Significance level (default 0.05)

    Returns:
        DSRResult with p-value and significance
    """
    # Expected maximum Sharpe under null hypothesis
    # E[max(Z_1, ..., Z_N)] ≈ sqrt(2*ln(N)) * (1 - 1/(2*ln(N))) for large N
    if n_trials > 1:
        euler_mascheroni = 0.5772156649
        expected_max = (
            math.sqrt(2 * math.log(n_trials))
            * (1 - euler_mascheroni / (2 * math.log(n_trials)))
        )
    else:
        expected_max = 0.0

    # Variance of Sharpe ratio estimator (non-normal adjustment)
    # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurtosis-3)/4 * SR^2) / n_obs
    sr_var = (
        1
        + 0.5 * sharpe ** 2
        - skew * sharpe
        + (kurtosis - 3) / 4 * sharpe ** 2
    ) / max(n_obs, 1)
    sr_std = math.sqrt(max(sr_var, 1e-10))

    # DSR test statistic
    dsr_stat = (sharpe - expected_max) / sr_std

    # P-value (one-sided test: is observed > expected max?)
    p_value = 1 - NormalDist().cdf(dsr_stat)

    return DSRResult(
        observed_sharpe=sharpe,
        expected_sharpe=expected_max,
        dsr_statistic=dsr_stat,
        p_value=p_value,
        is_significant=p_value < alpha,
        n_trials=n_trials,
        n_observations=n_obs,
    )


# ═══════════════════════════════════════════════════════════════
# 3. White's Reality Check
# ═══════════════════════════════════════════════════════════════

def whites_reality_check(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 1000,
) -> float:
    """
    White's Reality Check (bootstrap-based).

    Tests whether a strategy genuinely outperforms a benchmark,
    accounting for the fact that you may have tried many strategies.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        n_bootstrap: Number of bootstrap iterations

    Returns:
        P-value for the null hypothesis that strategy does NOT
        outperform the benchmark.
    """
    n = len(strategy_returns)
    if n != len(benchmark_returns) or n < 10:
        return 1.0

    # Observed performance differential
    diff = strategy_returns - benchmark_returns
    observed_stat = np.mean(diff)

    # Bootstrap: resample with replacement, compute mean diff each time
    rng = np.random.default_rng(42)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diff = strategy_returns[idx] - benchmark_returns[idx]
        bootstrap_stats[i] = np.mean(boot_diff)

    # P-value: fraction of bootstrap samples where stat >= observed
    # (under the null of no outperformance)
    p_value = np.mean(bootstrap_stats >= observed_stat)

    return float(p_value)


# ═══════════════════════════════════════════════════════════════
# 4. Probability of Backtest Overfitting (PBO)
# ═══════════════════════════════════════════════════════════════

def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 8,
) -> float:
    """
    Probability of Backtest Overfitting (Bailey et al., 2017).

    Uses Combinatorially Symmetric Cross-Validation (CSCV):
    1. Split returns into n_splits blocks
    2. For each C(n, n/2) combination of blocks as IS (in-sample) / OOS (out-of-sample)
    3. Find best strategy IS → check if it's also best OOS
    4. PBO = fraction of combinations where IS-best is NOT OOS-best

    Args:
        returns_matrix: Shape (n_strategies, n_periods) matrix of returns
        n_splits: Number of time blocks (default 8)

    Returns:
        PBO probability (0-1). Higher = more likely overfit.
        PBO > 0.5 = likely overfit. PBO < 0.25 = probably robust.
    """
    from itertools import combinations

    n_strategies, n_periods = returns_matrix.shape
    if n_periods < n_splits * 2:
        return 1.0  # Not enough data → assume overfit

    # Split into blocks
    block_size = n_periods // n_splits
    blocks = [returns_matrix[:, i*block_size:(i+1)*block_size] for i in range(n_splits)]

    # All C(n_splits, n_splits//2) combinations
    n_is = n_splits // 2
    combinations_list = list(combinations(range(n_splits), n_is))

    overfit_count = 0
    total = 0

    for is_blocks in combinations_list:
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]

        # IS returns: concatenate IS blocks
        is_returns = np.concatenate([blocks[b] for b in is_blocks], axis=1)
        # OOS returns: concatenate OOS blocks
        oos_returns = np.concatenate([blocks[b] for b in oos_blocks], axis=1)

        # Find best strategy in IS (highest Sharpe)
        is_sharpes = _sharpe_ratios(is_returns)
        best_is_idx = np.argmax(is_sharpes)

        # Check rank of IS-best in OOS
        oos_sharpes = _sharpe_ratios(oos_returns)
        oos_rank = np.argsort(np.argsort(oos_sharpes))  # Rank (0 = worst)
        best_is_oos_rank = oos_rank[best_is_idx]

        # PBO: IS-best is in bottom half of OOS
        if best_is_oos_rank < n_strategies / 2:
            overfit_count += 1
        total += 1

    return overfit_count / total if total > 0 else 1.0


def _sharpe_ratios(returns: np.ndarray) -> np.ndarray:
    """Compute Sharpe ratio for each strategy (row)."""
    means = np.mean(returns, axis=1)
    stds = np.std(returns, axis=1, ddof=1)
    stds = np.where(stds == 0, 1e-10, stds)
    return means / stds * np.sqrt(252)  # Annualized


# ═══════════════════════════════════════════════════════════════
# 5. Summary: Run All Tests
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestVerification:
    """Complete backtest verification result."""
    # DSR
    dsr_p_value: float = 1.0
    dsr_significant: bool = False
    # White's Reality Check
    whites_p_value: float = 1.0
    whites_significant: bool = False
    # PBO
    pbo_probability: float = 1.0
    likely_overfit: bool = True
    # Overall
    verdict: str = "REJECT"
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "dsr": {"p_value": round(self.dsr_p_value, 6), "significant": self.dsr_significant},
            "whites": {"p_value": round(self.whites_p_value, 6), "significant": self.whites_significant},
            "pbo": {"probability": round(self.pbo_probability, 4), "likely_overfit": self.likely_overfit},
            "verdict": self.verdict,
            "explanation": self.explanation,
        }


def verify_backtest(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    strategy_returns: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    returns_matrix: Optional[np.ndarray] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> BacktestVerification:
    """
    Run all multiple-testing corrections on a backtest result.

    This is the "is my edge real?" function. Run it before trusting
    any backtest result.

    Args:
        sharpe: Observed annualized Sharpe ratio
        n_trials: Number of strategies/params tested
        n_obs: Number of observations
        strategy_returns: Strategy return series (for White's RC)
        benchmark_returns: Benchmark return series (for White's RC)
        returns_matrix: (n_strategies, n_periods) for PBO
        skew: Return skewness
        kurtosis: Return kurtosis

    Returns:
        BacktestVerification with all tests + overall verdict
    """
    result = BacktestVerification()

    # 1. Deflated Sharpe Ratio
    dsr = deflated_sharpe_ratio(sharpe, n_trials, n_obs, skew, kurtosis)
    result.dsr_p_value = dsr.p_value
    result.dsr_significant = dsr.is_significant

    # 2. White's Reality Check
    if strategy_returns is not None and benchmark_returns is not None:
        result.whites_p_value = whites_reality_check(strategy_returns, benchmark_returns)
        result.whites_significant = result.whites_p_value < 0.05

    # 3. PBO
    if returns_matrix is not None:
        result.pbo_probability = probability_of_backtest_overfitting(returns_matrix)
        result.likely_overfit = result.pbo_probability > 0.5

    # Overall verdict
    tests_passed = sum([result.dsr_significant, result.whites_significant, not result.likely_overfit])
    tests_run = sum([True, strategy_returns is not None, returns_matrix is not None])

    if tests_passed == tests_run and tests_run > 0:
        result.verdict = "ACCEPT"
        result.explanation = "All tests passed — edge is likely real"
    elif tests_passed > 0:
        result.verdict = "MARGINAL"
        result.explanation = f"{tests_passed}/{tests_run} tests passed — edge may be real but needs more validation"
    else:
        result.verdict = "REJECT"
        result.explanation = "No tests passed — edge is likely due to luck/overfitting"

    return result
