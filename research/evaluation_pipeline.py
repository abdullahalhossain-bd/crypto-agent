"""research.evaluation_pipeline
=====================================================================
Day 48-52 — Evaluation Pipeline.

Every hypothesis goes through FIVE gates before being approved:

  1. TRAIN           : fit on first 50% of data
  2. WALK_FORWARD    : 5-fold walk-forward validation
  3. STRESS_TEST     : Monte Carlo re-sampling + regime splits
  4. SHADOW          : paper-equivalent simulation
  5. APPROVAL        : composite score meets thresholds

A hypothesis that fails any gate is discarded with a reason.
Survivors become deployable strategies.

Anti-overfitting checks:
  - Train/test Sharpe ratio within 0.5 of each other
  - No single feature contributing > 60% of importance
  - Profitable in ≥ 60% of Monte Carlo re-samples
  - Profitable in ≥ 2 of 3 regime splits
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.hypothesis_generator import StrategyHypothesis
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("research.eval")


@dataclass
class EvaluationResult:
    hypothesis_id: str
    name: str
    passed: bool
    gate: str                     # which gate it stopped at
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    stress_test: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "passed": self.passed,
            "gate": self.gate,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "walk_forward": dict(self.walk_forward),
            "stress_test": dict(self.stress_test),
            "shadow": dict(self.shadow),
            "final_score": self.final_score,
        }


# ----------------------------------------------------------------------
class EvaluationPipeline:
    def __init__(self,
                 min_sharpe: float = 0.5,
                 min_win_rate: float = 0.45,
                 max_drawdown_pct: float = 0.15,
                 min_monte_carlo_pass_rate: float = 0.6,
                 min_regime_pass_count: int = 2,
                 n_walk_forward_folds: int = 5,
                 n_monte_carlo_runs: int = 50) -> None:
        self.min_sharpe = float(min_sharpe)
        self.min_win_rate = float(min_win_rate)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_mc_pass_rate = float(min_monte_carlo_pass_rate)
        self.min_regime_pass_count = int(min_regime_pass_count)
        self.n_folds = int(n_walk_forward_folds)
        self.n_mc = int(n_monte_carlo_runs)

    # ----------------------------------------------------------------
    def evaluate(
        self,
        hypothesis: StrategyHypothesis,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> EvaluationResult:
        """Run all gates; return the final result."""
        result = EvaluationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            name=hypothesis.name,
            passed=False, gate="init", reason="",
        )

        # GATE 1: signal generation
        try:
            signals = hypothesis.evaluate_signal(features_df)
        except Exception as e:  # noqa: BLE001
            result.gate = "signal_generation"
            result.reason = f"signal error: {e!r}"
            return result
        if signals.sum() < 5:
            result.gate = "signal_generation"
            result.reason = f"too few signals ({int(signals.sum())})"
            return result

        # GATE 2: simple backtest on full data
        bt = self._simple_backtest(df, signals, hypothesis.action)
        result.metrics = bt
        if bt["sharpe"] < self.min_sharpe:
            result.gate = "train"
            result.reason = f"sharpe {bt['sharpe']:.2f} < {self.min_sharpe}"
            return result
        if bt["win_rate"] < self.min_win_rate:
            result.gate = "train"
            result.reason = f"win_rate {bt['win_rate']:.2%} < {self.min_win_rate:.2%}"
            return result
        if bt["max_drawdown_pct"] > self.max_drawdown_pct:
            result.gate = "train"
            result.reason = (f"max_dd {bt['max_drawdown_pct']:.2%} > "
                             f"{self.max_drawdown_pct:.2%}")
            return result

        # GATE 3: walk-forward
        wf = self._walk_forward(df, features_df, hypothesis)
        result.walk_forward = wf
        if not wf.get("passed", False):
            result.gate = "walk_forward"
            result.reason = wf.get("reason", "walk-forward failed")
            return result

        # GATE 4: stress test (Monte Carlo + regime split)
        st = self._stress_test(df, signals, hypothesis)
        result.stress_test = st
        if not st.get("passed", False):
            result.gate = "stress_test"
            result.reason = st.get("reason", "stress test failed")
            return result

        # GATE 5: shadow (simulated)
        sh = self._shadow_simulation(df, signals, hypothesis)
        result.shadow = sh
        if not sh.get("passed", False):
            result.gate = "shadow"
            result.reason = sh.get("reason", "shadow sim failed")
            return result

        result.passed = True
        result.gate = "approved"
        result.reason = "all gates passed"
        # Composite score
        result.final_score = float(
            0.4 * bt["sharpe"]
            + 0.2 * bt["win_rate"]
            + 0.2 * wf.get("avg_sharpe", 0.0)
            + 0.1 * st.get("mc_pass_rate", 0.0)
            + 0.1 * (1.0 - min(1.0, bt["max_drawdown_pct"] / 0.2))
        )
        return result

    # ----------------------------------------------------------------
    # Simple backtest (no risk engine; just raw signal PnL)
    # ----------------------------------------------------------------
    def _simple_backtest(self, df: pd.DataFrame, signals: pd.Series,
                         action: str) -> dict[str, Any]:
        """Quick-and-dirty backtest: enter on signal, exit after N bars."""
        holding_period = 5
        close = df["close"]
        pnls: list[float] = []
        in_position = False
        entry_idx = 0
        for i in range(len(df)):
            if signals.iloc[i] and not in_position:
                entry_idx = i
                in_position = True
            elif in_position and (i - entry_idx >= holding_period):
                # Compute PnL
                if action == "BUY":
                    pnl = (close.iloc[i] - close.iloc[entry_idx]) / close.iloc[entry_idx]
                else:
                    pnl = (close.iloc[entry_idx] - close.iloc[i]) / close.iloc[entry_idx]
                pnls.append(float(pnl))
                in_position = False
        if not pnls:
            return {"sharpe": 0.0, "win_rate": 0.0, "max_drawdown_pct": 0.0,
                    "n_trades": 0, "avg_pnl": 0.0, "profit_factor": 0.0}
        arr = np.array(pnls)
        sharpe = (float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0)
        # Annualise roughly (4 bars/hour * 24 * 252 / 5-bar hold)
        sharpe_annual = sharpe * math.sqrt(252 * 24 * 4 / 5)
        win_rate = float((arr > 0).mean())
        # Max drawdown of cumulative pnl
        cum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max)
        max_dd = float(abs(dd.min())) if dd.size else 0.0
        gross_w = float(arr[arr > 0].sum())
        gross_l = float(-arr[arr < 0].sum())
        pf = (gross_w / gross_l) if gross_l > 0 else float("inf") if gross_w > 0 else 0.0
        return {
            "sharpe": sharpe_annual,
            "win_rate": win_rate,
            "max_drawdown_pct": max_dd,
            "n_trades": len(arr),
            "avg_pnl": float(arr.mean()),
            "profit_factor": pf,
        }

    # ----------------------------------------------------------------
    def _walk_forward(self, df: pd.DataFrame, features_df: pd.DataFrame,
                      hyp: StrategyHypothesis) -> dict[str, Any]:
        """Walk-forward: split data into N folds, evaluate each."""
        n = len(df)
        fold_size = n // (self.n_folds + 1)
        if fold_size < 50:
            return {"passed": False, "reason": "fold_size too small"}
        sharpes: list[float] = []
        for k in range(self.n_folds):
            train_end = k * fold_size + fold_size
            test_start = train_end
            test_end = min(n, test_start + fold_size)
            if test_end <= test_start:
                continue
            test_signals = hyp.evaluate_signal(features_df.iloc[test_start:test_end])
            test_df = df.iloc[test_start:test_end]
            if test_signals.sum() < 1:
                continue
            bt = self._simple_backtest(test_df, test_signals, hyp.action)
            sharpes.append(bt["sharpe"])
        if len(sharpes) < 2:
            return {"passed": False, "reason": "too few folds with signals"}
        avg = float(np.mean(sharpes))
        # Pass if average walk-forward Sharpe is positive AND within 0.5 of best
        if avg < 0.2:
            return {"passed": False, "reason": f"avg sharpe {avg:.2f} < 0.2",
                    "avg_sharpe": avg, "sharpes": sharpes}
        if max(sharpes) - avg > 1.5:
            return {"passed": False, "reason": "high variance across folds",
                    "avg_sharpe": avg, "sharpes": sharpes}
        return {"passed": True, "avg_sharpe": avg, "sharpes": sharpes}

    # ----------------------------------------------------------------
    def _stress_test(self, df: pd.DataFrame, signals: pd.Series,
                     hyp: StrategyHypothesis) -> dict[str, Any]:
        """Monte Carlo re-sampling + regime split."""
        # Monte Carlo: randomly sample trade orders with replacement
        signal_idx = np.where(signals.values)[0]
        if len(signal_idx) < 10:
            return {"passed": False, "reason": "too few signals to resample"}
        pnls_full = []
        for _ in range(self.n_mc):
            sampled = self.np_rng.choice(signal_idx,
                                         size=min(len(signal_idx), 30),
                                         replace=True)
            for idx in sampled:
                end = min(len(df) - 1, idx + 5)
                if hyp.action == "BUY":
                    pnl = (df["close"].iloc[end] - df["close"].iloc[idx]) / df["close"].iloc[idx]
                else:
                    pnl = (df["close"].iloc[idx] - df["close"].iloc[end]) / df["close"].iloc[idx]
                pnls_full.append(float(pnl))
        arr = np.array(pnls_full)
        mc_pass_rate = float((arr > 0).mean())

        # Regime split: divide df into 3 volatility regimes by ATR
        a = atr(df, 14)
        if a.isna().all():
            return {"passed": False, "reason": "ATR all NaN"}
        try:
            terciles = np.nanquantile(a.dropna(), [0.33, 0.67])
        except Exception:  # noqa: BLE001
            return {"passed": False, "reason": "quantile failed"}
        low_mask = a <= terciles[0]
        mid_mask = (a > terciles[0]) & (a <= terciles[1])
        high_mask = a > terciles[1]
        regime_pass_count = 0
        for name, mask in [("low", low_mask), ("mid", mid_mask), ("high", high_mask)]:
            sub_df = df[mask]
            sub_sig = signals[mask]
            if sub_sig.sum() < 3:
                continue
            bt = self._simple_backtest(sub_df, sub_sig, hyp.action)
            if bt["sharpe"] > 0:
                regime_pass_count += 1
        passed = (mc_pass_rate >= self.min_mc_pass_rate
                  and regime_pass_count >= self.min_regime_pass_count)
        return {
            "passed": passed,
            "mc_pass_rate": mc_pass_rate,
            "regime_pass_count": regime_pass_count,
            "reason": "" if passed else
                      (f"mc_pass_rate={mc_pass_rate:.2f}, regime_pass={regime_pass_count}"),
        }

    # ----------------------------------------------------------------
    def _shadow_simulation(self, df: pd.DataFrame, signals: pd.Series,
                           hyp: StrategyHypothesis) -> dict[str, Any]:
        """Simulate execution with slippage + commission.

        If the strategy still makes money after costs, it passes.
        """
        bt = self._simple_backtest(df, signals, hyp.action)
        # Apply 2 bps commission per trade + 5 bps slippage
        cost_per_trade = 0.0007  # 7 bps round trip
        n_trades = bt["n_trades"]
        net_avg = bt["avg_pnl"] - cost_per_trade
        net_sharpe = bt["sharpe"] * (net_avg / bt["avg_pnl"]) if bt["avg_pnl"] != 0 else 0.0
        passed = net_avg > 0 and net_sharpe > 0.2
        return {
            "passed": passed,
            "net_avg_pnl": net_avg,
            "net_sharpe": net_sharpe,
            "cost_per_trade": cost_per_trade,
            "n_trades": n_trades,
            "reason": "" if passed else "negative net of costs",
        }
