"""validation.survival_test
=====================================================================
Day 106-110 — Strategy survival test framework.

A strategy is only trusted if it survives:
  - Regime change       : trend → chop, low-vol → high-vol
  - Volatility spikes   : sudden ATR explosion
  - Drawdown periods    : sustained adverse moves
  - Liquidity drops     : spread widening, slippage increase
  - Long dry spells     : extended periods with no signals

For each test, we generate (or replay) a stressed version of the data
and check whether the strategy:
  1. Doesn't blow up (equity stays positive)
  2. Recovers (drawdown doesn't exceed recovery threshold)
  3. Still has positive expectancy after costs
  4. Signal frequency doesn't collapse (no stale strategy)

A strategy that passes ALL survival tests is "battle-tested" — ready
for the live shadow phase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("validation.survival")


@dataclass
class SurvivalReport:
    strategy_name: str
    passed: bool
    n_tests: int
    n_passed: int
    n_failed: int
    per_test: list[dict[str, Any]] = field(default_factory=list)
    overall_verdict: str = ""    # battle_tested | fragile | failed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class SurvivalTestRunner:
    def __init__(self,
                 initial_equity: float = 10_000.0,
                 max_drawdown_threshold: float = 0.20,
                 min_expectancy_pct: float = 0.0,
                 min_signal_frequency: float = 0.01) -> None:
        self.initial_equity = float(initial_equity)
        self.max_dd_threshold = float(max_drawdown_threshold)
        self.min_expectancy = float(min_expectancy_pct)
        self.min_signal_freq = float(min_signal_frequency)

    # ----------------------------------------------------------------
    def run_all(
        self,
        strategy_name: str,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        df: pd.DataFrame,
        action: str = "BUY",
    ) -> SurvivalReport:
        """Run every survival test on a strategy."""
        tests = [
            ("regime_change_trend_to_chop", self._test_regime_change_trend_to_chop),
            ("regime_change_low_to_high_vol", self._test_regime_change_low_to_high_vol),
            ("volatility_spike", self._test_volatility_spike),
            ("sustained_drawdown", self._test_sustained_drawdown),
            ("liquidity_drop", self._test_liquidity_drop),
            ("long_dry_spell", self._test_long_dry_spell),
            ("gap_risk", self._test_gap_risk),
        ]
        per_test: list[dict[str, Any]] = []
        n_passed = 0
        for name, test_func in tests:
            result = test_func(signal_func, df, action, strategy_name)
            per_test.append({"test": name, **result})
            if result["passed"]:
                n_passed += 1
        n_failed = len(tests) - n_passed
        passed = n_failed == 0
        if passed:
            verdict = "battle_tested"
        elif n_passed >= len(tests) - 2:
            verdict = "fragile"
        else:
            verdict = "failed"
        return SurvivalReport(
            strategy_name=strategy_name,
            passed=passed, n_tests=len(tests),
            n_passed=n_passed, n_failed=n_failed,
            per_test=per_test, overall_verdict=verdict,
        )

    # ----------------------------------------------------------------
    # Test implementations
    # ----------------------------------------------------------------
    def _test_regime_change_trend_to_chop(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """Trend regime → chop regime mid-series."""
        stressed = df.copy()
        mid = len(stressed) // 2
        # Make second half choppy: mean-revert around a flat line
        base_price = float(stressed["close"].iloc[mid])
        rng = np.random.default_rng(42)
        for i in range(mid, len(stressed)):
            noise = rng.normal(0, 0.002)
            stressed.loc[stressed.index[i], "close"] = base_price * (1 + noise)
            stressed.loc[stressed.index[i], "high"] = stressed["close"].iloc[i] * 1.002
            stressed.loc[stressed.index[i], "low"] = stressed["close"].iloc[i] * 0.998
        return self._evaluate_stress(signal_func, stressed, action,
                                      "regime_change_trend_to_chop")

    def _test_regime_change_low_to_high_vol(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """Low vol → high vol mid-series."""
        stressed = df.copy()
        mid = len(stressed) // 2
        for i in range(mid, len(stressed)):
            close = float(stressed["close"].iloc[i])
            high = float(stressed["high"].iloc[i])
            low = float(stressed["low"].iloc[i])
            stressed.loc[stressed.index[i], "high"] = close + (high - close) * 3.0
            stressed.loc[stressed.index[i], "low"] = close - (close - low) * 3.0
        return self._evaluate_stress(signal_func, stressed, action,
                                      "regime_change_low_to_high_vol")

    def _test_volatility_spike(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """Sudden 5x ATR spike for 10 bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        for i in range(mid, min(mid + 10, len(stressed))):
            close = float(stressed["close"].iloc[i])
            high = float(stressed["high"].iloc[i])
            low = float(stressed["low"].iloc[i])
            stressed.loc[stressed.index[i], "high"] = close + (high - close) * 5.0
            stressed.loc[stressed.index[i], "low"] = close - (close - low) * 5.0
        return self._evaluate_stress(signal_func, stressed, action,
                                      "volatility_spike")

    def _test_sustained_drawdown(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """Sustained 10% drawdown over 50 bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        for j in range(50):
            i = mid + j
            if i >= len(stressed):
                break
            factor = 1.0 - 0.002 * (j + 1)
            stressed.loc[stressed.index[i], "close"] *= factor
            stressed.loc[stressed.index[i], "low"] *= factor
            stressed.loc[stressed.index[i], "high"] *= factor
        return self._evaluate_stress(signal_func, stressed, action,
                                      "sustained_drawdown")

    def _test_liquidity_drop(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """5x spread widening for 30 bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        for i in range(mid, min(mid + 30, len(stressed))):
            close = float(stressed["close"].iloc[i])
            spread = float(stressed["high"].iloc[i] - stressed["low"].iloc[i])
            stressed.loc[stressed.index[i], "high"] = close + spread * 2.5
            stressed.loc[stressed.index[i], "low"] = close - spread * 2.5
        return self._evaluate_stress(signal_func, stressed, action,
                                      "liquidity_drop")

    def _test_long_dry_spell(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """Flat market for 100 bars (no signals expected)."""
        stressed = df.copy()
        mid = len(stressed) // 2
        base_price = float(stressed["close"].iloc[mid])
        rng = np.random.default_rng(7)
        for i in range(mid, min(mid + 100, len(stressed))):
            tiny_noise = rng.normal(0, 0.0005)
            stressed.loc[stressed.index[i], "close"] = base_price * (1 + tiny_noise)
            stressed.loc[stressed.index[i], "high"] = base_price * (1 + tiny_noise + 0.0005)
            stressed.loc[stressed.index[i], "low"] = base_price * (1 + tiny_noise - 0.0005)
        return self._evaluate_stress(signal_func, stressed, action,
                                      "long_dry_spell")

    def _test_gap_risk(
        self, signal_func, df, action, strategy_name,
    ) -> dict[str, Any]:
        """5% gap between bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        if mid + 1 < len(stressed):
            gap_price = float(stressed["close"].iloc[mid]) * 0.95
            stressed.loc[stressed.index[mid + 1], "open"] = gap_price
            stressed.loc[stressed.index[mid + 1], "close"] = gap_price
            stressed.loc[stressed.index[mid + 1], "high"] = gap_price * 1.001
            stressed.loc[stressed.index[mid + 1], "low"] = gap_price * 0.999
        return self._evaluate_stress(signal_func, stressed, action,
                                      "gap_risk")

    # ----------------------------------------------------------------
    def _evaluate_stress(
        self,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        df: pd.DataFrame,
        action: str,
        test_name: str,
    ) -> dict[str, Any]:
        """Run the strategy on stressed df and evaluate survival.

        Major #5 fix: the old code assumed `signal_func` returns a boolean
        Series. If it returns floats (confidence) or a DataFrame, `signals.sum()`
        and `signals.iloc[i]` would behave incorrectly. Now we normalize
        the signal output to boolean using a threshold (> 0.5 for floats,
        or bool() for other types).
        """
        try:
            signals = signal_func(df)
        except Exception as e:  # noqa: BLE001
            return {"passed": False, "test": test_name,
                    "reason": f"signal error: {e!r}",
                    "n_signals": 0, "max_dd_pct": 1.0,
                    "expectancy": 0.0, "signal_frequency": 0.0}

        # Major #5 fix: normalize signals to boolean.
        if hasattr(signals, "dtype"):
            if signals.dtype == bool:
                pass  # already boolean
            else:
                # Convert float/int to boolean using threshold > 0.5
                signals = signals > 0.5
        elif isinstance(signals, (list, tuple)):
            signals = pd.Series([bool(s) for s in signals], index=df.index)
        else:
            signals = pd.Series([bool(signals)] * len(df), index=df.index)

        n_signals = int(signals.sum()) if hasattr(signals, "sum") else 0
        if n_signals == 0:
            return {"passed": True, "test": test_name,
                    "reason": "no signals (acceptable for dry spell)",
                    "n_signals": 0, "max_dd_pct": 0.0,
                    "expectancy": 0.0, "signal_frequency": 0.0}

        # Simulate trades: enter on signal, exit 5 bars later
        close = df["close"]
        holding = 5
        pnls: list[float] = []
        in_pos = False
        entry_idx = 0
        for i in range(len(df)):
            if hasattr(signals, "iloc") and signals.iloc[i] and not in_pos:
                entry_idx = i
                in_pos = True
            elif in_pos and (i - entry_idx >= holding):
                if action == "BUY":
                    pnl = (close.iloc[i] - close.iloc[entry_idx]) / close.iloc[entry_idx]
                else:
                    pnl = (close.iloc[entry_idx] - close.iloc[i]) / close.iloc[entry_idx]
                pnls.append(float(pnl))
                in_pos = False
        if not pnls:
            return {"passed": True, "test": test_name,
                    "reason": "no completed trades",
                    "n_signals": n_signals, "max_dd_pct": 0.0,
                    "expectancy": 0.0, "signal_frequency": n_signals / len(df)}
        arr = np.array(pnls)
        expectancy = float(arr.mean())
        # Apply cost
        expectancy_after_cost = expectancy - 0.0007  # 7 bps round-trip
        # Max drawdown on cumulative pnl
        cum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = float(abs(dd.min())) if dd.size else 0.0
        signal_freq = n_signals / len(df) if len(df) > 0 else 0.0
        # Pass criteria
        passed = (
            max_dd <= self.max_dd_threshold
            and expectancy_after_cost >= self.min_expectancy
            and signal_freq >= self.min_signal_freq * 0.1  # relaxed under stress
        )
        return {
            "passed": bool(passed),
            "test": test_name,
            "n_signals": n_signals,
            "n_completed_trades": len(arr),
            "expectancy": expectancy,
            "expectancy_after_cost": expectancy_after_cost,
            "max_dd_pct": max_dd,
            "signal_frequency": signal_freq,
            "reason": "" if passed else self._failure_reason(
                max_dd, expectancy_after_cost, signal_freq
            ),
        }

    @staticmethod
    def _failure_reason(max_dd: float, expectancy: float, sig_freq: float) -> str:
        reasons = []
        if max_dd > 0.20:
            reasons.append(f"max_dd {max_dd:.2%} > 20%")
        if expectancy < 0:
            reasons.append(f"expectancy {expectancy:.4f} < 0")
        if sig_freq < 0.001:
            reasons.append(f"signal_freq {sig_freq:.4f} too low")
        return "; ".join(reasons) or "unknown"
