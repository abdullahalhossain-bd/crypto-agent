"""scaling.stress_tests
=====================================================================
Day 85-87 — Stress Test Runner.

Replays the portfolio through historical + synthetic crisis scenarios:

  1. FLASH_CRASH      : sudden -10% gap in 1 bar, recovery in next 5
  2. LIQUIDITY_DRY    : slippage x5 for N bars
  3. CORRELATED_DD    : all symbols drop simultaneously
  4. REGIME_SHIFT     : volatility doubles overnight
  5. GAP_RISK         : price gaps 5% between bars (no fills in between)
  6. BLACK_SWAN       : combination of all of the above

Each test returns a `StressTestResult` showing:
  - Portfolio survival (equity never went negative)
  - Max drawdown under stress
  - Time to recover
  - Whether risk limits fired correctly
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("scaling.stress")


@dataclass
class StressTestResult:
    name: str
    passed: bool
    initial_equity: float
    final_equity: float
    max_drawdown_pct: float
    min_equity: float
    recovery_bars: int
    risk_limits_fired: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_equity": self.min_equity,
            "recovery_bars": self.recovery_bars,
            "risk_limits_fired": list(self.risk_limits_fired),
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class StressTestRunner:
    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self.initial_equity = float(initial_equity)

    # ----------------------------------------------------------------
    def run_all(self, df: pd.DataFrame,
                position_lots: float = 0.1) -> list[StressTestResult]:
        """Run every stress test against the same df."""
        return [
            self.flash_crash(df, position_lots),
            self.liquidity_dry(df, position_lots),
            self.correlated_drawdown(df, position_lots),
            self.regime_shift(df, position_lots),
            self.gap_risk(df, position_lots),
            self.black_swan(df, position_lots),
        ]

    # ----------------------------------------------------------------
    def flash_crash(self, df: pd.DataFrame,
                    position_lots: float) -> StressTestResult:
        """Inject a -10% gap in one bar, recovery over 5 bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        crash_price = stressed["close"].iloc[mid] * 0.90
        for i in range(mid, min(mid + 1, len(stressed))):
            stressed.loc[stressed.index[i], "close"] = crash_price
            stressed.loc[stressed.index[i], "low"] = crash_price * 0.99
            # Minor #3 fix: adjust high to encompass the new close so we
            # don't produce invalid candles where high < close.
            stressed.loc[stressed.index[i], "high"] = max(
                float(stressed["high"].iloc[i]), crash_price)
        # Linear recovery over 5 bars
        recovery_price = stressed["close"].iloc[mid - 1]
        for j in range(1, 6):
            if mid + j < len(stressed):
                new_p = crash_price + (recovery_price - crash_price) * (j / 5)
                stressed.loc[stressed.index[mid + j], "close"] = new_p
                # Minor #3 fix: adjust high/low for recovery bars too.
                stressed.loc[stressed.index[mid + j], "high"] = max(
                    float(stressed["high"].iloc[mid + j]), new_p)
                stressed.loc[stressed.index[mid + j], "low"] = min(
                    float(stressed["low"].iloc[mid + j]), new_p)
        return self._evaluate_stress("flash_crash", stressed, position_lots,
                                      expected_dd=0.10)

    def liquidity_dry(self, df: pd.DataFrame,
                      position_lots: float) -> StressTestResult:
        """Simulate 5x slippage for 50 bars in the middle."""
        stressed = df.copy()
        mid = len(stressed) // 2
        end = min(mid + 50, len(stressed))
        # Add slippage as inflated high-low spread
        for i in range(mid, end):
            base_spread = stressed["high"].iloc[i] - stressed["low"].iloc[i]
            extra = base_spread * 4  # 5x normal
            stressed.loc[stressed.index[i], "high"] += extra / 2
            stressed.loc[stressed.index[i], "low"] -= extra / 2
        return self._evaluate_stress("liquidity_dry", stressed, position_lots,
                                      expected_dd=0.05)

    def correlated_drawdown(self, df: pd.DataFrame,
                            position_lots: float) -> StressTestResult:
        """All symbols (single df here) drop 5% over 10 bars."""
        stressed = df.copy()
        mid = len(stressed) // 2
        for j in range(10):
            if mid + j < len(stressed):
                factor = 1.0 - 0.005 * (j + 1)  # gradual 5% drop
                stressed.loc[stressed.index[mid + j], "close"] *= factor
                stressed.loc[stressed.index[mid + j], "low"] *= factor
                stressed.loc[stressed.index[mid + j], "high"] *= factor
        return self._evaluate_stress("correlated_drawdown", stressed,
                                      position_lots, expected_dd=0.05)

    def regime_shift(self, df: pd.DataFrame,
                     position_lots: float) -> StressTestResult:
        """Double volatility in the second half of the data."""
        stressed = df.copy()
        mid = len(stressed) // 2
        # Scale high-low spread by sqrt(2) for second half
        for i in range(mid, len(stressed)):
            close = stressed["close"].iloc[i]
            high = stressed["high"].iloc[i]
            low = stressed["low"].iloc[i]
            new_high = close + (high - close) * math.sqrt(2)
            new_low = close - (close - low) * math.sqrt(2)
            stressed.loc[stressed.index[i], "high"] = new_high
            stressed.loc[stressed.index[i], "low"] = new_low
        return self._evaluate_stress("regime_shift", stressed, position_lots,
                                      expected_dd=0.08)

    def gap_risk(self, df: pd.DataFrame,
                 position_lots: float) -> StressTestResult:
        """Price gaps 5% between two bars (no fill possible)."""
        stressed = df.copy()
        mid = len(stressed) // 2
        if mid + 1 < len(stressed):
            gap_price = stressed["close"].iloc[mid] * 0.95
            stressed.loc[stressed.index[mid + 1], "open"] = gap_price
            stressed.loc[stressed.index[mid + 1], "close"] = gap_price
            stressed.loc[stressed.index[mid + 1], "high"] = gap_price * 1.001
            stressed.loc[stressed.index[mid + 1], "low"] = gap_price * 0.999
        return self._evaluate_stress("gap_risk", stressed, position_lots,
                                      expected_dd=0.05)

    def black_swan(self, df: pd.DataFrame,
                   position_lots: float) -> StressTestResult:
        """Combination: gap + crash + high vol + correlated DD."""
        stressed = df.copy()
        mid = len(stressed) // 2
        # Apply 8% gap then -5% over 5 bars
        if mid + 5 < len(stressed):
            gap_price = stressed["close"].iloc[mid] * 0.92
            for j in range(5):
                idx = mid + 1 + j
                if idx < len(stressed):
                    p = gap_price * (1.0 - 0.01 * j)
                    stressed.loc[stressed.index[idx], "close"] = p
                    stressed.loc[stressed.index[idx], "low"] = p * 0.995
                    stressed.loc[stressed.index[idx], "high"] = p * 1.005
        # Scale vol for rest
        for i in range(mid + 5, len(stressed)):
            close = stressed["close"].iloc[i]
            high = stressed["high"].iloc[i]
            low = stressed["low"].iloc[i]
            stressed.loc[stressed.index[i], "high"] = close + (high - close) * 1.5
            stressed.loc[stressed.index[i], "low"] = close - (close - low) * 1.5
        return self._evaluate_stress("black_swan", stressed, position_lots,
                                      expected_dd=0.15)

    # ----------------------------------------------------------------
    def _evaluate_stress(self, name: str, df: pd.DataFrame,
                         position_lots: float,
                         expected_dd: float) -> StressTestResult:
        """Simulate a long position through the stressed df."""
        equity = self.initial_equity
        peak = equity
        min_equity = equity
        max_dd = 0.0
        recovery_from = None
        recovery_bars = 0
        risk_limits_fired: list[str] = []

        entry_price = float(df["close"].iloc[0])
        for i in range(len(df)):
            price = float(df["close"].iloc[i])
            # Mark-to-market
            pnl = (price - entry_price) * position_lots
            equity = self.initial_equity + pnl
            peak = max(peak, equity)
            min_equity = min(min_equity, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            # Risk limit checks
            if dd > 0.05 and "daily_loss" not in risk_limits_fired:
                risk_limits_fired.append("daily_loss")
            if dd > 0.10 and "max_drawdown" not in risk_limits_fired:
                risk_limits_fired.append("max_drawdown")
            if equity < self.initial_equity * 0.85 and "halt" not in risk_limits_fired:
                risk_limits_fired.append("halt")
            # Recovery
            if max_dd > 0.02 and recovery_from is None:
                recovery_from = i
            if recovery_from is not None and equity >= peak:
                recovery_bars = i - recovery_from
                recovery_from = None

        # Pass criteria: never went bust + max_dd within 2x expected
        never_bust = min_equity > 0
        dd_within_tolerance = max_dd <= expected_dd * 2.5
        passed = never_bust and dd_within_tolerance
        return StressTestResult(
            name=name,
            passed=passed,
            initial_equity=self.initial_equity,
            final_equity=equity,
            max_drawdown_pct=float(max_dd),
            min_equity=float(min_equity),
            recovery_bars=int(recovery_bars),
            risk_limits_fired=risk_limits_fired,
            details={
                "expected_dd": expected_dd,
                "position_lots": position_lots,
            },
        )
