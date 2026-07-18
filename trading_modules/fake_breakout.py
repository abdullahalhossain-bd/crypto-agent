"""
Fake Breakout Detector — "Is this breakout real or a trap?"
============================================================

Many breakouts are fake — price breaks a level, traps breakout traders,
then reverses. This module inspects the last N bars to detect:

    1. Real Breakout   — broke level + closed beyond + volume supports + retest held
    2. Fake Breakout   — broke level + closed back inside + (low volume OR reversal)
    3. Pending Breakout— broke level but has not yet closed beyond

A "level" can be a recent swing high/low, a session high/low, or any
caller-supplied price.

Usage:
    from trading_modules.fake_breakout import FakeBreakoutDetector
    detector = FakeBreakoutDetector()
    result = detector.analyze(df_m15, level=65200.0, direction="BUY")
    if result.is_fake:
        # skip — this breakout is a trap
    elif result.is_real:
        # confirmed real breakout
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BreakoutResult:
    is_real: bool = False
    is_fake: bool = False
    is_pending: bool = False
    broke_level: bool = False
    closed_beyond: bool = False
    retest_completed: bool = False
    volume_supports: bool = False
    breakout_type: str = "none"        # "real" / "fake" / "pending" / "none"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_real": self.is_real,
            "is_fake": self.is_fake,
            "is_pending": self.is_pending,
            "broke_level": self.broke_level,
            "closed_beyond": self.closed_beyond,
            "retest_completed": self.retest_completed,
            "volume_supports": self.volume_supports,
            "breakout_type": self.breakout_type,
            "notes": self.notes,
        }


class FakeBreakoutDetector:
    """
    Detect real vs fake breakouts.

    Parameters:
        lookback: bars to inspect for the breakout (default 5)
        volume_min_ratio: volume must be >= this * avg volume to confirm (default 1.5)
        retest_lookback: bars after the break to look for a retest (default 5)
        retest_atr_tolerance: retest within this * ATR of the level (default 0.3)
        atr_period: ATR lookback (default 14)
    """

    def __init__(
        self, lookback: int = 5, volume_min_ratio: float = 1.5,
        retest_lookback: int = 5, retest_atr_tolerance: float = 0.3,
        atr_period: int = 14,
    ) -> None:
        self.lookback = lookback
        self.volume_min_ratio = volume_min_ratio
        self.retest_lookback = retest_lookback
        self.retest_atr_tolerance = retest_atr_tolerance
        self.atr_period = atr_period

    def analyze(
        self, df: pd.DataFrame, level: float, direction: str = "BUY",
    ) -> BreakoutResult:
        """Analyze the most recent bars for a breakout of `level`.

        Args:
            df: OHLCV dataframe (columns: open, high, low, close, volume)
            level: price level to test for breakout
            direction: "BUY" = expect breakout ABOVE level; "SELL" = BELOW
        """
        if df is None or len(df) < self.lookback + 1:
            return BreakoutResult()
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            direction = "BUY"

        atr_series = self._atr(df, self.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr <= 0 or not np.isfinite(atr):
            atr = 1.0

        recent = df.tail(self.lookback + 1).reset_index(drop=True)
        notes: list[str] = []
        broke_level = False
        closed_beyond = False
        retest_completed = False
        volume_supports = False

        # Avg volume of bars before `recent`
        prior = df.iloc[:-len(recent)] if len(df) > len(recent) else df
        avg_vol = float(prior["volume"].mean()) if len(prior) > 0 else 1.0
        if avg_vol <= 0:
            avg_vol = 1.0

        # Detect the breakout bar
        break_idx: Optional[int] = None
        for i in range(len(recent)):
            row = recent.iloc[i]
            if direction == "BUY":
                # Look for the first bar that broke above the level
                if float(row["high"]) > level:
                    broke_level = True
                    # Closed beyond?
                    if float(row["close"]) > level:
                        closed_beyond = True
                    break_idx = i
                    vol_ratio = float(row["volume"]) / avg_vol
                    if vol_ratio >= self.volume_min_ratio:
                        volume_supports = True
                    notes.append(f"break bar @ idx {i}: vol_ratio={vol_ratio:.2f}")
                    break
            else:  # SELL
                if float(row["low"]) < level:
                    broke_level = True
                    if float(row["close"]) < level:
                        closed_beyond = True
                    break_idx = i
                    vol_ratio = float(row["volume"]) / avg_vol
                    if vol_ratio >= self.volume_min_ratio:
                        volume_supports = True
                    notes.append(f"break bar @ idx {i}: vol_ratio={vol_ratio:.2f}")
                    break

        if not broke_level:
            return BreakoutResult(breakout_type="none", notes=["no breakout detected"])

        # Inspect subsequent bars for retest / fake
        subsequent = recent.iloc[break_idx + 1:]
        last_close = float(recent.iloc[-1]["close"])

        # Fake breakout: price closed back inside the level
        if direction == "BUY" and last_close < level:
            notes.append(f"closed back below level ({last_close:.2f} < {level:.2f})")
            return BreakoutResult(
                is_fake=True, broke_level=True, closed_beyond=closed_beyond,
                volume_supports=volume_supports, breakout_type="fake", notes=notes,
            )
        if direction == "SELL" and last_close > level:
            notes.append(f"closed back above level ({last_close:.2f} > {level:.2f})")
            return BreakoutResult(
                is_fake=True, broke_level=True, closed_beyond=closed_beyond,
                volume_supports=volume_supports, breakout_type="fake", notes=notes,
            )

        # Pending: broke but has not closed beyond
        if not closed_beyond:
            notes.append("broke level but did not close beyond — pending")
            return BreakoutResult(
                is_pending=True, broke_level=True, closed_beyond=False,
                volume_supports=volume_supports, breakout_type="pending", notes=notes,
            )

        # Look for retest in subsequent bars
        retest_tol = atr * self.retest_atr_tolerance
        for _, row in subsequent.iterrows():
            if direction == "BUY":
                # Retest: price came back down near the level
                if abs(float(row["low"]) - level) <= retest_tol:
                    retest_completed = True
                    notes.append("retest of level completed")
                    break
            else:
                if abs(float(row["high"]) - level) <= retest_tol:
                    retest_completed = True
                    notes.append("retest of level completed")
                    break

        # Real breakout: closed beyond + volume supports (retest is bonus)
        is_real = closed_beyond and volume_supports
        if is_real:
            notes.append("real breakout — closed beyond + volume supports")
            if retest_completed:
                notes.append("bonus: retest held")

        return BreakoutResult(
            is_real=is_real, broke_level=True, closed_beyond=closed_beyond,
            retest_completed=retest_completed, volume_supports=volume_supports,
            breakout_type="real" if is_real else ("pending" if not retest_completed else "weak_real"),
            notes=notes,
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()


__all__ = ["FakeBreakoutDetector", "BreakoutResult"]