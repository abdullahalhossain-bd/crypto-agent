"""
Chart Pattern Recognition — geometric pattern detection
========================================================

Detects classical chart patterns from OHLCV data using geometric
heuristics (no ML, no image processing):

    1. Head & Shoulders / Inverse Head & Shoulders
    2. Double Top / Double Bottom
    3. Triple Top / Triple Bottom
    4. Ascending / Descending / Symmetrical Triangles
    5. Rising / Falling Wedges
    6. Bullish / Bearish Flags
    7. Rectangle (range) breakout

Each pattern detection returns:
    - detected: bool
    - pattern_type: str
    - neckline / support / resistance levels
    - projected_target: float (measured move target)
    - confidence: 0..1

Usage:
    from trading_modules.chart_patterns import ChartPatternDetector
    detector = ChartPatternDetector()
    result = detector.detect_all(df_m15)
    for p in result.patterns:
        print(f"{p.pattern_type}: target={p.projected_target:.2f}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    pattern_type: str               # "head_and_shoulders", "double_top", etc.
    direction: str                  # "bullish" / "bearish"
    detected: bool
    confidence: float               # 0..1
    key_levels: dict[str, float]    # e.g., {"neckline": 65000, "head": 66000}
    projected_target: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "detected": self.detected,
            "confidence": round(self.confidence, 2),
            "key_levels": {k: round(v, 2) for k, v in self.key_levels.items()},
            "projected_target": round(self.projected_target, 2) if self.projected_target else None,
            "notes": self.notes,
        }


@dataclass
class PatternDetectionResult:
    patterns: list[Pattern] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patterns": [p.to_dict() for p in self.patterns],
            "notes": self.notes,
        }


class ChartPatternDetector:
    """Detect classical chart patterns from OHLCV data.

    Parameters:
        swing_window: half-window for swing detection (default 5)
        atr_period: ATR lookback for tolerance scaling (default 14)
        level_tolerance_atr: levels within this × ATR = "equal" (default 0.3)
    """

    def __init__(
        self, swing_window: int = 5, atr_period: int = 14,
        level_tolerance_atr: float = 0.3,
    ) -> None:
        self.swing_window = swing_window
        self.atr_period = atr_period
        self.level_tolerance_atr = level_tolerance_atr

    def detect_all(self, df: pd.DataFrame) -> PatternDetectionResult:
        """Run all pattern detectors on `df`."""
        if df is None or len(df) < 3 * self.swing_window + 5:
            return PatternDetectionResult(notes=["insufficient data"])
        atr_series = self._atr(df, self.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 1.0
        if atr <= 0:
            atr = 1.0
        tol = atr * self.level_tolerance_atr

        swing_highs = self._swing_highs(df["high"].to_numpy(dtype=float), self.swing_window)
        swing_lows = self._swing_lows(df["low"].to_numpy(dtype=float), self.swing_window)

        patterns: list[Pattern] = []
        patterns.extend(self._detect_head_and_shoulders(swing_highs, swing_lows, atr, tol))
        patterns.extend(self._detect_inverse_head_and_shoulders(swing_highs, swing_lows, atr, tol))
        patterns.extend(self._detect_double_top(swing_highs, atr, tol))
        patterns.extend(self._detect_double_bottom(swing_lows, atr, tol))
        patterns.extend(self._detect_triple_top(swing_highs, atr, tol))
        patterns.extend(self._detect_triple_bottom(swing_lows, atr, tol))
        patterns.extend(self._detect_triangle(df, atr, tol))
        patterns.extend(self._detect_wedge(df, atr, tol))
        patterns.extend(self._detect_flag(df, atr, tol))
        patterns.extend(self._detect_rectangle(df, atr, tol))

        detected = [p for p in patterns if p.detected]
        notes = [f"scanned {len(patterns)} patterns, {len(detected)} detected"]
        if detected:
            notes.append("detected: " + ", ".join(p.pattern_type for p in detected))
        return PatternDetectionResult(patterns=patterns, notes=notes)

    # ──────────────────────────────────────────────────────────────────
    # Individual pattern detectors
    # ──────────────────────────────────────────────────────────────────
    def _detect_head_and_shoulders(
        self, highs: list[float], lows: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        """Head & Shoulders: LS - H - RS where H is highest."""
        if len(highs) < 3:
            return [Pattern("head_and_shoulders", "bearish", False, 0, {})]
        # Find last 3 swing highs
        last3 = highs[-3:]
        ls, h, rs = last3
        # H must be highest
        if not (h > ls and h > rs):
            return [Pattern("head_and_shoulders", "bearish", False, 0, {})]
        # LS and RS should be approximately equal (within tolerance)
        if abs(ls - rs) > tol:
            return [Pattern("head_and_shoulders", "bearish", False, 0, {})]
        # Neckline = the low between LS-H and H-RS
        if len(lows) < 2:
            return [Pattern("head_and_shoulders", "bearish", False, 0, {})]
        neckline = min(lows[-2], lows[-1])
        # Projected target: neckline - (H - neckline)
        target = neckline - (h - neckline)
        confidence = 0.6 + 0.4 * (1 - abs(ls - rs) / max(tol, 1e-10))
        return [Pattern(
            pattern_type="head_and_shoulders", direction="bearish",
            detected=True, confidence=float(min(1.0, confidence)),
            key_levels={"left_shoulder": ls, "head": h, "right_shoulder": rs, "neckline": neckline},
            projected_target=float(target),
            notes=f"LS={ls:.2f} H={h:.2f} RS={rs:.2f} neckline={neckline:.2f} target={target:.2f}",
        )]

    def _detect_inverse_head_and_shoulders(
        self, highs: list[float], lows: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        """Inverse Head & Shoulders: LS - H - RS where H is lowest."""
        if len(lows) < 3:
            return [Pattern("inverse_head_and_shoulders", "bullish", False, 0, {})]
        last3 = lows[-3:]
        ls, h, rs = last3
        if not (h < ls and h < rs):
            return [Pattern("inverse_head_and_shoulders", "bullish", False, 0, {})]
        if abs(ls - rs) > tol:
            return [Pattern("inverse_head_and_shoulders", "bullish", False, 0, {})]
        if len(highs) < 2:
            return [Pattern("inverse_head_and_shoulders", "bullish", False, 0, {})]
        neckline = max(highs[-2], highs[-1])
        target = neckline + (neckline - h)
        confidence = 0.6 + 0.4 * (1 - abs(ls - rs) / max(tol, 1e-10))
        return [Pattern(
            pattern_type="inverse_head_and_shoulders", direction="bullish",
            detected=True, confidence=float(min(1.0, confidence)),
            key_levels={"left_shoulder": ls, "head": h, "right_shoulder": rs, "neckline": neckline},
            projected_target=float(target),
            notes=f"LS={ls:.2f} H={h:.2f} RS={rs:.2f} neckline={neckline:.2f} target={target:.2f}",
        )]

    def _detect_double_top(
        self, highs: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        """Double Top: two roughly equal highs with a trough between."""
        if len(highs) < 3:
            return [Pattern("double_top", "bearish", False, 0, {})]
        last3 = highs[-3:]
        h1, trough, h2 = last3
        if abs(h1 - h2) <= tol and h1 > trough and h2 > trough:
            target = trough - (h1 - trough)
            confidence = 0.7
            return [Pattern(
                pattern_type="double_top", direction="bearish",
                detected=True, confidence=confidence,
                key_levels={"high1": h1, "trough": trough, "high2": h2, "neckline": trough},
                projected_target=float(target),
                notes=f"h1={h1:.2f} trough={trough:.2f} h2={h2:.2f} target={target:.2f}",
            )]
        return [Pattern("double_top", "bearish", False, 0, {})]

    def _detect_double_bottom(
        self, lows: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        if len(lows) < 3:
            return [Pattern("double_bottom", "bullish", False, 0, {})]
        last3 = lows[-3:]
        l1, peak, l2 = last3
        if abs(l1 - l2) <= tol and l1 < peak and l2 < peak:
            target = peak + (peak - l1)
            return [Pattern(
                pattern_type="double_bottom", direction="bullish",
                detected=True, confidence=0.7,
                key_levels={"low1": l1, "peak": peak, "low2": l2, "neckline": peak},
                projected_target=float(target),
                notes=f"l1={l1:.2f} peak={peak:.2f} l2={l2:.2f} target={target:.2f}",
            )]
        return [Pattern("double_bottom", "bullish", False, 0, {})]

    def _detect_triple_top(
        self, highs: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        if len(highs) < 5:
            return [Pattern("triple_top", "bearish", False, 0, {})]
        last5 = highs[-5:]
        # Pattern: h1 - trough1 - h2 - trough2 - h3, all h ~ equal
        h1, t1, h2, t2, h3 = last5
        if (abs(h1 - h2) <= tol and abs(h2 - h3) <= tol and
                h1 > t1 and h2 > t1 and h2 > t2 and h3 > t2):
            neckline = min(t1, t2)
            target = neckline - (h1 - neckline)
            return [Pattern(
                pattern_type="triple_top", direction="bearish",
                detected=True, confidence=0.8,
                key_levels={"high1": h1, "high2": h2, "high3": h3, "neckline": neckline},
                projected_target=float(target),
            )]
        return [Pattern("triple_top", "bearish", False, 0, {})]

    def _detect_triple_bottom(
        self, lows: list[float], atr: float, tol: float,
    ) -> list[Pattern]:
        if len(lows) < 5:
            return [Pattern("triple_bottom", "bullish", False, 0, {})]
        last5 = lows[-5:]
        l1, p1, l2, p2, l3 = last5
        if (abs(l1 - l2) <= tol and abs(l2 - l3) <= tol and
                l1 < p1 and l2 < p1 and l2 < p2 and l3 < p2):
            neckline = max(p1, p2)
            target = neckline + (neckline - l1)
            return [Pattern(
                pattern_type="triple_bottom", direction="bullish",
                detected=True, confidence=0.8,
                key_levels={"low1": l1, "low2": l2, "low3": l3, "neckline": neckline},
                projected_target=float(target),
            )]
        return [Pattern("triple_bottom", "bullish", False, 0, {})]

    def _detect_triangle(
        self, df: pd.DataFrame, atr: float, tol: float,
    ) -> list[Pattern]:
        """Detect ascending / descending / symmetrical triangles."""
        if len(df) < 20:
            return [Pattern("triangle", "neutral", False, 0, {})]
        recent = df.tail(20)
        highs = recent["high"].to_numpy(dtype=float)
        lows = recent["low"].to_numpy(dtype=float)
        # Linear regression on swing highs and lows
        x = np.arange(len(recent))
        try:
            high_slope = float(np.polyfit(x, highs, 1)[0])
            low_slope = float(np.polyfit(x, lows, 1)[0])
        except Exception:
            return [Pattern("triangle", "neutral", False, 0, {})]
        # Classify
        if abs(high_slope) < 0.1 * atr and low_slope > 0.3 * atr:
            # Flat top, rising bottom → ascending triangle (bullish)
            resistance = float(highs.mean())
            target = resistance + (resistance - float(lows.min()))
            return [Pattern(
                pattern_type="ascending_triangle", direction="bullish",
                detected=True, confidence=0.65,
                key_levels={"resistance": resistance, "rising_support_slope": low_slope},
                projected_target=target,
                notes=f"flat top, rising bottom (slope {low_slope:.4f})",
            )]
        if abs(low_slope) < 0.1 * atr and high_slope < -0.3 * atr:
            support = float(lows.mean())
            target = support - (float(highs.max()) - support)
            return [Pattern(
                pattern_type="descending_triangle", direction="bearish",
                detected=True, confidence=0.65,
                key_levels={"support": support, "falling_resistance_slope": high_slope},
                projected_target=target,
            )]
        if high_slope < -0.2 * atr and low_slope > 0.2 * atr:
            # Converging → symmetrical triangle
            return [Pattern(
                pattern_type="symmetrical_triangle", direction="neutral",
                detected=True, confidence=0.55,
                key_levels={"apex_high_slope": high_slope, "apex_low_slope": low_slope},
                notes="converging — breakout direction TBD",
            )]
        return [Pattern("triangle", "neutral", False, 0, {})]

    def _detect_wedge(
        self, df: pd.DataFrame, atr: float, tol: float,
    ) -> list[Pattern]:
        if len(df) < 20:
            return [Pattern("wedge", "neutral", False, 0, {})]
        recent = df.tail(20)
        highs = recent["high"].to_numpy(dtype=float)
        lows = recent["low"].to_numpy(dtype=float)
        x = np.arange(len(recent))
        try:
            high_slope = float(np.polyfit(x, highs, 1)[0])
            low_slope = float(np.polyfit(x, lows, 1)[0])
        except Exception:
            return [Pattern("wedge", "neutral", False, 0, {})]
        # Rising wedge: both slopes positive, high slope < low slope (converging upward)
        if high_slope > 0.2 * atr and low_slope > 0.2 * atr and low_slope > high_slope:
            return [Pattern(
                pattern_type="rising_wedge", direction="bearish",
                detected=True, confidence=0.6,
                key_levels={"high_slope": high_slope, "low_slope": low_slope},
                notes="rising wedge — typically breaks down",
            )]
        # Falling wedge: both slopes negative
        if high_slope < -0.2 * atr and low_slope < -0.2 * atr and high_slope < low_slope:
            return [Pattern(
                pattern_type="falling_wedge", direction="bullish",
                detected=True, confidence=0.6,
                key_levels={"high_slope": high_slope, "low_slope": low_slope},
                notes="falling wedge — typically breaks up",
            )]
        return [Pattern("wedge", "neutral", False, 0, {})]

    def _detect_flag(
        self, df: pd.DataFrame, atr: float, tol: float,
    ) -> list[Pattern]:
        """Bull/bear flag: strong move (pole) followed by small counter-trend channel."""
        if len(df) < 15:
            return [Pattern("flag", "neutral", False, 0, {})]
        # Pole = first 5 bars, flag = next 10 bars
        pole = df.iloc[:5]
        flag = df.iloc[5:15]
        pole_move = float(pole["close"].iloc[-1] - pole["open"].iloc[0])
        flag_highs = flag["high"].to_numpy(dtype=float)
        flag_lows = flag["low"].to_numpy(dtype=float)
        x = np.arange(len(flag))
        try:
            flag_high_slope = float(np.polyfit(x, flag_highs, 1)[0])
            flag_low_slope = float(np.polyfit(x, flag_lows, 1)[0])
        except Exception:
            return [Pattern("flag", "neutral", False, 0, {})]
        # Bull flag: pole up, flag slight downward
        if pole_move > 2 * atr and flag_high_slope < 0 and flag_low_slope < 0:
            target = float(flag["close"].iloc[-1]) + pole_move
            return [Pattern(
                pattern_type="bull_flag", direction="bullish",
                detected=True, confidence=0.6,
                key_levels={"pole_high": float(pole["high"].max()), "flag_low": float(flag["low"].min())},
                projected_target=target,
            )]
        # Bear flag: pole down, flag slight upward
        if pole_move < -2 * atr and flag_high_slope > 0 and flag_low_slope > 0:
            target = float(flag["close"].iloc[-1]) + pole_move
            return [Pattern(
                pattern_type="bear_flag", direction="bearish",
                detected=True, confidence=0.6,
                key_levels={"pole_low": float(pole["low"].min()), "flag_high": float(flag["high"].max())},
                projected_target=target,
            )]
        return [Pattern("flag", "neutral", False, 0, {})]

    def _detect_rectangle(
        self, df: pd.DataFrame, atr: float, tol: float,
    ) -> list[Pattern]:
        """Rectangle: price bouncing between horizontal support and resistance."""
        if len(df) < 20:
            return [Pattern("rectangle", "neutral", False, 0, {})]
        recent = df.tail(20)
        highs = recent["high"].to_numpy(dtype=float)
        lows = recent["low"].to_numpy(dtype=float)
        # Check if highs are flat and lows are flat
        high_var = float(highs.std() / max(highs.mean(), 1e-10))
        low_var = float(lows.std() / max(lows.mean(), 1e-10))
        if high_var < 0.01 and low_var < 0.01:
            resistance = float(highs.mean())
            support = float(lows.mean())
            if resistance - support > 2 * atr:
                return [Pattern(
                    pattern_type="rectangle", direction="neutral",
                    detected=True, confidence=0.55,
                    key_levels={"resistance": resistance, "support": support},
                    notes="range-bound — wait for breakout",
                )]
        return [Pattern("rectangle", "neutral", False, 0, {})]

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _swing_highs(self, highs: np.ndarray, k: int) -> list[float]:
        swings = []
        n = len(highs)
        for i in range(k, n - k):
            window = highs[i - k:i + k + 1]
            if highs[i] == window.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swings.append(float(highs[i]))
        return swings[-10:]  # last 10

    def _swing_lows(self, lows: np.ndarray, k: int) -> list[float]:
        swings = []
        n = len(lows)
        for i in range(k, n - k):
            window = lows[i - k:i + k + 1]
            if lows[i] == window.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swings.append(float(lows[i]))
        return swings[-10:]

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


__all__ = ["ChartPatternDetector", "Pattern", "PatternDetectionResult"]
