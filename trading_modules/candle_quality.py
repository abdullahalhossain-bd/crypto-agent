"""
Candle Quality Analyzer — "Is this candle worth trading?"
============================================================

Not every bullish candle is a buy signal. Institutional traders inspect:

    1. Body size relative to ATR (body must be meaningful)
    2. Wick ratio (long wick = rejection = good for reversals)
    3. Close position (close near high = bullish conviction)
    4. Momentum vs prior candles (accelerating or decelerating)
    5. Body-to-range ratio (>= 0.6 = strong, < 0.3 = doji / indecision)

Output: a 0..1 quality score plus a qualitative label.

Usage:
    from trading_modules.candle_quality import CandleQualityAnalyzer

    analyzer = CandleQualityAnalyzer()
    cq = analyzer.analyze(df_m15, direction="BUY")

    if cq.score >= 0.7:
        # high-quality candle — entry confirmed
    elif cq.score < 0.3:
        # weak / indecision candle — skip
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CandleQuality:
    score: float                # 0..1 overall quality in trade direction
    label: str                  # "strong" / "good" / "weak" / "indecision" / "rejection"
    body_ratio: float           # body / range (0..1)
    upper_wick_ratio: float     # upper wick / range
    lower_wick_ratio: float     # lower wick / range
    close_position: float       # 0..1 — 1 = close at high, 0 = close at low
    body_atr_ratio: float       # body size in ATR units
    momentum_accel: float       # current body / avg body of last 5
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "body_ratio": round(self.body_ratio, 2),
            "upper_wick_ratio": round(self.upper_wick_ratio, 2),
            "lower_wick_ratio": round(self.lower_wick_ratio, 2),
            "close_position": round(self.close_position, 2),
            "body_atr_ratio": round(self.body_atr_ratio, 2),
            "momentum_accel": round(self.momentum_accel, 2),
            "notes": self.notes,
        }


class CandleQualityAnalyzer:
    """
    Analyze the last candle's quality in the context of a trade direction.

    Parameters:
        atr_period: ATR lookback (default 14)
        momentum_window: # of prior candles to compare body size (default 5)
        strong_body_ratio: body/range >= this = strong (default 0.6)
        weak_body_ratio: body/range < this = indecision (default 0.3)
        strong_body_atr: body in ATR units >= this = strong (default 0.6)
        rejection_wick_ratio: wick/range >= this = rejection (default 0.5)
    """

    def __init__(
        self,
        atr_period: int = 14,
        momentum_window: int = 5,
        strong_body_ratio: float = 0.6,
        weak_body_ratio: float = 0.3,
        strong_body_atr: float = 0.6,
        rejection_wick_ratio: float = 0.5,
    ) -> None:
        self.atr_period = atr_period
        self.momentum_window = momentum_window
        self.strong_body_ratio = strong_body_ratio
        self.weak_body_ratio = weak_body_ratio
        self.strong_body_atr = strong_body_atr
        self.rejection_wick_ratio = rejection_wick_ratio

    def analyze(self, df: pd.DataFrame, direction: str = "BUY") -> CandleQuality:
        """Analyze the last candle in `df` for `direction` (BUY or SELL)."""
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            direction = "BUY"

        if df is None or len(df) < max(self.atr_period + 5, self.momentum_window + 1):
            return CandleQuality(
                score=0.5, label="insufficient_data",
                body_ratio=0, upper_wick_ratio=0, lower_wick_ratio=0,
                close_position=0.5, body_atr_ratio=0, momentum_accel=1.0,
                notes=["insufficient data"],
            )

        # Last candle
        row = df.iloc[-1]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        rng = h - l
        if rng <= 0:
            return CandleQuality(
                score=0.0, label="invalid",
                body_ratio=0, upper_wick_ratio=0, lower_wick_ratio=0,
                close_position=0.5, body_atr_ratio=0, momentum_accel=0,
                notes=["zero-range candle"],
            )

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        body_ratio = body / rng
        upper_wick_ratio = upper_wick / rng
        lower_wick_ratio = lower_wick / rng
        close_position = (c - l) / rng  # 1 = close at high

        # ATR
        atr = self._atr(df, self.atr_period).iloc[-1]
        body_atr_ratio = body / atr if atr > 0 else 0

        # Momentum acceleration — current body vs avg of last N bodies
        recent = df.tail(self.momentum_window + 1).iloc[:-1]
        recent_bodies = (recent["close"] - recent["open"]).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body
        momentum_accel = body / avg_body if avg_body > 0 else 1.0

        # Score the candle in the direction of the trade
        score, label, notes = self._score(
            direction=direction,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            close_position=close_position,
            body_atr_ratio=body_atr_ratio,
            momentum_accel=momentum_accel,
            is_bullish=(c > o),
        )

        return CandleQuality(
            score=score, label=label,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            close_position=close_position,
            body_atr_ratio=body_atr_ratio,
            momentum_accel=momentum_accel,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _score(
        self,
        direction: str,
        body_ratio: float,
        upper_wick_ratio: float,
        lower_wick_ratio: float,
        close_position: float,
        body_atr_ratio: float,
        momentum_accel: float,
        is_bullish: bool,
    ) -> tuple[float, str, list[str]]:
        """Return (score 0..1, label, notes)."""
        notes: list[str] = []
        aligned = (
            (direction == "BUY" and is_bullish) or
            (direction == "SELL" and not is_bullish)
        )
        score = 0.0

        # Body ratio component (0..0.3)
        if body_ratio >= self.strong_body_ratio:
            score += 0.3
            notes.append(f"strong body ({body_ratio:.2f})")
        elif body_ratio >= self.weak_body_ratio:
            score += 0.15
            notes.append(f"moderate body ({body_ratio:.2f})")
        else:
            notes.append(f"weak body ({body_ratio:.2f}) — indecision")

        # Body in ATR terms (0..0.2)
        if body_atr_ratio >= self.strong_body_atr:
            score += 0.2
            notes.append(f"body {body_atr_ratio:.2f}x ATR")
        elif body_atr_ratio >= 0.3:
            score += 0.1

        # Close position (0..0.2)
        if direction == "BUY":
            if close_position >= 0.7:
                score += 0.2
                notes.append(f"close near high ({close_position:.2f})")
            elif close_position >= 0.5:
                score += 0.1
        else:  # SELL
            if close_position <= 0.3:
                score += 0.2
                notes.append(f"close near low ({close_position:.2f})")
            elif close_position <= 0.5:
                score += 0.1

        # Rejection wick (0..0.2) — opposite-side long wick = rejection
        if direction == "BUY" and lower_wick_ratio >= self.rejection_wick_ratio:
            score += 0.2
            notes.append(f"bullish rejection wick ({lower_wick_ratio:.2f})")
        elif direction == "SELL" and upper_wick_ratio >= self.rejection_wick_ratio:
            score += 0.2
            notes.append(f"bearish rejection wick ({upper_wick_ratio:.2f})")

        # Momentum acceleration (0..0.1)
        if momentum_accel >= 1.5:
            score += 0.1
            notes.append(f"accelerating momentum ({momentum_accel:.2f}x)")
        elif momentum_accel >= 1.0:
            score += 0.05

        # Direction alignment penalty
        if not aligned:
            score *= 0.4  # candle is opposite of trade direction — heavy penalty
            notes.append("candle opposite to direction — heavy penalty")

        # Label
        if score >= 0.7:
            label = "strong"
        elif score >= 0.5:
            label = "good"
        elif score >= 0.3:
            label = "weak"
        elif body_ratio < self.weak_body_ratio:
            label = "indecision"
        else:
            label = "rejection" if (upper_wick_ratio >= self.rejection_wick_ratio or
                                     lower_wick_ratio >= self.rejection_wick_ratio) else "weak"

        return min(1.0, score), label, notes

    # ------------------------------------------------------------------
    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
