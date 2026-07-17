"""engine.candlestick.pullback_detector
=====================================================================
Day 130 — Pullback detection within trends.

The book's core insight: trades taken on pullbacks (not breakouts)
have better risk/reward. We detect:
  - Trend is established (HH/HL or LH/LL pattern)
  - Price has pulled back to a value area (MA, fib zone, prior swing)
  - Pullback is shallow (≤ 50% of prior impulse)
  - Pullback shows signs of exhaustion (smaller bars, lower volume)

Output: PullbackResult with quality score + entry suggestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.indicators import atr, sma


@dataclass
class PullbackResult:
    detected: bool
    quality: float                # 0-100
    pullback_depth_pct: float     # how deep (0 = no pullback, 1 = full retracement)
    entry_zone_low: float
    entry_zone_high: float
    direction: str                # "bullish" / "bearish" / "none"
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "quality": self.quality,
            "pullback_depth_pct": self.pullback_depth_pct,
            "entry_zone_low": self.entry_zone_low,
            "entry_zone_high": self.entry_zone_high,
            "direction": self.direction,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class PullbackDetector:
    def __init__(self,
                 trend_window: int = 50,
                 pullback_window: int = 10,
                 max_pullback_pct: float = 0.618,    # fib 61.8% max
                 min_pullback_pct: float = 0.20,
                 ma_period: int = 20) -> None:
        self.trend_window = int(trend_window)
        self.pullback_window = int(pullback_window)
        self.max_pullback_pct = float(max_pullback_pct)
        self.min_pullback_pct = float(min_pullback_pct)
        self.ma_period = int(ma_period)

    # ----------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> PullbackResult:
        if len(df) < self.trend_window + self.pullback_window:
            return PullbackResult(
                detected=False, quality=0.0,
                pullback_depth_pct=0.0,
                entry_zone_low=0.0, entry_zone_high=0.0,
                direction="none",
                components={"reason": "warmup"},
            )
        close = df["close"]
        # Detect trend direction
        recent = close.tail(self.trend_window)
        x = np.arange(len(recent), dtype=float)
        denom = ((x - x.mean()) ** 2).sum()
        slope = float(((x - x.mean()) * (recent.values - recent.values.mean())).sum()
                      / denom) if denom > 0 else 0.0
        is_uptrend = slope > 0
        is_downtrend = slope < 0
        if not (is_uptrend or is_downtrend):
            return PullbackResult(
                detected=False, quality=0.0,
                pullback_depth_pct=0.0,
                entry_zone_low=float(close.iloc[-1]),
                entry_zone_high=float(close.iloc[-1]),
                direction="none",
                components={"slope": slope},
            )

        # Find recent swing high (uptrend) or swing low (downtrend)
        window = df.tail(self.pullback_window + 5)
        if is_uptrend:
            swing_high_idx = int(window["high"].idxmax())
            swing_high = float(window.loc[swing_high_idx, "high"])
            current = float(close.iloc[-1])
            # Pullback depth
            if swing_high > 0:
                pullback_depth = (swing_high - current) / swing_high
            else:
                pullback_depth = 0.0
            direction = "bullish"
            entry_zone_low = current * 0.995
            entry_zone_high = current * 1.005
        else:
            swing_low_idx = int(window["low"].idxmin())
            swing_low = float(window.loc[swing_low_idx, "low"])
            current = float(close.iloc[-1])
            if swing_low > 0:
                pullback_depth = (current - swing_low) / swing_low
            else:
                pullback_depth = 0.0
            direction = "bearish"
            entry_zone_low = current * 0.995
            entry_zone_high = current * 1.005

        # Check if pullback is in valid zone
        valid_depth = self.min_pullback_pct <= pullback_depth <= self.max_pullback_pct

        # MA confluence: is price near the MA?
        ma_series = sma(close, self.ma_period)
        ma_val = float(ma_series.iloc[-1]) if not ma_series.isna().iloc[-1] else current
        atr_series = atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.001
        ma_distance = abs(current - ma_val) / max(atr_val, 1e-9)
        near_ma = ma_distance < 1.0

        # Volume contraction (pullback should have lower volume)
        vol_contraction = False
        if "volume" in df.columns and len(df) >= 20:
            recent_vol = df["volume"].tail(self.pullback_window).mean()
            baseline_vol = df["volume"].tail(self.trend_window).mean()
            if baseline_vol > 0:
                vol_contraction = recent_vol < baseline_vol * 0.8

        # Quality score
        depth_score = 50.0 if valid_depth else 0.0
        depth_score += (1.0 - abs(pullback_depth - 0.382) / 0.382) * 25 if valid_depth else 0  # 38.2% fib is ideal
        ma_score = 15.0 if near_ma else 0.0
        vol_score = 10.0 if vol_contraction else 0.0
        quality = float(max(0.0, min(100.0, depth_score + ma_score + vol_score)))

        detected = valid_depth and (near_ma or vol_contraction)

        return PullbackResult(
            detected=detected,
            quality=quality,
            pullback_depth_pct=float(pullback_depth),
            entry_zone_low=float(entry_zone_low),
            entry_zone_high=float(entry_zone_high),
            direction=direction,
            components={
                "slope": float(slope),
                "swing_high": float(swing_high) if is_uptrend else None,
                "swing_low": float(swing_low) if not is_uptrend else None,
                "ma_value": float(ma_val),
                "ma_distance_atr": float(ma_distance),
                "near_ma": bool(near_ma),
                "volume_contraction": bool(vol_contraction),
                "valid_depth": bool(valid_depth),
            },
        )
