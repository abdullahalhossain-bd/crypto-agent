"""
EMA Ribbon Analyzer — trend strength via EMA stack structure
=============================================================

A single EMA cross is noisy. A ribbon of multiple EMAs reveals:

    1. EMA separation  — distance between fast and slow EMAs (trend strength)
    2. EMA compression — EMAs converging (trend exhaustion / pending breakout)
    3. EMA slope       — direction of the slow EMA (trend direction)
    4. EMA stack order — bullish (f>s) or bearish (f<s) or mixed
    5. Price position  — above all / below all / mixed

The ribbon uses 8 EMAs: 5, 8, 13, 21, 34, 55, 89, 144 (Fibonacci-tuned).
A perfectly stacked bullish ribbon has EMA5 > EMA8 > ... > EMA144.

Usage:
    from trading_modules.ema_ribbon import EMARibbonAnalyzer
    analyzer = EMARibbonAnalyzer()
    result = analyzer.analyze(df_m15)
    if result.fully_stacked_bull and result.compression_ratio > 0.6:
        # breakout setup — compressed + stacked
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PERIODS = (5, 8, 13, 21, 34, 55, 89, 144)


@dataclass
class EMARibbonResult:
    stack_order: str                # "bull" / "bear" / "mixed"
    fully_stacked_bull: bool        # EMA5 > EMA8 > ... > EMA144
    fully_stacked_bear: bool        # EMA5 < EMA8 < ... < EMA144
    separation_fast_slow: float     # (fast - slow) / slow — trend strength
    separation_pct: float           # as percent
    compression_ratio: float        # 0..1 — 1 = fully compressed (all EMAs equal)
    slow_ema_slope: float           # slope of slowest EMA over last N bars
    slope_direction: str            # "up" / "down" / "flat"
    price_above_all: bool
    price_below_all: bool
    price_position: str             # "above_all" / "below_all" / "mixed"
    ribbon_values: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stack_order": self.stack_order,
            "fully_stacked_bull": self.fully_stacked_bull,
            "fully_stacked_bear": self.fully_stacked_bear,
            "separation_fast_slow": round(self.separation_fast_slow, 4),
            "separation_pct": round(self.separation_pct, 2),
            "compression_ratio": round(self.compression_ratio, 3),
            "slow_ema_slope": round(self.slow_ema_slope, 4),
            "slope_direction": self.slope_direction,
            "price_above_all": self.price_above_all,
            "price_below_all": self.price_below_all,
            "price_position": self.price_position,
            "ribbon_values": {k: round(v, 4) for k, v in self.ribbon_values.items()},
            "notes": self.notes,
        }


class EMARibbonAnalyzer:
    """Analyze EMA ribbon structure for trend strength and compression.

    Parameters:
        periods: tuple of EMA periods (default Fibonacci-tuned 5..144)
        slope_lookback: bars to measure slope (default 10)
        compression_threshold: separation below this * slow = compressed (default 0.02)
    """

    def __init__(
        self, periods: tuple[int, ...] = DEFAULT_PERIODS,
        slope_lookback: int = 10, compression_threshold: float = 0.02,
    ) -> None:
        if len(periods) < 2:
            raise ValueError("need at least 2 EMA periods")
        if any(p < 1 for p in periods):
            raise ValueError("all periods must be >= 1")
        self.periods = tuple(periods)
        self.slope_lookback = slope_lookback
        self.compression_threshold = compression_threshold

    def analyze(self, df: pd.DataFrame) -> EMARibbonResult:
        if df is None or len(df) < max(self.periods) + self.slope_lookback + 5:
            return EMARibbonResult(
                stack_order="mixed", slope_direction="flat",
                price_position="mixed", notes=["insufficient data"],
            )

        close = df["close"]
        ema_values: list[float] = []
        ema_series_dict: dict[int, pd.Series] = {}
        for p in self.periods:
            s = close.ewm(span=p, adjust=False).mean()
            ema_series_dict[p] = s
            ema_values.append(float(s.iloc[-1]))

        # Stack order: are EMAs monotonically decreasing (bull) or increasing (bear)?
        bull_stack = all(
            ema_values[i] > ema_values[i + 1] for i in range(len(ema_values) - 1)
        )
        bear_stack = all(
            ema_values[i] < ema_values[i + 1] for i in range(len(ema_values) - 1)
        )
        if bull_stack:
            stack_order = "bull"
        elif bear_stack:
            stack_order = "bear"
        else:
            stack_order = "mixed"

        # Separation between fastest and slowest
        fast = ema_values[0]
        slow = ema_values[-1]
        separation = (fast - slow) / slow if slow != 0 else 0.0
        separation_pct = separation * 100

        # Compression ratio: 1 - (max-min) / slow — high when all EMAs are clustered
        ema_max = max(ema_values)
        ema_min = min(ema_values)
        spread = ema_max - ema_min
        compression_ratio = 1.0 - (spread / slow if slow != 0 else 0.0)
        compression_ratio = max(0.0, min(1.0, compression_ratio))

        # Slow EMA slope
        slow_series = ema_series_dict[self.periods[-1]]
        if len(slow_series) > self.slope_lookback:
            slope = (slow_series.iloc[-1] - slow_series.iloc[-self.slope_lookback]) / \
                    slow_series.iloc[-self.slope_lookback]
        else:
            slope = 0.0
        if abs(slope) < self.compression_threshold:
            slope_direction = "flat"
        elif slope > 0:
            slope_direction = "up"
        else:
            slope_direction = "down"

        # Price position
        last_close = float(close.iloc[-1])
        price_above_all = all(last_close > v for v in ema_values)
        price_below_all = all(last_close < v for v in ema_values)
        if price_above_all:
            price_position = "above_all"
        elif price_below_all:
            price_position = "below_all"
        else:
            price_position = "mixed"

        notes: list[str] = []
        if bull_stack:
            notes.append("fully stacked bullish ribbon")
        if bear_stack:
            notes.append("fully stacked bearish ribbon")
        if compression_ratio >= 0.9:
            notes.append(f"extreme compression ({compression_ratio:.2f}) — breakout pending")
        elif compression_ratio >= 0.75:
            notes.append(f"compressed ({compression_ratio:.2f})")
        if abs(separation) >= 0.05:
            notes.append(f"strong separation ({separation_pct:.2f}%) — strong trend")
        if slope_direction != "flat":
            notes.append(f"slow EMA slope {slope_direction}")

        return EMARibbonResult(
            stack_order=stack_order,
            fully_stacked_bull=bool(bull_stack),
            fully_stacked_bear=bool(bear_stack),
            separation_fast_slow=float(separation),
            separation_pct=float(separation_pct),
            compression_ratio=float(compression_ratio),
            slow_ema_slope=float(slope),
            slope_direction=slope_direction,
            price_above_all=bool(price_above_all),
            price_below_all=bool(price_below_all),
            price_position=price_position,
            ribbon_values={str(p): float(v) for p, v in zip(self.periods, ema_values)},
            notes=notes,
        )


__all__ = ["EMARibbonAnalyzer", "EMARibbonResult", "DEFAULT_PERIODS"]
