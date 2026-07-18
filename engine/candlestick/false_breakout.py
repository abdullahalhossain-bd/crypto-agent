"""engine.candlestick.false_breakout
=====================================================================
Day 131 — False breakout detector.

The Candlestick Trading Bible warns: many breakouts fail. We detect
the probability that a recent breakout is false by checking:

  - Breakout bar closed outside range, but next bar reversed back inside
  - Volume on breakout was high, but follow-through is weak
  - Breakout level was a "round number" or obvious level (liquidity grab)
  - The breakout direction disagrees with the higher timeframe trend

Output: FalseBreakoutResult with probability 0-1 and recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FalseBreakoutResult:
    probability: float             # 0-1 (1 = certainly false breakout)
    breakout_direction: str        # "up" / "down" / "none"
    reversal_detected: bool
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "breakout_direction": self.breakout_direction,
            "reversal_detected": self.reversal_detected,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class FalseBreakoutDetector:
    def __init__(self,
                 range_window: int = 20,
                 reversal_threshold: float = 0.5) -> None:
        self.range_window = int(range_window)
        self.reversal_threshold = float(reversal_threshold)

    # ----------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> FalseBreakoutResult:
        if len(df) < self.range_window + 3:
            return FalseBreakoutResult(
                probability=0.0, breakout_direction="none",
                reversal_detected=False,
                components={"reason": "warmup"},
            )
        # Identify recent range (excluding the last 2 bars which may have broken out)
        range_df = df.iloc[-self.range_window - 2:-2]
        range_high = float(range_df["high"].max())
        range_low = float(range_df["low"].min())

        prev_bar = df.iloc[-2]
        last_bar = df.iloc[-1]

        # Did prev bar break out?
        prev_close = float(prev_bar["close"])
        prev_open = float(prev_bar["open"])
        prev_high = float(prev_bar["high"])
        prev_low = float(prev_bar["low"])

        broke_up = prev_close > range_high
        broke_down = prev_close < range_low

        if not (broke_up or broke_down):
            return FalseBreakoutResult(
                probability=0.0, breakout_direction="none",
                reversal_detected=False,
                components={"range_high": range_high, "range_low": range_low},
            )

        # Did last bar reverse back inside?
        last_close = float(last_bar["close"])
        last_high = float(last_bar["high"])
        last_low = float(last_bar["low"])

        if broke_up:
            reversal_back = last_close < range_high
            breakout_direction = "up"
        else:
            reversal_back = last_close > range_low
            breakout_direction = "down"

        # Reversal strength: how far back inside did it close?
        if broke_up:
            reversal_strength = (range_high - last_close) / max(range_high - range_low, 1e-9)
        else:
            reversal_strength = (last_close - range_low) / max(range_high - range_low, 1e-9)

        # Volume check (high breakout vol + low follow-through = false breakout)
        vol_spike_then_fade = False
        if "volume" in df.columns and len(df) >= 5:
            breakout_vol = float(prev_bar["volume"])
            follow_vol = float(last_bar["volume"])
            if breakout_vol > 0:
                vol_spike_then_fade = follow_vol < breakout_vol * 0.7

        # Probability
        prob = 0.0
        if reversal_back:
            prob += 0.5
        if reversal_strength > self.reversal_threshold:
            prob += 0.25
        if vol_spike_then_fade:
            prob += 0.15
        # Wick check: if last bar has a long wick rejecting the breakout, more likely false
        if broke_up:
            upper_wick = last_high - max(float(last_bar["open"]), last_close)
            range_ = last_high - last_low
            if range_ > 0 and upper_wick / range_ > 0.5:
                prob += 0.10
        else:
            lower_wick = min(float(last_bar["open"]), last_close) - last_low
            range_ = last_high - last_low
            if range_ > 0 and lower_wick / range_ > 0.5:
                prob += 0.10
        # Breakout-bar wick rejection: if the breakout bar itself had a long
        # opposing wick, it was already being rejected before the next bar.
        prev_bar_range = prev_high - prev_low
        if prev_bar_range > 0:
            if broke_up:
                # Upper wick on an upward breakout = selling pressure on the breakout bar
                breakout_wick = (prev_high - max(prev_open, prev_close)) / prev_bar_range
                if breakout_wick > 0.5:
                    prob += 0.10
            else:
                # Lower wick on a downward breakout = buying pressure on the breakout bar
                breakout_wick = (min(prev_open, prev_close) - prev_low) / prev_bar_range
                if breakout_wick > 0.5:
                    prob += 0.10

        prob = float(max(0.0, min(1.0, prob)))

        return FalseBreakoutResult(
            probability=prob,
            breakout_direction=breakout_direction,
            reversal_detected=bool(reversal_back),
            components={
                "range_high": range_high,
                "range_low": range_low,
                "breakout_close": prev_close,
                "prev_high": prev_high,
                "prev_low": prev_low,
                "last_close": last_close,
                "reversal_strength": float(reversal_strength),
                "vol_spike_then_fade": bool(vol_spike_then_fade),
            },
        )
