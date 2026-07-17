"""engine.candlestick.rejection_strength
=====================================================================
Day 127 — Rejection strength scorer.

Quantifies how strongly a bar rejects a price level. A pin bar with
a long wick is "rejection" — but how strong is that rejection?

Components:
  - Upper wick % of range
  - Lower wick % of range
  - Body % of range
  - ATR % (bar size relative to recent volatility)
  - Volume multiplier (vs. recent average)

Output: RejectionScore in [0, 100] per bar, with directional bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.indicators import atr


@dataclass
class RejectionScore:
    score: float                    # 0-100
    direction: str                  # "bullish" | "bearish" | "neutral"
    upper_wick_pct: float
    lower_wick_pct: float
    body_pct: float
    atr_ratio: float
    volume_mult: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "direction": self.direction,
            "upper_wick_pct": self.upper_wick_pct,
            "lower_wick_pct": self.lower_wick_pct,
            "body_pct": self.body_pct,
            "atr_ratio": self.atr_ratio,
            "volume_mult": self.volume_mult,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class RejectionStrengthScorer:
    def __init__(self, atr_period: int = 14, volume_period: int = 20) -> None:
        self.atr_period = int(atr_period)
        self.volume_period = int(volume_period)

    # ----------------------------------------------------------------
    def score(self, df: pd.DataFrame, index: int = -1) -> RejectionScore:
        """Score the bar at `index` (default: last bar)."""
        if len(df) < max(self.atr_period, self.volume_period, 5):
            return RejectionScore(
                score=0.0, direction="neutral",
                upper_wick_pct=0.0, lower_wick_pct=0.0,
                body_pct=0.0, atr_ratio=0.0, volume_mult=1.0,
            )
        if index < 0:
            index = len(df) + index
        bar = df.iloc[index]
        open_, high, low, close = (
            float(bar["open"]), float(bar["high"]),
            float(bar["low"]), float(bar["close"]),
        )
        range_ = high - low
        if range_ <= 0:
            return RejectionScore(
                score=0.0, direction="neutral",
                upper_wick_pct=0.0, lower_wick_pct=0.0,
                body_pct=0.0, atr_ratio=0.0, volume_mult=1.0,
            )
        body = abs(close - open_)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        upper_pct = upper_wick / range_
        lower_pct = lower_wick / range_
        body_pct = body / range_

        # ATR context
        atr_series = atr(df, self.atr_period)
        atr_val = float(atr_series.iloc[index]) if not atr_series.isna().iloc[index] else range_
        atr_ratio = range_ / atr_val if atr_val > 0 else 1.0

        # Volume multiplier
        vol_mult = 1.0
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(self.volume_period, min_periods=5).mean()
            if not vol_mean.isna().iloc[index] and vol_mean.iloc[index] > 0:
                vol_mult = float(bar["volume"] / vol_mean.iloc[index])

        # Score: rejection is strong when one wick dominates AND body is small
        # AND bar is ATR-significant AND volume confirms
        dominant_wick = max(upper_pct, lower_pct)
        wick_score = min(1.0, dominant_wick / 0.7) * 35        # 0-35
        body_penalty = body_pct * 25                            # 0-25 penalty
        atr_score = min(1.0, atr_ratio / 2.0) * 20              # 0-20
        vol_score = min(1.0, max(0.0, (vol_mult - 0.5)) / 1.5) * 20  # 0-20

        score = wick_score - body_penalty + atr_score + vol_score
        score = float(max(0.0, min(100.0, score)))

        # Direction
        if lower_pct > upper_pct and lower_pct > 0.5:
            direction = "bullish"     # rejected lower prices
        elif upper_pct > lower_pct and upper_pct > 0.5:
            direction = "bearish"     # rejected higher prices
        else:
            direction = "neutral"

        return RejectionScore(
            score=score, direction=direction,
            upper_wick_pct=float(upper_pct),
            lower_wick_pct=float(lower_pct),
            body_pct=float(body_pct),
            atr_ratio=float(atr_ratio),
            volume_mult=float(vol_mult),
            components={
                "wick_score": float(wick_score),
                "body_penalty": float(body_penalty),
                "atr_score": float(atr_score),
                "vol_score": float(vol_score),
            },
        )
