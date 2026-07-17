"""
Wyckoff Logic Analyzer — institutional accumulation/distribution detection
==========================================================================

Wyckoff theory identifies where institutional money is accumulating or
distributing an asset, before a major move. This module detects simplified
versions of the classic Wyckoff phases and events:

Events detected:
    - Spring       — false breakdown below support, then close back above (bullish)
    - Upthrust     — false breakout above resistance, then close back below (bearish)
    - SOS (Sign of Strength)  — strong up-move on high volume after consolidation
    - SOW (Sign of Weakness)  — strong down-move on high volume after consolidation
    - Accumulation — sideways range with positive volume bias (spring + SOS combo)
    - Distribution — sideways range with negative volume bias (upthrust + SOW combo)

The detector is heuristic — it does not perform full Wyckoff phase
classification (which requires multi-week context). It focuses on the most
actionable short-term patterns.

Usage:
    from trading_modules.wyckoff import WyckoffAnalyzer
    analyzer = WyckoffAnalyzer()
    result = analyzer.analyze(df_m15, direction="BUY")
    if result.spring_detected:
        # bullish reversal from accumulation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WyckoffResult:
    spring_detected: bool = False
    upthrust_detected: bool = False
    sos_detected: bool = False           # Sign of Strength
    sow_detected: bool = False           # Sign of Weakness
    accumulation_likely: bool = False
    distribution_likely: bool = False
    in_consolidation: bool = False
    consolidation_range: Optional[tuple[float, float]] = None
    volume_bias: str = "neutral"         # "bull" / "bear" / "neutral"
    confidence: float = 0.0              # 0..1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spring_detected": self.spring_detected,
            "upthrust_detected": self.upthrust_detected,
            "sos_detected": self.sos_detected,
            "sow_detected": self.sow_detected,
            "accumulation_likely": self.accumulation_likely,
            "distribution_likely": self.distribution_likely,
            "in_consolidation": self.in_consolidation,
            "consolidation_range": list(self.consolidation_range)
                if self.consolidation_range else None,
            "volume_bias": self.volume_bias,
            "confidence": round(self.confidence, 2),
            "notes": self.notes,
        }


class WyckoffAnalyzer:
    """
    Detect Wyckoff events (spring/upthrust/SOS/SOW) and classify the
    current market as accumulation or distribution.

    Parameters:
        consolidation_lookback: bars to identify the trading range (default 50)
        consolidation_atr_threshold: range width / ATR below this = consolidation (default 4.0)
        breakout_atr_multiple: wick beyond range by this much = spring/upthrust (default 0.2)
        volume_spike_ratio: volume / avg volume above this = SOS/SOW (default 1.8)
        body_atr_ratio: body / ATR above this = strong move (default 0.8)
        atr_period: ATR lookback (default 14)
    """

    def __init__(
        self, consolidation_lookback: int = 50,
        consolidation_atr_threshold: float = 4.0,
        breakout_atr_multiple: float = 0.2,
        volume_spike_ratio: float = 1.8,
        body_atr_ratio: float = 0.8,
        atr_period: int = 14,
    ) -> None:
        self.consolidation_lookback = consolidation_lookback
        self.consolidation_atr_threshold = consolidation_atr_threshold
        self.breakout_atr_multiple = breakout_atr_multiple
        self.volume_spike_ratio = volume_spike_ratio
        self.body_atr_ratio = body_atr_ratio
        self.atr_period = atr_period

    def analyze(
        self, df: pd.DataFrame, direction: str = "BUY",
    ) -> WyckoffResult:
        if df is None or len(df) < max(self.consolidation_lookback, self.atr_period + 10):
            return WyckoffResult()
        direction = direction.upper()

        recent = df.tail(self.consolidation_lookback).reset_index(drop=True)
        atr_series = self._atr(df, self.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr <= 0 or not np.isfinite(atr):
            return WyckoffResult()

        # Identify range
        rng_high = float(recent["high"].iloc[:-1].max())  # exclude last bar
        rng_low = float(recent["low"].iloc[:-1].min())
        rng_width = rng_high - rng_low
        in_consolidation = (rng_width / atr) <= self.consolidation_atr_threshold

        notes: list[str] = []
        last = recent.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        v = float(last["volume"])
        body = abs(c - o)
        avg_vol = float(recent["volume"].iloc[:-1].mean()) if len(recent) > 1 else v
        vol_ratio = v / avg_vol if avg_vol > 0 else 1.0

        # Spring: false breakdown below range low
        spring = (
            in_consolidation
            and l < rng_low - atr * self.breakout_atr_multiple
            and c > rng_low
            and (min(o, c) - l) / max(h - l, 1e-9) >= 0.4  # meaningful lower wick
        )

        # Upthrust: false breakout above range high
        upthrust = (
            in_consolidation
            and h > rng_high + atr * self.breakout_atr_multiple
            and c < rng_high
            and (h - max(o, c)) / max(h - l, 1e-9) >= 0.4  # meaningful upper wick
        )

        # SOS: strong up bar with volume spike after consolidation
        sos = (
            in_consolidation
            and c > o
            and body >= atr * self.body_atr_ratio
            and vol_ratio >= self.volume_spike_ratio
            and c > recent["close"].iloc[:-1].rolling(5).mean().iloc[-1]
            if len(recent) > 5 else False
        )

        # SOW: strong down bar with volume spike after consolidation
        sow = (
            in_consolidation
            and c < o
            and body >= atr * self.body_atr_ratio
            and vol_ratio >= self.volume_spike_ratio
            and c < recent["close"].iloc[:-1].rolling(5).mean().iloc[-1]
            if len(recent) > 5 else False
        )

        # Volume bias — compare avg up-day volume vs avg down-day volume
        up_days = recent[recent["close"] > recent["open"]]
        dn_days = recent[recent["close"] < recent["open"]]
        up_vol = float(up_days["volume"].mean()) if len(up_days) > 0 else 0
        dn_vol = float(dn_days["volume"].mean()) if len(dn_days) > 0 else 0
        if up_vol > dn_vol * 1.2:
            volume_bias = "bull"
        elif dn_vol > up_vol * 1.2:
            volume_bias = "bear"
        else:
            volume_bias = "neutral"

        # Accumulation = spring + bullish volume bias (and/or SOS)
        accumulation = bool(spring or (sos and volume_bias == "bull"))
        # Distribution = upthrust + bearish volume bias (and/or SOW)
        distribution = bool(upthrust or (sow and volume_bias == "bear"))

        # Confidence
        conf = 0.0
        if spring:
            conf = max(conf, 0.7 if volume_bias == "bull" else 0.5)
            notes.append("spring detected")
        if upthrust:
            conf = max(conf, 0.7 if volume_bias == "bear" else 0.5)
            notes.append("upthrust detected")
        if sos:
            conf = max(conf, 0.65)
            notes.append(f"SOS (vol {vol_ratio:.2f}x, body {body/atr:.2f}x ATR)")
        if sow:
            conf = max(conf, 0.65)
            notes.append(f"SOW (vol {vol_ratio:.2f}x, body {body/atr:.2f}x ATR)")
        if accumulation:
            notes.append("accumulation pattern")
        if distribution:
            notes.append("distribution pattern")
        if in_consolidation:
            notes.append(f"in consolidation range [{rng_low:.2f}, {rng_high:.2f}]")
        notes.append(f"volume_bias={volume_bias}")

        return WyckoffResult(
            spring_detected=bool(spring),
            upthrust_detected=bool(upthrust),
            sos_detected=bool(sos),
            sow_detected=bool(sow),
            accumulation_likely=bool(accumulation),
            distribution_likely=bool(distribution),
            in_consolidation=bool(in_consolidation),
            consolidation_range=(rng_low, rng_high) if in_consolidation else None,
            volume_bias=volume_bias,
            confidence=conf,
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


__all__ = ["WyckoffAnalyzer", "WyckoffResult"]
