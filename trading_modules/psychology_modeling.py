"""
Psychology Modeling — Fear / Greed / FOMO / Panic / Euphoria detection
======================================================================

Market prices are the aggregate expression of human (and bot) psychology.
This module infers the dominant emotional state of the market from price +
volume action:

    - Fear          — sharp drops on rising volume (capitulation)
    - Greed         — strong uptrend, low pullbacks, rising volume
    - FOMO          — parabolic move, volume spike, RSI > 80
    - Panic         — V-shaped bottom, extreme volume, large body
    - Euphoria      — consecutive green bars, low volume on red, RSI > 85
    - Capitulation  — long lower wicks on extreme volume after downtrend
    - Complacency   — low volatility, low volume, drifting

Each state gets a 0..1 intensity score. The dominant state is the one with
the highest intensity.

Usage:
    from trading_modules.psychology_modeling import PsychologyAnalyzer
    analyzer = PsychologyAnalyzer()
    result = analyzer.analyze(df_m15)
    print(f"Dominant: {result.dominant_state} (intensity {result.intensity:.2f})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PsychologyResult:
    dominant_state: str            # "fear" / "greed" / "fomo" / "panic" / "euphoria" / "capitulation" / "complacency" / "neutral"
    intensity: float               # 0..1 — strength of dominant emotion
    fear_greed_index: float        # 0..100 (0 = extreme fear, 100 = extreme greed)
    scores: dict[str, float] = field(default_factory=dict)  # all state scores
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dominant_state": self.dominant_state,
            "intensity": round(self.intensity, 3),
            "fear_greed_index": round(self.fear_greed_index, 1),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "notes": self.notes,
        }


class PsychologyAnalyzer:
    """Infer market psychology from price + volume action.

    Parameters:
        lookback: bars for analysis (default 50)
        atr_period: ATR lookback (default 14)
        rsi_period: RSI lookback (default 14)
        volume_period: volume avg lookback (default 20)
        rsi_euphoria: RSI threshold for euphoria (default 80)
        rsi_panic: RSI threshold for panic (default 20)
    """

    def __init__(
        self, lookback: int = 50, atr_period: int = 14,
        rsi_period: int = 14, volume_period: int = 20,
        rsi_euphoria: float = 80, rsi_panic: float = 20,
    ) -> None:
        self.lookback = lookback
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.rsi_euphoria = rsi_euphoria
        self.rsi_panic = rsi_panic

    def analyze(self, df: pd.DataFrame) -> PsychologyResult:
        if df is None or len(df) < max(self.lookback, self.atr_period + 5, self.rsi_period + 5):
            return PsychologyResult("neutral", 0.0, 50.0, notes=["insufficient data"])
        recent = df.tail(self.lookback).reset_index(drop=True)
        close = recent["close"]
        high = recent["high"]
        low = recent["low"]
        vol = recent["volume"]

        # ── Indicators ────────────────────────────────────────────
        # Returns
        rets = close.pct_change().fillna(0)
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(50)
        rsi_now = float(rsi.iloc[-1])
        # ATR
        atr_series = self._atr(df, self.atr_period)
        atr_now = float(atr_series.iloc[-1])
        atr_baseline = float(atr_series.rolling(50).mean().iloc[-1]) if len(atr_series) > 50 else atr_now
        atr_ratio = atr_now / atr_baseline if atr_baseline > 0 else 1.0
        # Volume
        vol_now = float(vol.iloc[-1])
        vol_avg = float(vol.iloc[:-1].tail(self.volume_period).mean()) if len(vol) > 1 else vol_now
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
        # Recent trend (last 10 bars)
        recent_close = close.tail(10)
        trend_pct = float((recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0])
        # Green/red ratio
        green_bars = int((close > recent["open"]).sum())
        red_bars = int((close < recent["open"]).sum())
        # Candle bodies and wicks
        last = recent.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = float(last["high"]) - float(last["low"])
        lower_wick = float(min(last["open"], last["close"])) - float(last["low"])
        upper_wick = float(last["high"]) - float(max(last["open"], last["close"]))
        body_ratio = body / rng if rng > 0 else 0
        lower_wick_ratio = lower_wick / rng if rng > 0 else 0
        upper_wick_ratio = upper_wick / rng if rng > 0 else 0

        # ── Score each emotional state (0..1) ─────────────────────
        scores: dict[str, float] = {}

        # Fear: sharp drops, rising volume
        scores["fear"] = float(
            0.5 * max(0, -trend_pct / 0.05) +  # 5% drop = full
            0.3 * max(0, vol_ratio - 1) / 2 +
            0.2 * max(0, (50 - rsi_now) / 30)
        )
        scores["fear"] = min(1.0, scores["fear"])

        # Greed: steady uptrend, low pullback, healthy volume
        scores["greed"] = float(
            0.5 * max(0, trend_pct / 0.05) +
            0.3 * max(0, (rsi_now - 50) / 30) +
            0.2 * (green_bars / max(1, green_bars + red_bars))
        )
        scores["greed"] = min(1.0, scores["greed"])

        # FOMO: parabolic move, volume spike, RSI > 80
        scores["fomo"] = float(
            0.4 * max(0, (rsi_now - self.rsi_euphoria) / 20) +
            0.3 * max(0, vol_ratio - 1.5) / 2 +
            0.3 * max(0, trend_pct / 0.08)
        )
        scores["fomo"] = min(1.0, scores["fomo"])

        # Panic: V-bottom, extreme volume, large body, low RSI
        scores["panic"] = float(
            0.4 * max(0, (self.rsi_panic - rsi_now) / 20) +
            0.3 * max(0, vol_ratio - 2) / 2 +
            0.3 * max(0, body_ratio - 0.6)
        )
        scores["panic"] = min(1.0, scores["panic"])

        # Euphoria: consecutive greens, RSI > 85, low volume on reds
        if rsi_now > 85:
            scores["euphoria"] = float(
                0.5 * (rsi_now - 85) / 15 +
                0.3 * (green_bars / max(1, green_bars + red_bars)) +
                0.2 * max(0, trend_pct / 0.06)
            )
        else:
            scores["euphoria"] = 0.0
        scores["euphoria"] = min(1.0, scores["euphoria"])

        # Capitulation: long lower wicks on extreme volume after downtrend
        if trend_pct < 0:
            scores["capitulation"] = float(
                0.4 * max(0, lower_wick_ratio - 0.4) +
                0.3 * max(0, vol_ratio - 2) / 2 +
                0.3 * max(0, -trend_pct / 0.05)
            )
        else:
            scores["capitulation"] = 0.0
        scores["capitulation"] = min(1.0, scores["capitulation"])

        # Complacency: low volatility, low volume, drifting
        scores["complacency"] = float(
            0.5 * max(0, 1 - atr_ratio) +     # ATR below baseline
            0.3 * max(0, 1 - vol_ratio) +
            0.2 * max(0, 1 - abs(trend_pct) / 0.02)
        )
        scores["complacency"] = min(1.0, scores["complacency"])

        # Neutral
        scores["neutral"] = 1.0 - max(scores.values())

        # ── Dominant state ────────────────────────────────────────
        dominant = max(scores, key=scores.get)
        intensity = float(scores[dominant])

        # ── Fear/Greed Index (0..100) ─────────────────────────────
        # Composite: 50 + weighted combination
        fg_index = 50.0
        fg_index += 25 * (scores["greed"] + scores["euphoria"] + scores["fomo"] * 0.5)
        fg_index -= 25 * (scores["fear"] + scores["panic"] + scores["capitulation"] * 0.5)
        fg_index += 10 * (rsi_now - 50) / 50
        fg_index += 5 * (trend_pct / 0.05)
        fg_index = max(0.0, min(100.0, fg_index))

        notes: list[str] = []
        notes.append(f"RSI={rsi_now:.1f} vol_ratio={vol_ratio:.2f} atr_ratio={atr_ratio:.2f}")
        notes.append(f"trend={trend_pct:+.2%} green/red={green_bars}/{red_bars}")
        notes.append(f"body_ratio={body_ratio:.2f} lower_wick={lower_wick_ratio:.2f} upper_wick={upper_wick_ratio:.2f}")
        notes.append(f"fear_greed_index={fg_index:.1f}")

        return PsychologyResult(
            dominant_state=dominant,
            intensity=intensity,
            fear_greed_index=float(fg_index),
            scores=scores,
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


__all__ = ["PsychologyAnalyzer", "PsychologyResult"]
