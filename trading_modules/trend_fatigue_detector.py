"""trading_modules/trend_fatigue_detector.py
=====================================================================
Trend Fatigue Detector (Principle #184, #191)
=====================================================================
Detects when a trend is losing steam BEFORE it reverses. This gives the
bot time to tighten stops, take profits, or exit entirely.

Fatigue Signals (5 indicators):
    1. ATR DROP         — volatility contracting (trend running out of fuel)
    2. VOLUME DROP      — participation declining (no new buyers/sellers)
    3. FAILED BREAKOUT  — price broke a level but couldn't hold
    4. MOMENTUM LOSS    — RSI/MACD diverging from price
    5. STRUCTURE WEAKENING — smaller pullbacks, shallower thrusts

Fatigue Score (0-100):
    0-30   = Trend is strong, no fatigue
    30-50  = Early fatigue signs, tighten stops
    50-70  = Moderate fatigue, consider taking profits
    70-100 = Severe fatigue, exit or reverse

Usage:
    detector = TrendFatigueDetector()
    fatigue = detector.detect(df, trend_direction="up")
    # fatigue = {
    #     "score": 65,
    #     "level": "moderate",
    #     "signals": ["atr_drop", "volume_drop", "momentum_loss"],
    #     "recommendation": "Take profits, tighten stops",
    #     "reversal_probability": 0.45,
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.trend_fatigue_detector")


@dataclass
class FatigueResult:
    """Trend fatigue detection result."""
    score: float = 0.0               # 0-100 (100 = exhausted)
    level: str = "none"              # none/early/moderate/severe
    trend_direction: str = "up"      # up/down/none

    # Individual signals
    atr_drop: bool = False
    volume_drop: bool = False
    failed_breakout: bool = False
    momentum_loss: bool = False
    structure_weakening: bool = False

    # Details
    signals_detected: List[str] = field(default_factory=list)
    reversal_probability: float = 0.0  # 0-1
    recommendation: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "trend_direction": self.trend_direction,
            "signals": {
                "atr_drop": self.atr_drop,
                "volume_drop": self.volume_drop,
                "failed_breakout": self.failed_breakout,
                "momentum_loss": self.momentum_loss,
                "structure_weakening": self.structure_weakening,
            },
            "signals_detected": self.signals_detected,
            "reversal_probability": round(self.reversal_probability, 3),
            "recommendation": self.recommendation,
            "description": self.description,
        }


class TrendFatigueDetector:
    """Detects trend fatigue before reversal."""

    def __init__(self,
                 atr_lookback: int = 20,
                 volume_lookback: int = 20,
                 breakout_lookback: int = 20):
        """Initialize detector.

        Args:
            atr_lookback: bars for ATR comparison
            volume_lookback: bars for volume comparison
            breakout_lookback: bars for breakout detection
        """
        self.atr_lookback = atr_lookback
        self.volume_lookback = volume_lookback
        self.breakout_lookback = breakout_lookback

    def detect(self, df: pd.DataFrame,
               trend_direction: str = "up") -> FatigueResult:
        """Detect trend fatigue.

        Args:
            df: OHLCV DataFrame
            trend_direction: "up" or "down" (the trend we're monitoring)

        Returns:
            FatigueResult with score + signals + recommendation
        """
        result = FatigueResult(trend_direction=trend_direction)

        if df is None or df.empty or len(df) < 50:
            result.description = "insufficient data"
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # === 1. ATR drop ===
        result.atr_drop = self._detect_atr_drop(df)
        if result.atr_drop:
            result.signals_detected.append("atr_drop")

        # === 2. Volume drop ===
        result.volume_drop = self._detect_volume_drop(vol)
        if result.volume_drop:
            result.signals_detected.append("volume_drop")

        # === 3. Failed breakout ===
        result.failed_breakout = self._detect_failed_breakout(df, trend_direction)
        if result.failed_breakout:
            result.signals_detected.append("failed_breakout")

        # === 4. Momentum loss (divergence) ===
        result.momentum_loss = self._detect_momentum_loss(df, trend_direction)
        if result.momentum_loss:
            result.signals_detected.append("momentum_loss")

        # === 5. Structure weakening ===
        result.structure_weakening = self._detect_structure_weakening(df, trend_direction)
        if result.structure_weakening:
            result.signals_detected.append("structure_weakening")

        # === Compute fatigue score ===
        signal_count = len(result.signals_detected)
        result.score = min(100, signal_count * 22)  # each signal = ~22 points

        # === Level ===
        if result.score < 30:
            result.level = "none"
        elif result.score < 50:
            result.level = "early"
        elif result.score < 70:
            result.level = "moderate"
        else:
            result.level = "severe"

        # === Reversal probability ===
        result.reversal_probability = min(0.9, result.score / 120)

        # === Recommendation ===
        result.recommendation = self._recommend(result)
        result.description = self._describe(result)

        return result

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------
    def _detect_atr_drop(self, df: pd.DataFrame) -> bool:
        """ATR declining — trend losing fuel."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        recent_atr = float(atr.tail(5).mean())
        older_atr = float(atr.tail(self.atr_lookback).head(10).mean())
        if older_atr == 0:
            return False
        return recent_atr < older_atr * 0.7  # 30% drop

    def _detect_volume_drop(self, vol: pd.Series) -> bool:
        """Volume declining — participation waning."""
        recent_vol = float(vol.tail(5).mean())
        older_vol = float(vol.tail(self.volume_lookback).head(10).mean())
        if older_vol == 0:
            return False
        return recent_vol < older_vol * 0.6  # 40% drop

    def _detect_failed_breakout(self, df: pd.DataFrame,
                                 direction: str) -> bool:
        """Price broke a level but couldn't hold."""
        if len(df) < 20:
            return False
        high = df["high"]
        low = df["low"]
        close = df["close"]

        if direction == "up":
            # Check if recent high broke above previous high, but close fell back
            recent_high = float(high.tail(5).max())
            prev_high = float(high.tail(20).head(15).max())
            last_close = float(close.iloc[-1])
            return recent_high > prev_high * 1.005 and last_close < prev_high
        else:
            recent_low = float(low.tail(5).min())
            prev_low = float(low.tail(20).head(15).min())
            last_close = float(close.iloc[-1])
            return recent_low < prev_low * 0.995 and last_close > prev_low

    def _detect_momentum_loss(self, df: pd.DataFrame,
                               direction: str) -> bool:
        """RSI diverging from price — momentum loss."""
        close = df["close"]
        if len(close) < 30:
            return False
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Price making higher highs but RSI making lower highs = bearish divergence
        price_20_ago = float(close.iloc[-20])
        price_now = float(close.iloc[-1])
        rsi_20_ago = float(rsi.iloc[-20]) if not pd.isna(rsi.iloc[-20]) else 50
        rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        if direction == "up":
            # Price up but RSI down = bearish divergence
            return price_now > price_20_ago and rsi_now < rsi_20_ago - 5
        else:
            # Price down but RSI up = bullish divergence
            return price_now < price_20_ago and rsi_now > rsi_20_ago + 5

    def _detect_structure_weakening(self, df: pd.DataFrame,
                                     direction: str) -> bool:
        """Smaller thrusts, shallower pullbacks."""
        if len(df) < 40:
            return False
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Compare recent thrust size vs older thrust size
        recent_thrusts = []
        older_thrusts = []
        for i in range(len(df) - 10, len(df)):
            if i > 0:
                recent_thrusts.append(abs(close.iloc[i] - close.iloc[i - 1]))
        for i in range(len(df) - 30, len(df) - 20):
            if i > 0:
                older_thrusts.append(abs(close.iloc[i] - close.iloc[i - 1]))

        if not recent_thrusts or not older_thrusts:
            return False

        recent_avg = float(np.mean(recent_thrusts))
        older_avg = float(np.mean(older_thrusts))
        if older_avg == 0:
            return False
        return recent_avg < older_avg * 0.6  # thrusts 40% smaller

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def _recommend(self, r: FatigueResult) -> str:
        """Generate recommendation based on fatigue level."""
        if r.level == "none":
            return "Trend strong — hold positions, maintain stops"
        elif r.level == "early":
            return "Early fatigue — tighten stops, prepare to exit"
        elif r.level == "moderate":
            return "Moderate fatigue — take partial profits, tighten stops significantly"
        else:  # severe
            return "Severe fatigue — exit positions, prepare for reversal"

    def _describe(self, r: FatigueResult) -> str:
        """Human-readable description."""
        signals_str = ", ".join(r.signals_detected) if r.signals_detected else "none"
        return (
            f"Trend fatigue: {r.level} ({r.score:.0f}/100). "
            f"Signals: {signals_str}. "
            f"Reversal probability: {r.reversal_probability:.0%}. "
            f"Recommendation: {r.recommendation}"
        )
