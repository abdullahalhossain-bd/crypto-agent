"""trading_modules/market_phase_detector.py
=====================================================================
Market Phase Detector (Principle #109 — Wyckoff Phases)
=====================================================================
Classifies the market into one of 4 Wyckoff phases:

    ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN → (repeat)

Detection Logic:
    ACCUMULATION:
        - Price in a trading range after a downtrend
        - Volume declining, selling pressure exhausted
        - Smart money quietly buying
        - SPR (Spring) pattern often marks end

    MARKUP:
        - Price breaking above accumulation range
        - Volume increasing on up bars
        - Higher highs, higher lows
        - Pullbacks are shallow

    DISTRIBUTION:
        - Price in a trading range after an uptrend
        - Volume high but price stalling
        - Smart money selling to retail
        - UTAD (Upthrust After Distribution) often marks end

    MARKDOWN:
        - Price breaking below distribution range
        - Volume increasing on down bars
        - Lower highs, lower lows
        - Bounces are shallow

Usage:
    detector = MarketPhaseDetector()
    phase = detector.detect(df)
    # phase = {"phase": "markup", "confidence": 0.82, "sub_phase": "BC",
    #          "range": {"support": 42000, "resistance": 45000}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.market_phase_detector")


class MarketPhase(str, Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class SubPhase(str, Enum):
    # Accumulation sub-phases (Wyckoff)
    PS = "preliminary_support"   # Initial selling pressure decrease
    SC = "selling_climax"        # High volume selling, smart money starts buying
    AR = "automatic_rally"       # Natural rebound after SC
    ST = "secondary_test"        # Test of SC lows with lower volume
    SPRING = "spring"            # Final shakeout below support (best entry)
    # Markup sub-phases
    LPS = "last_point_of_support"  # Pullback before markup
    # Distribution sub-phases
    PSY = "preliminary_supply"   # Initial buying pressure decrease
    BC = "buying_climax"         # High volume buying, smart money starts selling
    AR_D = "automatic_reaction"  # Natural pullback after BC
    ST_D = "secondary_test"      # Test of BC highs with lower volume
    UTAD = "upthrust"            # Final spike above resistance (trap)
    # Markdown sub-phases
    LPSY = "last_point_of_supply"  # Bounce before markdown


@dataclass
class PhaseResult:
    """Result of market phase detection."""
    phase: MarketPhase = MarketPhase.UNKNOWN
    sub_phase: Optional[SubPhase] = None
    confidence: float = 0.0       # 0-1
    range_support: float = 0.0    # detected range low
    range_resistance: float = 0.0 # detected range high
    phase_duration_bars: int = 0  # how long in this phase
    volume_trend: str = "stable"  # increasing/decreasing/stable
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "sub_phase": self.sub_phase.value if self.sub_phase else None,
            "confidence": round(self.confidence, 3),
            "range_support": self.range_support,
            "range_resistance": self.range_resistance,
            "phase_duration_bars": self.phase_duration_bars,
            "volume_trend": self.volume_trend,
            "description": self.description,
        }


class MarketPhaseDetector:
    """Detects Wyckoff market phases from OHLCV data.

    Uses a combination of:
        - Price range analysis (is price range-bound?)
        - Volume analysis (increasing/decreasing?)
        - Trend analysis (before range: uptrend or downtrend?)
        - Squeeze detection (volatility compression)
        - Spring/UTAD detection (false breakouts)
    """

    def __init__(self,
                 range_lookback: int = 50,
                 range_threshold_pct: float = 0.03,
                 min_phase_duration: int = 10):
        """Initialize detector.

        Args:
            range_lookback: bars to look back for range detection
            range_threshold_pct: max % deviation from midpoint to be "in range"
            min_phase_duration: minimum bars to confirm a phase
        """
        self.range_lookback = range_lookback
        self.range_threshold = range_threshold_pct
        self.min_duration = min_phase_duration

    def detect(self, df: pd.DataFrame) -> PhaseResult:
        """Detect the current market phase.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            PhaseResult with detected phase + metadata
        """
        if df is None or df.empty or len(df) < self.range_lookback:
            return PhaseResult()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # Get recent window
        window = df.tail(self.range_lookback)
        w_close = window["close"]
        w_high = window["high"]
        w_low = window["low"]
        w_vol = window["volume"]

        # === Step 1: Is price in a range? ===
        range_high = float(w_high.max())
        range_low = float(w_low.min())
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        range_pct = range_size / max(range_mid, 1e-10)

        # Current price position within range
        current_price = float(close.iloc[-1])
        price_position = (current_price - range_low) / max(range_size, 1e-10)

        # Is price range-bound? (position between 0.1 and 0.9)
        is_range_bound = 0.10 < price_position < 0.90 and range_pct < 0.15

        # === Step 2: What was the trend BEFORE the range? ===
        # Use longer lookback (2x range_lookback)
        pre_lookback = min(len(df), self.range_lookback * 2)
        pre_window = df.tail(pre_lookback).head(self.range_lookback)
        pre_close = pre_window["close"]

        if len(pre_close) > 10:
            pre_return = (pre_close.iloc[-1] - pre_close.iloc[0]) / max(pre_close.iloc[0], 1e-10)
            pre_trend = "up" if pre_return > 0.02 else "down" if pre_return < -0.02 else "flat"
        else:
            pre_trend = "flat"

        # === Step 3: Volume trend ===
        vol_recent = w_vol.tail(10).mean()
        vol_older = w_vol.head(20).mean()
        if vol_recent > vol_older * 1.2:
            volume_trend = "increasing"
        elif vol_recent < vol_older * 0.8:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        # === Step 4: Detect Spring / UTAD ===
        spring_detected = self._detect_spring(df)
        utad_detected = self._detect_utad(df)

        # === Step 5: Squeeze (volatility compression) ===
        recent_atr = self._compute_atr(df, 14).tail(10).mean()
        older_atr = self._compute_atr(df, 14).tail(30).head(20).mean()
        squeeze = recent_atr < older_atr * 0.7 if older_atr > 0 else False

        # === Phase Determination ===
        result = PhaseResult(range_support=range_low, range_resistance=range_high,
                            volume_trend=volume_trend)

        if is_range_bound:
            # In a range — is it accumulation or distribution?
            if pre_trend == "down":
                # After downtrend = accumulation
                result.phase = MarketPhase.ACCUMULATION
                result.confidence = self._accumulation_confidence(
                    volume_trend, squeeze, spring_detected, price_position)
                result.sub_phase = self._accumulation_subphase(
                    volume_trend, spring_detected, price_position)
                result.description = f"Accumulation after downtrend — smart money buying"
            elif pre_trend == "up":
                # After uptrend = distribution
                result.phase = MarketPhase.DISTRIBUTION
                result.confidence = self._distribution_confidence(
                    volume_trend, squeeze, utad_detected, price_position)
                result.sub_phase = self._distribution_subphase(
                    volume_trend, utad_detected, price_position)
                result.description = f"Distribution after uptrend — smart money selling"
            else:
                # After flat = consolidation (treat as accumulation-ish)
                result.phase = MarketPhase.ACCUMULATION
                result.confidence = 0.4
                result.description = "Range-bound, unclear prior trend"
        else:
            # Not range-bound — in trending phase
            # Check recent slope
            recent_return = (close.iloc[-1] - close.iloc[-20]) / max(close.iloc[-20], 1e-10)
            if recent_return > 0.02:
                result.phase = MarketPhase.MARKUP
                result.confidence = min(1.0, abs(recent_return) * 20)
                result.description = "Markup phase — price trending up"
            elif recent_return < -0.02:
                result.phase = MarketPhase.MARKDOWN
                result.confidence = min(1.0, abs(recent_return) * 20)
                result.description = "Markdown phase — price trending down"
            else:
                result.phase = MarketPhase.TRANSITION
                result.confidence = 0.3
                result.description = "Transition — phase unclear"

        result.phase_duration_bars = self._estimate_phase_duration(df, result.phase)
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def _detect_spring(self, df: pd.DataFrame) -> bool:
        """Detect a Spring pattern (false breakdown below support).

        Spring = price briefly dips below recent range low, then snaps back.
        """
        if len(df) < 20:
            return False
        recent_low = df["low"].tail(50).head(49).min()
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        # Spring: low broke support but close recovered above
        return last_low < recent_low * 0.998 and last_close > recent_low

    def _detect_utad(self, df: pd.DataFrame) -> bool:
        """Detect an Upthrust After Distribution (false breakout above resistance)."""
        if len(df) < 20:
            return False
        recent_high = df["high"].tail(50).head(49).max()
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        # UTAD: high broke resistance but close fell back below
        return last_high > recent_high * 1.002 and last_close < recent_high

    def _accumulation_confidence(self, vol_trend: str, squeeze: bool,
                                 spring: bool, price_pos: float) -> float:
        """Confidence score for accumulation phase."""
        score = 0.3  # base
        if vol_trend == "decreasing":
            score += 0.2  # selling pressure exhausting
        if squeeze:
            score += 0.2  # compression before breakout
        if spring:
            score += 0.25  # spring is strong accumulation signal
        if 0.3 < price_pos < 0.7:
            score += 0.05  # mid-range = safe
        return min(1.0, score)

    def _distribution_confidence(self, vol_trend: str, squeeze: bool,
                                  utad: bool, price_pos: float) -> float:
        """Confidence score for distribution phase."""
        score = 0.3
        if vol_trend == "increasing":
            score += 0.2  # buying climax
        if squeeze:
            score += 0.15
        if utad:
            score += 0.25  # UTAD is strong distribution signal
        if 0.3 < price_pos < 0.7:
            score += 0.05
        return min(1.0, score)

    def _accumulation_subphase(self, vol_trend: str, spring: bool,
                                price_pos: float) -> Optional[SubPhase]:
        """Determine which sub-phase of accumulation we're in."""
        if spring:
            return SubPhase.SPRING
        if vol_trend == "decreasing" and price_pos < 0.3:
            return SubPhase.ST  # secondary test
        if vol_trend == "increasing" and price_pos > 0.6:
            return SubPhase.LPS  # last point of support → markup coming
        return SubPhase.AR

    def _distribution_subphase(self, vol_trend: str, utad: bool,
                                price_pos: float) -> Optional[SubPhase]:
        """Determine which sub-phase of distribution we're in."""
        if utad:
            return SubPhase.UTAD
        if vol_trend == "increasing" and price_pos > 0.7:
            return SubPhase.BC  # buying climax
        if vol_trend == "decreasing" and price_pos < 0.4:
            return SubPhase.LPSY  # last point of supply → markdown coming
        return SubPhase.AR_D

    def _estimate_phase_duration(self, df: pd.DataFrame,
                                  phase: MarketPhase) -> int:
        """Estimate how many bars we've been in this phase."""
        # Simple: count bars since last significant trend change
        if len(df) < 20:
            return 0
        close = df["close"]
        changes = 0
        for i in range(len(close) - 1, max(0, len(close) - 100), -1):
            if abs(close.iloc[i] - close.iloc[i - 1]) / max(close.iloc[i - 1], 1e-10) > 0.02:
                changes += 1
                if changes >= 3:
                    return len(close) - i
        return min(len(df), 50)


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------
def detect_market_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """One-shot phase detection. Returns dict."""
    detector = MarketPhaseDetector()
    result = detector.detect(df)
    return result.to_dict()
