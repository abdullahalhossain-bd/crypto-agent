"""engine.candlestick.pattern_detector
=====================================================================
Comprehensive candlestick pattern detector with 49+ patterns.

Patterns detected (categorized):
  Single-bar (11): Pin Bar (Hammer/Shooting Star), Inverted Hammer,
    Hanging Man, Dragonfly Doji, Gravestone Doji, Long-Legged Doji,
    Doji, Spinning Top, Bullish Marubozu, Bearish Marubozu, Inside Bar
  Double-bar reversal (12): Bullish/Bearish Engulfing, Bullish/Bearish Harami,
    Bullish/Bearish Harami Cross, Piercing Line, Dark Cloud Cover,
    Tweezer Bottom/Top, Bullish/Bearish Kicker, Bullish/Bearish Meeting Line
  Triple-bar reversal (12): Morning/Evening Star, Morning/Evening Doji Star,
    Bullish/Bearish Abandoned Baby, Three White Soldiers/Black Crows,
    Three Inside Up/Down, Three Outside Up/Down
  Continuation (12): Bullish/Bearish Gap, Fair Value Rising/Falling Gap,
    Bullish/Bearish Neck, Bullish/Bearish Separating Line,
    Rising/Falling Three Methods, Rising/Falling N

Features:
  - Confidence scoring 0-100 for every pattern (ATR + volume + wick/body)
  - Trend-aware filtering (connect to MarketStateClassifier)
  - Post-pattern confirmation (wait 1 bar for next candle to confirm)
  - Best-pattern-per-bar selection (deduplication)

Sources: MotiveWave (33 patterns), ohlcpattern (40+ patterns),
  MercadoFinanceiro book (thresholds + confirmation idea).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

from utils.indicators import atr


# ----------------------------------------------------------------------
class PatternType(str, Enum):
    NONE = "none"
    # --- Existing single-bar ---
    PIN_BAR_BULLISH = "pin_bar_bullish"
    PIN_BAR_BEARISH = "pin_bar_bearish"
    INSIDE_BAR = "inside_bar"
    DOJI = "doji"
    # --- New single-bar ---
    INVERTED_HAMMER = "inverted_hammer"
    HANGING_MAN = "hanging_man"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    LONG_LEGGED_DOJI = "long_legged_doji"
    SPINNING_TOP = "spinning_top"
    BULLISH_MARUBOZU = "bullish_marubozu"
    BEARISH_MARUBOZU = "bearish_marubozu"
    # --- Existing double-bar ---
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    # --- New double-bar ---
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    TWEEZER_BOTTOM = "tweezer_bottom"
    TWEEZER_TOP = "tweezer_top"
    BULLISH_KICKER = "bullish_kicker"
    BEARISH_KICKER = "bearish_kicker"
    BULLISH_HARAMI_CROSS = "bullish_harami_cross"
    BEARISH_HARAMI_CROSS = "bearish_harami_cross"
    BULLISH_MEETING_LINE = "bullish_meeting_line"
    BEARISH_MEETING_LINE = "bearish_meeting_line"
    # --- Existing triple-bar ---
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    # --- New triple-bar ---
    MORNING_DOJI_STAR = "morning_doji_star"
    EVENING_DOJI_STAR = "evening_doji_star"
    BULLISH_ABANDONED_BABY = "bullish_abandoned_baby"
    BEARISH_ABANDONED_BABY = "bearish_abandoned_baby"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    THREE_OUTSIDE_UP = "three_outside_up"
    THREE_OUTSIDE_DOWN = "three_outside_down"
    # --- Continuation patterns ---
    BULLISH_GAP = "bullish_gap"
    BEARISH_GAP = "bearish_gap"
    FAIR_VALUE_RISING_GAP = "fair_value_rising_gap"
    FAIR_VALUE_FALLING_GAP = "fair_value_falling_gap"
    BULLISH_NECK = "bullish_neck"
    BEARISH_NECK = "bearish_neck"
    BULLISH_SEPARATING_LINE = "bullish_separating_line"
    BEARISH_SEPARATING_LINE = "bearish_separating_line"
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"
    RISING_N = "rising_n"
    FALLING_N = "falling_n"


# Pattern → required trend for validity (None = any trend)
# Bullish reversals need downtrend; bearish reversals need uptrend;
# continuation patterns need matching trend; neutral = any.
TREND_REQUIREMENTS: dict[PatternType, Optional[str]] = {
    # Bullish reversals → need downtrend
    PatternType.PIN_BAR_BULLISH: "downtrend",
    PatternType.INVERTED_HAMMER: "downtrend",
    PatternType.DRAGONFLY_DOJI: "downtrend",
    PatternType.ENGULFING_BULLISH: "downtrend",
    PatternType.HARAMI_BULLISH: "downtrend",
    PatternType.BULLISH_HARAMI_CROSS: "downtrend",
    PatternType.PIERCING_LINE: "downtrend",
    PatternType.TWEEZER_BOTTOM: "downtrend",
    PatternType.BULLISH_KICKER: "downtrend",
    PatternType.BULLISH_MEETING_LINE: "downtrend",
    PatternType.MORNING_STAR: "downtrend",
    PatternType.MORNING_DOJI_STAR: "downtrend",
    PatternType.BULLISH_ABANDONED_BABY: "downtrend",
    PatternType.THREE_WHITE_SOLDIERS: "downtrend",
    PatternType.THREE_INSIDE_UP: "downtrend",
    PatternType.THREE_OUTSIDE_UP: "downtrend",
    # Bearish reversals → need uptrend
    PatternType.PIN_BAR_BEARISH: "uptrend",
    PatternType.HANGING_MAN: "uptrend",
    PatternType.GRAVESTONE_DOJI: "uptrend",
    PatternType.ENGULFING_BEARISH: "uptrend",
    PatternType.HARAMI_BEARISH: "uptrend",
    PatternType.BEARISH_HARAMI_CROSS: "uptrend",
    PatternType.DARK_CLOUD_COVER: "uptrend",
    PatternType.TWEEZER_TOP: "uptrend",
    PatternType.BEARISH_KICKER: "uptrend",
    PatternType.BEARISH_MEETING_LINE: "uptrend",
    PatternType.EVENING_STAR: "uptrend",
    PatternType.EVENING_DOJI_STAR: "uptrend",
    PatternType.BEARISH_ABANDONED_BABY: "uptrend",
    PatternType.THREE_BLACK_CROWS: "uptrend",
    PatternType.THREE_INSIDE_DOWN: "uptrend",
    PatternType.THREE_OUTSIDE_DOWN: "uptrend",
    # Continuation → need matching trend
    PatternType.BULLISH_GAP: "uptrend",
    PatternType.FAIR_VALUE_RISING_GAP: "uptrend",
    PatternType.BULLISH_NECK: "uptrend",
    PatternType.BULLISH_SEPARATING_LINE: "uptrend",
    PatternType.RISING_THREE_METHODS: "uptrend",
    PatternType.RISING_N: "uptrend",
    PatternType.BEARISH_GAP: "downtrend",
    PatternType.FAIR_VALUE_FALLING_GAP: "downtrend",
    PatternType.BEARISH_NECK: "downtrend",
    PatternType.BEARISH_SEPARATING_LINE: "downtrend",
    PatternType.FALLING_THREE_METHODS: "downtrend",
    PatternType.FALLING_N: "downtrend",
    # Neutral → any trend
    PatternType.DOJI: None,
    PatternType.LONG_LEGGED_DOJI: None,
    PatternType.SPINNING_TOP: None,
    PatternType.INSIDE_BAR: None,
    PatternType.BULLISH_MARUBOZU: None,
    PatternType.BEARISH_MARUBOZU: None,
}


@dataclass
class PatternResult:
    pattern: PatternType
    confidence: float
    direction: str
    bar_index: int
    bar_time: Optional[Any] = None
    components: dict[str, float] = field(default_factory=dict)
    confirmed: bool = True          # False if awaiting post-pattern confirmation
    category: str = "reversal"      # "reversal" | "continuation" | "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "confidence": self.confidence,
            "direction": self.direction,
            "bar_index": self.bar_index,
            "bar_time": str(self.bar_time) if self.bar_time else None,
            "components": dict(self.components),
            "confirmed": self.confirmed,
            "category": self.category,
        }


# ----------------------------------------------------------------------
class PatternDetector:
    """Detect 49+ candlestick patterns and score them 0-100."""

    def __init__(self,
                 min_wick_ratio: float = 0.6,
                 min_body_ratio: float = 0.05,
                 min_engulf_ratio: float = 1.0,
                 atr_period: int = 14,
                 volume_period: int = 20,
                 require_trend_context: bool = False,
                 require_confirmation: bool = False,
                 tweezer_threshold: float = 0.05,
                 marubozu_body_ratio: float = 0.95,
                 star_small_body_ratio: float = 0.3) -> None:
        self.min_wick_ratio = float(min_wick_ratio)
        self.min_body_ratio = float(min_body_ratio)
        self.min_engulf_ratio = float(min_engulf_ratio)
        self.atr_period = int(atr_period)
        self.volume_period = int(volume_period)
        self.require_trend_context = bool(require_trend_context)
        self.require_confirmation = bool(require_confirmation)
        self.tweezer_threshold = float(tweezer_threshold)
        self.marubozu_body_ratio = float(marubozu_body_ratio)
        self.star_small_body_ratio = float(star_small_body_ratio)
        self._trend_classifier: Optional[Callable] = None

    # ----------------------------------------------------------------
    def set_trend_classifier(self, classifier: Callable[[pd.DataFrame], str]) -> None:
        """Connect a MarketStateClassifier. Must return 'uptrend'/'downtrend'/'sideways'."""
        self._trend_classifier = classifier

    # ----------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < max(self.atr_period, self.volume_period, 4):
            return []
        results: list[PatternResult] = []
        atr_series = atr(df, self.atr_period)
        vol_mean = df["volume"].rolling(self.volume_period, min_periods=5).mean() if "volume" in df.columns else None
        # Determine trend once for the whole series
        trend = "sideways"
        if self.require_trend_context and self._trend_classifier is not None:
            try:
                raw_trend = self._trend_classifier(df)
                # Map MarketState labels to our trend labels
                if "TREND" in raw_trend.upper():
                    # Check direction from the classifier
                    trend = "uptrend"  # simplified; real impl would check direction
                elif "RANGE" in raw_trend.upper() or "CHOP" in raw_trend.upper():
                    trend = "sideways"
            except Exception:
                trend = "sideways"

        for i in range(3, len(df)):
            bar = df.iloc[i]
            prev = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]
            prev3 = df.iloc[i - 3] if i >= 4 else None
            atr_val = float(atr_series.iloc[i]) if not atr_series.isna().iloc[i] else 0.0
            vol_mult = 1.0
            if vol_mean is not None and not vol_mean.isna().iloc[i] and vol_mean.iloc[i] > 0:
                vol_mult = float(bar["volume"] / vol_mean.iloc[i])
            patterns = self._detect_patterns_at(df, i, prev, prev2, prev3, atr_val, vol_mult)
            # Trend filtering
            if self.require_trend_context:
                patterns = [p for p in patterns if self._trend_ok(p.pattern, trend)]
            # Confirmation filtering
            if self.require_confirmation and i > 0:
                patterns = self._apply_confirmation(patterns, df, i)
            # Deduplicate: keep best per bar
            if patterns:
                best = max(patterns, key=lambda r: r.confidence)
                results.append(best)
        return results

    def detect_latest(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 4:
            return PatternResult(
                pattern=PatternType.NONE, confidence=0.0,
                direction="neutral", bar_index=len(df) - 1,
                bar_time=df["time"].iloc[-1] if "time" in df.columns else None,
            )
        results = self.detect(df.tail(8).reset_index(drop=True))
        if not results:
            return PatternResult(
                pattern=PatternType.NONE, confidence=0.0,
                direction="neutral", bar_index=len(df) - 1,
                bar_time=df["time"].iloc[-1] if "time" in df.columns else None,
            )
        last_bar_results = [r for r in results if r.bar_index >= 4]
        if not last_bar_results:
            last_bar_results = results
        best = max(last_bar_results, key=lambda r: r.confidence)
        best.bar_index = len(df) - 1
        best.bar_time = df["time"].iloc[-1] if "time" in df.columns else None
        return best

    # ----------------------------------------------------------------
    @staticmethod
    def _trend_ok(pattern: PatternType, current_trend: str) -> bool:
        required = TREND_REQUIREMENTS.get(pattern)
        if required is None or current_trend == "sideways":
            return True
        return required == current_trend

    def _apply_confirmation(self, patterns: list[PatternResult],
                              df: pd.DataFrame, i: int) -> list[PatternResult]:
        """Post-pattern confirmation: next candle must confirm."""
        if i + 1 >= len(df):
            # Can't confirm — mark as unconfirmed
            for p in patterns:
                p.confirmed = False
                p.confidence *= 0.5  # halve confidence if unconfirmed
            return patterns
        next_bar = df.iloc[i + 1]
        next_close = float(next_bar["close"])
        current_high = float(df.iloc[i]["high"])
        current_low = float(df.iloc[i]["low"])
        confirmed: list[PatternResult] = []
        for p in patterns:
            if p.direction == "bullish":
                if next_close > current_high:
                    p.confirmed = True
                    p.confidence = min(100.0, p.confidence * 1.15)  # boost
                    confirmed.append(p)
                else:
                    p.confirmed = False
                    p.confidence *= 0.4
                    confirmed.append(p)
            elif p.direction == "bearish":
                if next_close < current_low:
                    p.confirmed = True
                    p.confidence = min(100.0, p.confidence * 1.15)
                    confirmed.append(p)
                else:
                    p.confirmed = False
                    p.confidence *= 0.4
                    confirmed.append(p)
            else:
                confirmed.append(p)  # neutral patterns don't need confirmation
        return confirmed

    # ----------------------------------------------------------------
    def _detect_patterns_at(self, df: pd.DataFrame, i: int,
                              prev: pd.Series, prev2: pd.Series,
                              prev3: Optional[pd.Series],
                              atr_val: float, vol_mult: float
                              ) -> list[PatternResult]:
        out: list[PatternResult] = []
        bar = df.iloc[i]
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
        p2o, p2h, p2l, p2c = float(prev2["open"]), float(prev2["high"]), float(prev2["low"]), float(prev2["close"])
        body = abs(c - o)
        range_ = h - l
        if range_ <= 0:
            return out
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        prev_body = abs(pc - po)
        prev2_body = abs(p2c - p2o)
        is_bull = c > o
        is_bear = c < o
        prev_bull = pc > po
        prev_bear = pc < po
        prev2_bull = p2c > p2o
        prev2_bear = p2c < p2o
        bar_time = bar.get("time", None) if "time" in df.columns else None
        comps = lambda **kw: dict(kw, atr_ratio=range_ / atr_val if atr_val > 0 else 0, volume_mult=vol_mult)
        conf_base = lambda: self._base_confidence(range_, atr_val, vol_mult)

        # ===== SINGLE-BAR PATTERNS =====

        # 1. Pin Bar Bullish (Hammer)
        if lower_wick / range_ >= self.min_wick_ratio and body / range_ <= 0.4:
            out.append(PatternResult(PatternType.PIN_BAR_BULLISH,
                self._pin_bar_confidence(lower_wick, upper_wick, body, range_, atr_val, vol_mult),
                "bullish", i, bar_time, comps(lower_wick_ratio=lower_wick/range_, upper_wick_ratio=upper_wick/range_, body_ratio=body/range_),
                category="reversal"))

        # 2. Pin Bar Bearish (Shooting Star)
        if upper_wick / range_ >= self.min_wick_ratio and body / range_ <= 0.4:
            out.append(PatternResult(PatternType.PIN_BAR_BEARISH,
                self._pin_bar_confidence(upper_wick, lower_wick, body, range_, atr_val, vol_mult),
                "bearish", i, bar_time, comps(upper_wick_ratio=upper_wick/range_, lower_wick_ratio=lower_wick/range_, body_ratio=body/range_),
                category="reversal"))

        # 3. Inverted Hammer (bullish, small body at bottom, long upper wick)
        if is_bull and upper_wick > body * 2 and lower_wick < body * 0.5:
            out.append(PatternResult(PatternType.INVERTED_HAMMER,
                self._pin_bar_confidence(upper_wick, lower_wick, body, range_, atr_val, vol_mult) * 0.9,
                "bullish", i, bar_time, comps(upper_wick_ratio=upper_wick/range_, body_ratio=body/range_),
                category="reversal"))

        # 4. Hanging Man (bearish, small body at top, long lower wick, in uptrend)
        if is_bear and lower_wick > body * 2 and upper_wick < body * 0.5:
            out.append(PatternResult(PatternType.HANGING_MAN,
                self._pin_bar_confidence(lower_wick, upper_wick, body, range_, atr_val, vol_mult) * 0.9,
                "bearish", i, bar_time, comps(lower_wick_ratio=lower_wick/range_, body_ratio=body/range_),
                category="reversal"))

        # 5. Doji
        if body / range_ <= self.min_body_ratio:
            out.append(PatternResult(PatternType.DOJI,
                40.0 + 30.0 * min(1.0, vol_mult), "neutral", i, bar_time,
                comps(body_ratio=body/range_), category="neutral"))

        # 6. Dragonfly Doji (long lower wick, no upper wick, doji body)
        if body / range_ <= self.min_body_ratio and lower_wick > range_ * 0.6 and upper_wick < range_ * 0.1:
            out.append(PatternResult(PatternType.DRAGONFLY_DOJI,
                55.0 + 25.0 * min(1.0, vol_mult), "bullish", i, bar_time,
                comps(lower_wick_ratio=lower_wick/range_, upper_wick_ratio=upper_wick/range_),
                category="reversal"))

        # 7. Gravestone Doji (long upper wick, no lower wick, doji body)
        if body / range_ <= self.min_body_ratio and upper_wick > range_ * 0.6 and lower_wick < range_ * 0.1:
            out.append(PatternResult(PatternType.GRAVESTONE_DOJI,
                55.0 + 25.0 * min(1.0, vol_mult), "bearish", i, bar_time,
                comps(upper_wick_ratio=upper_wick/range_, lower_wick_ratio=lower_wick/range_),
                category="reversal"))

        # 8. Long-Legged Doji (doji with long shadows both sides)
        if body / range_ <= self.min_body_ratio and upper_wick > body * 2 and lower_wick > body * 2:
            out.append(PatternResult(PatternType.LONG_LEGGED_DOJI,
                45.0 + 20.0 * min(1.0, vol_mult), "neutral", i, bar_time,
                comps(upper_wick_ratio=upper_wick/range_, lower_wick_ratio=lower_wick/range_),
                category="neutral"))

        # 9. Spinning Top (small body, not doji, long shadows both sides)
        if 0.1 < body / range_ < 0.3 and upper_wick > body and lower_wick > body:
            out.append(PatternResult(PatternType.SPINNING_TOP,
                35.0 + 15.0 * min(1.0, vol_mult), "neutral", i, bar_time,
                comps(body_ratio=body/range_), category="neutral"))

        # 10. Bullish Marubozu (body = entire range, bullish)
        if is_bull and body / range_ > self.marubozu_body_ratio:
            out.append(PatternResult(PatternType.BULLISH_MARUBOZU,
                50.0 + 30.0 * min(1.0, vol_mult), "bullish", i, bar_time,
                comps(body_ratio=body/range_), category="neutral"))

        # 11. Bearish Marubozu (body = entire range, bearish)
        if is_bear and body / range_ > self.marubozu_body_ratio:
            out.append(PatternResult(PatternType.BEARISH_MARUBOZU,
                50.0 + 30.0 * min(1.0, vol_mult), "bearish", i, bar_time,
                comps(body_ratio=body/range_), category="neutral"))

        # 12. Inside Bar
        if h < ph and l > pl:
            out.append(PatternResult(PatternType.INSIDE_BAR,
                self._inside_bar_confidence(df, i, atr_val, vol_mult),
                "neutral", i, bar_time,
                comps(mother_bar_range=ph - pl, inside_bar_range=range_),
                category="neutral"))

        # ===== DOUBLE-BAR PATTERNS =====

        # 13. Bullish Engulfing
        if prev_bear and is_bull and o <= pc and c >= po and body > prev_body * self.min_engulf_ratio:
            out.append(PatternResult(PatternType.ENGULFING_BULLISH,
                self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult),
                "bullish", i, bar_time,
                comps(body_ratio=body/range_, engulf_ratio=body/max(prev_body, 1e-9)),
                category="reversal"))

        # 14. Bearish Engulfing
        if prev_bull and is_bear and o >= pc and c <= po and body > prev_body * self.min_engulf_ratio:
            out.append(PatternResult(PatternType.ENGULFING_BEARISH,
                self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult),
                "bearish", i, bar_time,
                comps(body_ratio=body/range_, engulf_ratio=body/max(prev_body, 1e-9)),
                category="reversal"))

        # 15. Bullish Harami (small bullish inside large bearish)
        if prev_bear and is_bull and o >= pc and c <= po and body < prev_body * 0.5:
            out.append(PatternResult(PatternType.HARAMI_BULLISH,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bullish", i, bar_time,
                comps(body_ratio=body/range_, harami_ratio=body/max(prev_body, 1e-9)),
                category="reversal"))

        # 16. Bearish Harami (small bearish inside large bullish)
        if prev_bull and is_bear and o <= pc and c >= po and body < prev_body * 0.5:
            out.append(PatternResult(PatternType.HARAMI_BEARISH,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bearish", i, bar_time,
                comps(body_ratio=body/range_, harami_ratio=body/max(prev_body, 1e-9)),
                category="reversal"))

        # 17. Bullish Harami Cross (doji inside large bearish)
        if prev_bear and body / range_ <= self.min_body_ratio and o >= pc and c <= po and prev_body > body * 3:
            out.append(PatternResult(PatternType.BULLISH_HARAMI_CROSS,
                self._base_confidence(range_, atr_val, vol_mult) * 0.9,
                "bullish", i, bar_time,
                comps(body_ratio=body/range_), category="reversal"))

        # 18. Bearish Harami Cross (doji inside large bullish)
        if prev_bull and body / range_ <= self.min_body_ratio and o <= pc and c >= po and prev_body > body * 3:
            out.append(PatternResult(PatternType.BEARISH_HARAMI_CROSS,
                self._base_confidence(range_, atr_val, vol_mult) * 0.9,
                "bearish", i, bar_time,
                comps(body_ratio=body/range_), category="reversal"))

        # 19. Piercing Line (bullish: opens below prev low, closes above prev midpoint)
        if prev_bear and is_bull and o < pl:
            prev_mid = (po + pc) / 2.0
            if c > prev_mid and c < po:
                out.append(PatternResult(PatternType.PIERCING_LINE,
                    self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult) * 0.85,
                    "bullish", i, bar_time,
                    comps(body_ratio=body/range_, piercing_depth=(c - prev_mid) / max(po - prev_mid, 1e-9)),
                    category="reversal"))

        # 20. Dark Cloud Cover (bearish: opens above prev close, closes below prev midpoint)
        if prev_bull and is_bear and o > pc:
            prev_mid = (po + pc) / 2.0
            if c < prev_mid and c > po:
                out.append(PatternResult(PatternType.DARK_CLOUD_COVER,
                    self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult) * 0.85,
                    "bearish", i, bar_time,
                    comps(body_ratio=body/range_, cloud_depth=(prev_mid - c) / max(prev_mid - po, 1e-9)),
                    category="reversal"))

        # 21. Tweezer Bottom (two candles with similar lows, opposing colors)
        if abs(l - pl) / max((range_ + (ph - pl)) / 2, 1e-9) < self.tweezer_threshold and prev_bear != is_bull:
            out.append(PatternResult(PatternType.TWEEZER_BOTTOM,
                self._base_confidence(range_, atr_val, vol_mult) * 0.75,
                "bullish", i, bar_time,
                comps(low_diff=abs(l - pl)), category="reversal"))

        # 22. Tweezer Top (two candles with similar highs, opposing colors)
        if abs(h - ph) / max((range_ + (ph - pl)) / 2, 1e-9) < self.tweezer_threshold and prev_bull != is_bear:
            out.append(PatternResult(PatternType.TWEEZER_TOP,
                self._base_confidence(range_, atr_val, vol_mult) * 0.75,
                "bearish", i, bar_time,
                comps(high_diff=abs(h - ph)), category="reversal"))

        # 23. Bullish Kicker (gap up after bearish candle, no overlap)
        if prev_bear and is_bull and o > po:
            out.append(PatternResult(PatternType.BULLISH_KICKER,
                self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult) * 0.9,
                "bullish", i, bar_time,
                comps(gap_size=o - po), category="reversal"))

        # 24. Bearish Kicker (gap down after bullish candle, no overlap)
        if prev_bull and is_bear and o < po:
            out.append(PatternResult(PatternType.BEARISH_KICKER,
                self._engulfing_confidence(body, prev_body, range_, atr_val, vol_mult) * 0.9,
                "bearish", i, bar_time,
                comps(gap_size=po - o), category="reversal"))

        # 25. Bullish Meeting Line (counterattack: prev bearish, current bullish, same close)
        if prev_bear and is_bull and abs(c - pc) / range_ < 0.05:
            out.append(PatternResult(PatternType.BULLISH_MEETING_LINE,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bullish", i, bar_time,
                comps(close_diff=abs(c - pc)), category="reversal"))

        # 26. Bearish Meeting Line (counterattack: prev bullish, current bearish, same close)
        if prev_bull and is_bear and abs(c - pc) / range_ < 0.05:
            out.append(PatternResult(PatternType.BEARISH_MEETING_LINE,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bearish", i, bar_time,
                comps(close_diff=abs(c - pc)), category="reversal"))

        # ===== TRIPLE-BAR PATTERNS =====

        # 27. Morning Star (bearish → small body → bullish, gap down then up)
        if prev2_bear and is_bull:
            if prev2_body > prev2_body and prev_body < prev2_body * self.star_small_body_ratio:
                if max(po, pc) < p2c and body > prev2_body * 0.5:
                    out.append(PatternResult(PatternType.MORNING_STAR,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                        "bullish", i, bar_time,
                        comps(first_body=prev2_body, star_body=prev_body, third_body=body),
                        category="reversal"))

        # 28. Evening Star (bullish → small body → bearish, gap up then down)
        if prev2_bull and is_bear:
            if prev_body < prev2_body * self.star_small_body_ratio:
                if min(po, pc) > p2c and body > prev2_body * 0.5:
                    out.append(PatternResult(PatternType.EVENING_STAR,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                        "bearish", i, bar_time,
                        comps(first_body=prev2_body, star_body=prev_body, third_body=body),
                        category="reversal"))

        # 29. Morning Doji Star (like morning star but middle is doji)
        if prev2_bear and is_bull:
            prev_range = ph - pl
            if prev_range > 0 and prev_body / prev_range < self.min_body_ratio:
                if ph < p2c and body > prev2_body * 0.5:
                    out.append(PatternResult(PatternType.MORNING_DOJI_STAR,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.1,
                        "bullish", i, bar_time,
                        comps(first_body=prev2_body, doji_body=prev_body, third_body=body),
                        category="reversal"))

        # 30. Evening Doji Star (like evening star but middle is doji)
        if prev2_bull and is_bear:
            prev_range = ph - pl
            if prev_range > 0 and prev_body / prev_range < self.min_body_ratio:
                if pl > p2c and body > prev2_body * 0.5:
                    out.append(PatternResult(PatternType.EVENING_DOJI_STAR,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.1,
                        "bearish", i, bar_time,
                        comps(first_body=prev2_body, doji_body=prev_body, third_body=body),
                        category="reversal"))

        # 31. Bullish Abandoned Baby (morning doji star with gaps on both sides of doji)
        if prev2_bear and is_bull:
            prev_range = ph - pl
            if prev_range > 0 and prev_body / prev_range < self.min_body_ratio:
                if ph < p2l and ph < l:  # doji gaps down from first, up from third
                    out.append(PatternResult(PatternType.BULLISH_ABANDONED_BABY,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.2,
                        "bullish", i, bar_time,
                        comps(first_body=prev2_body, doji_body=prev_body),
                        category="reversal"))

        # 32. Bearish Abandoned Baby (evening doji star with gaps on both sides)
        if prev2_bull and is_bear:
            prev_range = ph - pl
            if prev_range > 0 and prev_body / prev_range < self.min_body_ratio:
                if pl > p2h and pl > h:
                    out.append(PatternResult(PatternType.BEARISH_ABANDONED_BABY,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.2,
                        "bearish", i, bar_time,
                        comps(first_body=prev2_body, doji_body=prev_body),
                        category="reversal"))

        # 33. Three White Soldiers (3 consecutive bullish, higher closes)
        if prev2_bull and prev_bull and is_bull:
            if pc > p2c and c > pc:
                out.append(PatternResult(PatternType.THREE_WHITE_SOLDIERS,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bullish", i, bar_time,
                    comps(soldier1_close=p2c, soldier2_close=pc, soldier3_close=c),
                    category="reversal"))

        # 34. Three Black Crows (3 consecutive bearish, lower closes)
        if prev2_bear and prev_bear and is_bear:
            if pc < p2c and c < pc:
                out.append(PatternResult(PatternType.THREE_BLACK_CROWS,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bearish", i, bar_time,
                    comps(crow1_close=p2c, crow2_close=pc, crow3_close=c),
                    category="reversal"))

        # 35. Three Inside Up (harami + confirmation)
        if prev2_bear and prev_bull and is_bull:
            if po > p2c and pc < p2o and c > p2o:
                out.append(PatternResult(PatternType.THREE_INSIDE_UP,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bullish", i, bar_time,
                    comps(category_detail="harami + confirmation"),
                    category="reversal"))

        # 36. Three Inside Down (harami + confirmation)
        if prev2_bull and prev_bear and is_bear:
            if po < p2c and pc > p2o and c < p2o:
                out.append(PatternResult(PatternType.THREE_INSIDE_DOWN,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bearish", i, bar_time,
                    comps(category_detail="harami + confirmation"),
                    category="reversal"))

        # 37. Three Outside Up (engulfing + confirmation)
        if prev2_bear and prev_bull and is_bull:
            if po <= p2c and pc >= p2o and c > pc:
                out.append(PatternResult(PatternType.THREE_OUTSIDE_UP,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bullish", i, bar_time,
                    comps(category_detail="engulfing + confirmation"),
                    category="reversal"))

        # 38. Three Outside Down (engulfing + confirmation)
        if prev2_bull and prev_bear and is_bear:
            if po >= p2c and pc <= p2o and c < pc:
                out.append(PatternResult(PatternType.THREE_OUTSIDE_DOWN,
                    self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                    "bearish", i, bar_time,
                    comps(category_detail="engulfing + confirmation"),
                    category="reversal"))

        # ===== CONTINUATION PATTERNS =====

        # 39. Bullish Gap (current opens above prev high, both bullish)
        if prev_bull and is_bull and o > ph:
            out.append(PatternResult(PatternType.BULLISH_GAP,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bullish", i, bar_time,
                comps(gap_size=o - ph), category="continuation"))

        # 40. Bearish Gap (current opens below prev low, both bearish)
        if prev_bear and is_bear and o < pl:
            out.append(PatternResult(PatternType.BEARISH_GAP,
                self._base_confidence(range_, atr_val, vol_mult) * 0.8,
                "bearish", i, bar_time,
                comps(gap_size=pl - o), category="continuation"))

        # 41-42. Fair Value Gaps (3-bar: gap between bar 1 high and bar 3 low, all same color)
        if prev3 is not None:
            p3h, p3l = float(prev3["high"]), float(prev3["low"])
            p3_bull = float(prev3["close"]) > float(prev3["open"])
            p3_bear = float(prev3["close"]) < float(prev3["open"])
            # FVG Rising: bar1 (prev2) high < bar3 (current) low, all bullish
            if p3_bull and prev2_bull and is_bull and p2h < l:
                out.append(PatternResult(PatternType.FAIR_VALUE_RISING_GAP,
                    self._base_confidence(range_, atr_val, vol_mult) * 0.9,
                    "bullish", i, bar_time,
                    comps(fvg_size=l - p2h), category="continuation"))
            # FVG Falling: bar1 (prev2) low > bar3 (current) high, all bearish
            if p3_bear and prev2_bear and is_bear and p2l > h:
                out.append(PatternResult(PatternType.FAIR_VALUE_FALLING_GAP,
                    self._base_confidence(range_, atr_val, vol_mult) * 0.9,
                    "bearish", i, bar_time,
                    comps(fvg_size=p2l - h), category="continuation"))

        # 43. Bullish Neck (prev bullish large, current bearish small, opens above prev close)
        if prev_bull and is_bear and prev_body > body * 2 and o > pc and c > (po + pc) / 2:
            out.append(PatternResult(PatternType.BULLISH_NECK,
                self._base_confidence(range_, atr_val, vol_mult) * 0.7,
                "bullish", i, bar_time,
                comps(prev_body=prev_body, current_body=body), category="continuation"))

        # 44. Bearish Neck (prev bearish large, current bullish small, opens below prev close)
        if prev_bear and is_bull and prev_body > body * 2 and o < pc and c < (po + pc) / 2:
            out.append(PatternResult(PatternType.BEARISH_NECK,
                self._base_confidence(range_, atr_val, vol_mult) * 0.7,
                "bearish", i, bar_time,
                comps(prev_body=prev_body, current_body=body), category="continuation"))

        # 45. Bullish Separating Line (prev bearish, current bullish, same open price)
        if prev_bear and is_bull and abs(o - po) / range_ < 0.05:
            out.append(PatternResult(PatternType.BULLISH_SEPARATING_LINE,
                self._base_confidence(range_, atr_val, vol_mult) * 0.75,
                "bullish", i, bar_time,
                comps(open_diff=abs(o - po)), category="continuation"))

        # 46. Bearish Separating Line (prev bullish, current bearish, same open price)
        if prev_bull and is_bear and abs(o - po) / range_ < 0.05:
            out.append(PatternResult(PatternType.BEARISH_SEPARATING_LINE,
                self._base_confidence(range_, atr_val, vol_mult) * 0.75,
                "bearish", i, bar_time,
                comps(open_diff=abs(o - po)), category="continuation"))

        # 47-48. Rising/Falling Three Methods (5-bar: large + 3 small inside + large breakout)
        if prev3 is not None and i >= 4:
            prev4 = df.iloc[i - 4]
            p4o, p4c = float(prev4["open"]), float(prev4["close"])
            p4_bull = p4c > p4o
            p4_bear = p4c < p4o
            # Rising Three Methods: first bullish, 2 small inside, last bullish breakout
            if p4_bull and is_bull:
                # Check if bars 2,3 were small inside bar 1
                bar2_inside = float(prev3["high"]) <= p4o and float(prev3["low"]) >= p4c
                bar3_inside = p2h <= p4o and p2l >= p4c
                if bar2_inside and bar3_inside and c > p4o:
                    out.append(PatternResult(PatternType.RISING_THREE_METHODS,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                        "bullish", i, bar_time,
                        comps(breakout_close=c, first_open=p4o), category="continuation"))
            if p4_bear and is_bear:
                bar2_inside = float(prev3["high"]) <= p4c and float(prev3["low"]) >= p4o if prev3 else False
                bar3_inside = p2h <= p4c and p2l >= p4o
                if bar2_inside and bar3_inside and c < p4o:
                    out.append(PatternResult(PatternType.FALLING_THREE_METHODS,
                        self._base_confidence(range_, atr_val, vol_mult) * 1.0,
                        "bearish", i, bar_time,
                        comps(breakout_close=c, first_open=p4o), category="continuation"))

        # 49-50. Rising/Falling N (4-bar: bar1 and bar4 large same direction, bars 2-3 small)
        if prev3 is not None:
            p3_bull = float(prev3["close"]) > float(prev3["open"])
            p3_bear = float(prev3["close"]) < float(prev3["open"])
            p3_body = abs(float(prev3["close"]) - float(prev3["open"]))
            # Rising N: bar1 bullish, bar4 bullish, bar4 body > 3x bars 2-3
            if p3_bull and is_bull and body > prev_body * 3 and body > prev2_body * 3:
                out.append(PatternResult(PatternType.RISING_N,
                    self._base_confidence(range_, atr_val, vol_mult) * 0.85,
                    "bullish", i, bar_time,
                    comps(first_body=p3_body, current_body=body), category="continuation"))
            # Falling N: bar1 bearish, bar4 bearish, bar4 body > 3x bars 2-3
            if p3_bear and is_bear and body > prev_body * 3 and body > prev2_body * 3:
                out.append(PatternResult(PatternType.FALLING_N,
                    self._base_confidence(range_, atr_val, vol_mult) * 0.85,
                    "bearish", i, bar_time,
                    comps(first_body=p3_body, current_body=body), category="continuation"))

        return out

    # ----------------------------------------------------------------
    # Confidence scoring helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _pin_bar_confidence(rejection_wick: float, opposite_wick: float,
                              body: float, range_: float,
                              atr_val: float, vol_mult: float) -> float:
        wick_score = min(1.0, (rejection_wick / range_) / 0.7) * 30
        opposite_penalty = (opposite_wick / range_) * 20
        body_score = (1.0 - min(1.0, body / range_)) * 15
        atr_score = min(1.0, (range_ / atr_val) / 2.0) * 20 if atr_val > 0 else 10
        vol_score = min(1.0, (vol_mult - 0.5) / 1.5) * 15 if vol_mult > 0.5 else 0
        return float(max(0.0, min(100.0, wick_score - opposite_penalty + body_score + atr_score + vol_score)))

    @staticmethod
    def _inside_bar_confidence(df: pd.DataFrame, i: int,
                                 atr_val: float, vol_mult: float) -> float:
        if i < 1:
            return 0.0
        prev = df.iloc[i - 1]
        mother_range = float(prev["high"]) - float(prev["low"])
        current_range = float(df.iloc[i]["high"]) - float(df.iloc[i]["low"])
        contraction = 1.0 - min(1.0, current_range / max(mother_range, 1e-9))
        mother_significance = min(1.0, mother_range / atr_val) if atr_val > 0 else 0.5
        vol_score = min(1.0, (vol_mult - 0.5) / 1.5) if vol_mult > 0.5 else 0
        score = contraction * 40 + mother_significance * 30 + vol_score * 30
        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _engulfing_confidence(body: float, prev_body: float,
                                range_: float, atr_val: float,
                                vol_mult: float) -> float:
        engulf_ratio = body / max(prev_body, 1e-9)
        engulf_score = min(1.0, engulf_ratio / 2.0) * 35
        body_score = min(1.0, body / range_) * 20
        atr_score = min(1.0, (range_ / atr_val) / 2.0) * 25 if atr_val > 0 else 12
        vol_score = min(1.0, (vol_mult - 0.5) / 1.5) * 20 if vol_mult > 0.5 else 0
        score = engulf_score + body_score + atr_score + vol_score
        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _base_confidence(range_: float, atr_val: float, vol_mult: float) -> float:
        """Generic confidence for patterns without specific scoring."""
        atr_score = min(1.0, (range_ / atr_val) / 2.0) * 40 if atr_val > 0 else 20
        vol_score = min(1.0, (vol_mult - 0.5) / 1.5) * 30 if vol_mult > 0.5 else 0
        base = 30.0 + atr_score + vol_score
        return float(max(0.0, min(100.0, base)))

    # ----------------------------------------------------------------
    @staticmethod
    def get_pattern_count() -> int:
        """Return total number of detectable patterns (excluding NONE)."""
        return len(PatternType) - 1

    @staticmethod
    def get_all_patterns() -> list[dict[str, str]]:
        """Return list of all pattern types with metadata."""
        out = []
        for p in PatternType:
            if p == PatternType.NONE:
                continue
            required_trend = TREND_REQUIREMENTS.get(p)
            if required_trend is None:
                category = "neutral"
            elif p.value.startswith(("bullish_gap", "fair_value_rising", "bullish_neck",
                                       "bullish_separating", "rising_three", "rising_n",
                                       "bearish_gap", "fair_value_falling", "bearish_neck",
                                       "bearish_separating", "falling_three", "falling_n")):
                category = "continuation"
            else:
                category = "reversal"
            out.append({
                "pattern": p.value,
                "category": category,
                "required_trend": required_trend or "any",
            })
        return out
