"""engine.candlestick.candlestick_features
=====================================================================
Day 138 — Candlestick feature extractor.

Instead of "Pin Bar = BUY", we extract 25+ features per bar so the
ML feature store can learn which combinations are predictive.

Features (per bar):
  Body:
    1. body_size_pct          (body / range)
    2. body_direction         (+1 bullish, -1 bearish)
    3. body_atr_ratio         (body / ATR)
  Wicks:
    4. upper_wick_pct
    5. lower_wick_pct
    6. wick_imbalance         (lower - upper / range)
    7. rejection_strength     (max wick - body) / range
  Range:
    8. range_atr_ratio        (bar size vs. recent vol)
    9. range_pct              (range / close)
  Volume:
    10. volume_mult           (vs. 20-bar avg)
    11. volume_trend          (recent vs. baseline)
  Position:
    12. close_position        (0 = low, 1 = high)
    13. close_vs_sma_fast     (close - sma(20)) / atr
    14. close_vs_sma_slow     (close - sma(50)) / atr
  Patterns (binary):
    15. is_pin_bar_bullish
    16. is_pin_bar_bearish
    17. is_inside_bar
    18. is_engulfing_bullish
    19. is_engulfing_bearish
    20. is_doji
  Pattern confidence:
    21. pattern_confidence    (0-100 from pattern_detector)
    22. rejection_score       (0-100 from rejection_strength)
  Context:
    23. market_state_score    (TREND=1, RANGE=0, CHOPPY=-1)
    24. trend_slope_atr
    25. autocorr_lag1
    26. mtf_alignment_score
    27. sr_proximity_score
    28. pullback_quality
    29. false_breakout_prob
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.candlestick.confluence_engine import ConfluenceEngine
from engine.candlestick.false_breakout import FalseBreakoutDetector
from engine.candlestick.market_state import MarketStateClassifier
from engine.candlestick.multi_timeframe import MultiTimeframeConfirmator
from engine.candlestick.pattern_detector import PatternDetector, PatternType
from engine.candlestick.pattern_decay import PatternDecayTracker
from engine.candlestick.pullback_detector import PullbackDetector
from engine.candlestick.rejection_strength import RejectionStrengthScorer
from engine.candlestick.sr_confidence import SupportResistanceConfidence
from utils.indicators import atr, sma


# ----------------------------------------------------------------------
class CandlestickFeatureExtractor:
    """Extract 25+ candlestick-specific features per bar."""

    # Largest lookback needed by any sub-detector. MultiTimeframeConfirmator
    # resamples the base df up to D1 and needs ~20 D1 candles for its trend
    # calc — how many BASE-timeframe bars that requires depends entirely on
    # what timeframe `df` is in (e.g. 20 D1 candles needs ~480 H1 bars, or
    # ~1920 M15 bars). There is no safe universal constant here, so this is
    # a constructor parameter, not a hardcoded default — callers must set it
    # to comfortably exceed their own base-timeframe bar count for ~20+ D1
    # candles, or MTF's D1 trend read will silently degrade.
    DEFAULT_MAX_LOOKBACK_BARS = 2000

    def __init__(self, max_lookback_bars: int = DEFAULT_MAX_LOOKBACK_BARS) -> None:
        self.pattern_detector = PatternDetector()
        self.rejection_scorer = RejectionStrengthScorer()
        self.market_state_clf = MarketStateClassifier()
        self.pullback_detector = PullbackDetector()
        self.false_breakout = FalseBreakoutDetector()
        self.sr_confidence = SupportResistanceConfidence()
        self.mtf = MultiTimeframeConfirmator()
        self.max_lookback_bars = int(max_lookback_bars)

    # ----------------------------------------------------------------
    def extract(self, df: pd.DataFrame) -> dict[str, float]:
        """Extract features for the LAST bar of `df`."""
        if len(df) < 50:
            return {}
        bar = df.iloc[-1]
        open_, high, low, close = (
            float(bar["open"]), float(bar["high"]),
            float(bar["low"]), float(bar["close"]),
        )
        body = abs(close - open_)
        range_ = high - low
        if range_ <= 0:
            return {}
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        body_dir = 1.0 if close > open_ else (-1.0 if close < open_ else 0.0)
        close_position = (close - low) / range_

        atr_series = atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else range_
        sma_fast = sma(df["close"], 20)
        sma_slow = sma(df["close"], 50)
        sma_f_val = float(sma_fast.iloc[-1]) if not sma_fast.isna().iloc[-1] else close
        sma_s_val = float(sma_slow.iloc[-1]) if not sma_slow.isna().iloc[-1] else close

        # Volume
        vol_mult = 1.0
        vol_trend = 0.0
        if "volume" in df.columns and len(df) >= 20:
            vol = df["volume"]
            vol_mean = vol.rolling(20, min_periods=5).mean()
            if not vol_mean.isna().iloc[-1] and vol_mean.iloc[-1] > 0:
                vol_mult = float(vol.iloc[-1] / vol_mean.iloc[-1])
            if len(vol) >= 50:
                recent = vol.tail(20).mean()
                baseline = vol.tail(50).mean()
                if baseline > 0:
                    vol_trend = float((recent - baseline) / baseline)

        # Patterns
        pattern_result = self.pattern_detector.detect_latest(df)
        rejection_result = self.rejection_scorer.score(df)

        # Market state
        market_state = self.market_state_clf.classify(df)
        market_score = {"TREND": 1.0, "RANGE": 0.0, "CHOPPY": -1.0}.get(
            market_state.label, 0.0
        )

        # Pullback
        pullback = self.pullback_detector.detect(df)

        # False breakout
        false_bo = self.false_breakout.detect(df)

        # S/R
        sr = self.sr_confidence.detect_and_score(df)

        # MTF
        mtf = self.mtf.confirm(df)

        return {
            # Body
            "body_size_pct": float(body / range_),
            "body_direction": float(body_dir),
            "body_atr_ratio": float(body / max(atr_val, 1e-9)),
            # Wicks
            "upper_wick_pct": float(upper_wick / range_),
            "lower_wick_pct": float(lower_wick / range_),
            "wick_imbalance": float((lower_wick - upper_wick) / range_),
            "rejection_strength": float((max(upper_wick, lower_wick) - body) / range_),
            # Range
            "range_atr_ratio": float(range_ / max(atr_val, 1e-9)),
            "range_pct": float(range_ / close),
            # Volume
            "volume_mult": float(vol_mult),
            "volume_trend": float(vol_trend),
            # Position
            "close_position": float(close_position),
            "close_vs_sma_fast": float((close - sma_f_val) / max(atr_val, 1e-9)),
            "close_vs_sma_slow": float((close - sma_s_val) / max(atr_val, 1e-9)),
            # Patterns (binary)
            "is_pin_bar_bullish": float(pattern_result.pattern == PatternType.PIN_BAR_BULLISH),
            "is_pin_bar_bearish": float(pattern_result.pattern == PatternType.PIN_BAR_BEARISH),
            "is_inside_bar": float(pattern_result.pattern == PatternType.INSIDE_BAR),
            "is_engulfing_bullish": float(pattern_result.pattern == PatternType.ENGULFING_BULLISH),
            "is_engulfing_bearish": float(pattern_result.pattern == PatternType.ENGULFING_BEARISH),
            "is_doji": float(pattern_result.pattern == PatternType.DOJI),
            # Pattern scores
            "pattern_confidence": float(pattern_result.confidence),
            "rejection_score": float(rejection_result.score),
            # Context
            "market_state_score": float(market_score),
            "trend_slope_atr": float(market_state.slope),
            "autocorr_lag1": float(market_state.autocorr),
            "mtf_alignment_score": float(mtf.score),
            "sr_proximity_score": float(sr.summary_score),
            "pullback_quality": float(pullback.quality),
            "false_breakout_prob": float(false_bo.probability),
        }

    # ----------------------------------------------------------------
    def extract_series(self, df: pd.DataFrame, min_history: int = 50
                        ) -> pd.DataFrame:
        """Extract features for every bar (after warmup).

        Each call to `extract()` is bounded to the last
        `self.max_lookback_bars` bars (instead of the entire history up
        to bar i). This turns what was an O(n^2) backtest scan
        (recomputing every sub-detector, including MTF resampling, over
        an ever-larger slice for every bar) into O(n) with a bounded
        constant factor per bar. Set `max_lookback_bars` generously
        relative to your base timeframe (see class docstring constant) —
        too small a cap will degrade MultiTimeframeConfirmator's D1/H4
        trend reads once bar index exceeds the cap.
        """
        records: list[dict[str, Any]] = []
        for i in range(min_history, len(df)):
            start = max(0, i + 1 - self.max_lookback_bars)
            window = df.iloc[start: i + 1]
            features = self.extract(window)
            features["bar_index"] = i
            if "time" in df.columns:
                features["time"] = df["time"].iloc[i]
            records.append(features)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)