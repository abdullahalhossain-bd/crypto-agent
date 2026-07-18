"""engine.candlestick.market_state
=====================================================================
Day 128 — Market State Classifier (TREND / RANGE / CHOPPY).

The Candlestick Trading Bible is explicit: identify market state
BEFORE applying any strategy. Trend strategies fail in ranges;
mean-reversion fails in trends.

Different from engine.regime.regime_classifier (which produces
"trend/chop/high_vol/calm" using ATR + slope). This module uses
a tighter, more classical definition:

  - TREND   : ADX-like directional strength, persistently making
              higher highs + higher lows (or lower lows + lower highs)
  - RANGE   : price oscillates between horizontal S/R, low directional
              conviction, ADX-like metric low
  - CHOPPY  : high volatility + low direction (whipsaw, hard to trade)

Method (no external dependency on ADX; we synthesize it):
  1. Compute slope of 20-bar linear regression
  2. Compute autocorrelation of returns (lag-1)
  3. Compute ATR ratio vs. baseline
  4. Compute "swing consistency" — do swings persist in one direction?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.indicators import atr
from utils.logger import get_logger

# BUG FIX: `log` was used in classify()'s except-handler but never defined
# anywhere in this module, so a classifier failure raised NameError instead
# of falling back safely to a low-confidence RANGE state as intended.
log = get_logger("engine.candlestick.market_state")


@dataclass
class MarketState:
    label: str                # "TREND" | "RANGE" | "CHOPPY"
    confidence: float         # 0-1
    direction: str            # "up" | "down" | "none" (only meaningful for TREND)
    slope: float
    autocorr: float
    atr_ratio: float
    swing_consistency: float
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "direction": self.direction,
            "slope": self.slope,
            "autocorr": self.autocorr,
            "atr_ratio": self.atr_ratio,
            "swing_consistency": self.swing_consistency,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class MarketStateClassifier:
    def __init__(self,
                 trend_window: int = 20,
                 atr_period: int = 14,
                 baseline_period: int = 50,
                 trend_slope_threshold: float = 0.6,
                 range_autocorr_threshold: float = -0.1,
                 choppy_atr_ratio: float = 1.8) -> None:
        self.trend_window = int(trend_window)
        self.atr_period = int(atr_period)
        self.baseline_period = int(baseline_period)
        self.trend_slope_threshold = float(trend_slope_threshold)
        self.range_autocorr_threshold = float(range_autocorr_threshold)
        self.choppy_atr_ratio = float(choppy_atr_ratio)

    # ----------------------------------------------------------------
    def classify(self, df: pd.DataFrame) -> MarketState:
        # C10 fix: wrap the entire classification in try-except so a bug
        # in the classifier doesn't crash the trading cycle. On failure,
        # return a low-confidence RANGE state so downstream gates can
        # decide whether to trade.
        try:
            return self._classify_impl(df)
        except Exception as e:
            log.error("MarketStateClassifier.classify raised — returning safe RANGE: %r", e)
            return MarketState(
                label="RANGE", confidence=0.0, direction="none",
                slope=0.0, autocorr=0.0, atr_ratio=1.0,
                swing_consistency=0.0,
                components={"reason": f"classifier_error: {e!r}"},
            )

    def _classify_impl(self, df: pd.DataFrame) -> MarketState:
        if len(df) < max(self.trend_window, self.baseline_period, self.atr_period) + 2:
            return MarketState(
                label="RANGE", confidence=0.0, direction="none",
                slope=0.0, autocorr=0.0, atr_ratio=1.0,
                swing_consistency=0.0,
                components={"reason": "warmup"},
            )
        close = df["close"]
        # Slope of last N bars (linear regression)
        window = close.tail(self.trend_window).values
        x = np.arange(len(window), dtype=float)
        denom = ((x - x.mean()) ** 2).sum()
        slope = float(((x - x.mean()) * (window - window.mean())).sum() / denom) if denom > 0 else 0.0
        # Normalise slope by ATR
        atr_series = atr(df, self.atr_period)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 1.0
        slope_atr = slope / atr_val if atr_val > 0 else 0.0

        # Autocorrelation of returns
        log_ret = np.log(close / close.shift(1)).dropna().tail(self.trend_window)
        if len(log_ret) >= 5 and log_ret.std() > 0:
            autocorr = float(log_ret.autocorr(lag=1) or 0.0)
        else:
            autocorr = 0.0

        # ATR ratio
        baseline = atr_series.rolling(self.baseline_period, min_periods=10).median()
        baseline_val = float(baseline.iloc[-1]) if not baseline.isna().iloc[-1] else atr_val
        atr_ratio = atr_val / baseline_val if baseline_val > 0 else 1.0

        # Swing consistency: do recent swings persist in one direction?
        swing_consistency = self._swing_consistency(df)

        # ---- Decision tree ----
        # CHOPPY: high vol + low directional conviction
        if atr_ratio >= self.choppy_atr_ratio and abs(slope_atr) < 0.3:
            label = "CHOPPY"
            confidence = float(min(1.0, (atr_ratio - 1.0) / 1.0))
            direction = "none"
        # TREND: strong slope + swing consistency + positive autocorr
        elif (abs(slope_atr) >= self.trend_slope_threshold
              and swing_consistency >= 0.6
              and autocorr > 0):
            label = "TREND"
            confidence = float(min(1.0, abs(slope_atr) * 0.5 + swing_consistency * 0.5))
            direction = "up" if slope > 0 else "down"
        # RANGE: low slope + negative autocorr (mean-reverting)
        elif abs(slope_atr) < 0.3 and autocorr < self.range_autocorr_threshold:
            label = "RANGE"
            confidence = float(min(1.0, abs(autocorr) * 2 + (1.0 - abs(slope_atr))))
            direction = "none"
        # Default: weak range
        else:
            label = "RANGE"
            confidence = 0.3
            direction = "none"

        return MarketState(
            label=label, confidence=confidence,
            direction=direction, slope=float(slope_atr),
            autocorr=autocorr, atr_ratio=float(atr_ratio),
            swing_consistency=float(swing_consistency),
            components={
                "raw_slope": float(slope),
                "atr_value": atr_val,
                "baseline_atr": baseline_val,
                "n_bars": len(df),
            },
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _swing_consistency(df: pd.DataFrame) -> float:
        """Compute swing consistency: fraction of recent swing highs/lows
        that agree with the dominant direction. Returns value in [0, 1]."""
        if len(df) < 10:
            return 0.5
        # Identify swing highs/lows (peaks/troughs) over last 30 bars
        window = df.tail(30)
        highs = window["high"].values
        lows = window["low"].values
        # Simple peak detection: bar i is a peak if high[i] > high[i-1] and high[i] > high[i+1]
        peaks_high: list[float] = []
        peaks_low: list[float] = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                peaks_high.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                peaks_low.append(lows[i])
        if len(peaks_high) < 2 or len(peaks_low) < 2:
            return 0.5
        # Are peaks_high trending up (HH) and peaks_low trending up (HL)?
        # Or both down (LH, LL)?
        hh_count = sum(1 for i in range(1, len(peaks_high))
                       if peaks_high[i] > peaks_high[i - 1])
        lh_count = sum(1 for i in range(1, len(peaks_high))
                       if peaks_high[i] < peaks_high[i - 1])
        hl_count = sum(1 for i in range(1, len(peaks_low))
                       if peaks_low[i] > peaks_low[i - 1])
        ll_count = sum(1 for i in range(1, len(peaks_low))
                       if peaks_low[i] < peaks_low[i - 1])
        total = len(peaks_high) - 1 + len(peaks_low) - 1
        if total == 0:
            return 0.5
        # Higher highs + higher lows = uptrend consistency
        # Lower highs + lower lows = downtrend consistency
        up_score = hh_count + hl_count
        down_score = lh_count + ll_count
        consistency = max(up_score, down_score) / total
        return float(consistency)
