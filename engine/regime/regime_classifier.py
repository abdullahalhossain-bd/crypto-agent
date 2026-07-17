"""engine.regime.regime_classifier
=====================================================================
Day 23 — Market regime classifier.

Classifies the current bar into one of:
  - "trend"     : directional, low chop, ATR stable
  - "chop"      : mean-reverting, low directional conviction
  - "high_vol"  : ATR > 1.5x baseline; reduce risk
  - "calm"      : ATR < 0.7x baseline, low ret magnitude

Method (deliberately simple, explainable):
  1. Compute ATR ratio (current ATR / 50-bar median ATR).
  2. Compute ADX-like trend strength (|slope of 20-bar linreg| / ATR).
  3. Compute realised volatility vs. baseline.
  4. Apply thresholds → label + soft confidence.

Returns a `RegimeState` with the label, confidence, and the underlying
metrics so the allocator and observability layer can audit the call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("engine.regime")


@dataclass
class RegimeState:
    label: str                    # trend | chop | high_vol | calm
    confidence: float             # 0..1
    atr_ratio: float
    trend_strength: float
    ret_volatility: float
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "atr_ratio": self.atr_ratio,
            "trend_strength": self.trend_strength,
            "ret_volatility": self.ret_volatility,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class RegimeClassifier:
    def __init__(self,
                 atr_period: int = 14,
                 baseline_period: int = 50,
                 trend_window: int = 20,
                 high_vol_threshold: float = 1.5,
                 calm_threshold: float = 0.7,
                 trend_strength_threshold: float = 0.6) -> None:
        self.atr_period = atr_period
        self.baseline_period = baseline_period
        self.trend_window = trend_window
        self.high_vol_threshold = high_vol_threshold
        self.calm_threshold = calm_threshold
        self.trend_strength_threshold = trend_strength_threshold

    # ----------------------------------------------------------------
    def classify(self, df: pd.DataFrame) -> RegimeState:
        if len(df) < max(self.baseline_period, self.trend_window) + 2:
            return RegimeState(
                label="calm", confidence=0.0,
                atr_ratio=1.0, trend_strength=0.0, ret_volatility=0.0,
                components={"reason": "warmup"},
            )

        # ATR + baseline
        a = atr(df, self.atr_period)
        # C18 fix: if baseline_period > len(df), rolling().median() returns
        # all NaN, which makes baseline_now fall back to atr_now (ratio=1.0).
        # This is already handled by the `else atr_now` fallback below, but
        # we add a min_periods floor so short dataframes still produce a
        # usable baseline instead of a guaranteed-NaN series.
        min_periods = min(20, max(5, len(df) // 3))
        baseline = a.rolling(self.baseline_period,
                              min_periods=min_periods).median()
        atr_now = float(a.iloc[-1]) if not a.isna().iloc[-1] else 0.0
        # C18 fix: if baseline is NaN (short df), fall back to atr_now;
        # if atr_now is also 0, fall back to a tiny positive number so
        # the ratio doesn't divide by zero.
        baseline_now = float(baseline.iloc[-1]) if not baseline.isna().iloc[-1] else atr_now
        if baseline_now <= 0:
            baseline_now = max(atr_now, 1e-8)
        atr_ratio = (atr_now / baseline_now) if baseline_now > 0 else 1.0

        # Realised vol (10-bar std of log-returns)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        ret_vol = float(log_ret.tail(10).std() or 0.0)
        ret_vol_baseline = float(log_ret.tail(self.baseline_period).std() or ret_vol)
        ret_vol_ratio = (ret_vol / ret_vol_baseline) if ret_vol_baseline > 0 else 1.0

        # Trend strength: |slope of last N closes| / ATR
        last_n = df["close"].tail(self.trend_window).values
        x = np.arange(len(last_n), dtype=float)
        if len(last_n) >= 2:
            denom = ((x - x.mean()) ** 2).sum()
            if denom > 0:
                slope = float(((x - x.mean()) * (last_n - last_n.mean())).sum() / denom)
            else:
                slope = 0.0
        else:
            slope = 0.0
        trend_strength = abs(slope) / atr_now if atr_now > 0 else 0.0

        # ---- Decision tree (soft) ----
        # High-vol overrides everything
        if atr_ratio >= self.high_vol_threshold:
            label = "high_vol"
            # Confidence scales with how extreme the ratio is
            confidence = float(min(1.0, (atr_ratio - 1.0) / 1.5))
        elif atr_ratio <= self.calm_threshold:
            label = "calm"
            confidence = float(min(1.0, (1.0 - atr_ratio) / 0.5))
        elif trend_strength >= self.trend_strength_threshold:
            label = "trend"
            confidence = float(min(1.0, trend_strength / 1.5))
        else:
            label = "chop"
            confidence = float(min(1.0, max(0.0, 1.0 - trend_strength / self.trend_strength_threshold)))

        components = {
            "atr_now": atr_now,
            "baseline_atr": baseline_now,
            "ret_vol": ret_vol,
            "ret_vol_baseline": ret_vol_baseline,
            "slope": slope,
            "atr_ratio": atr_ratio,
            "ret_vol_ratio": ret_vol_ratio,
            "trend_strength": trend_strength,
        }
        log.debug("REGIME %s conf=%.2f atr_ratio=%.2f trend=%.2f",
                  label, confidence, atr_ratio, trend_strength)
        return RegimeState(
            label=label, confidence=confidence,
            atr_ratio=float(atr_ratio),
            trend_strength=float(trend_strength),
            ret_volatility=float(ret_vol_ratio),
            components=components,
        )
