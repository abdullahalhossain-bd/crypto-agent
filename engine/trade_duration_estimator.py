"""engine.trade_duration_estimator
=====================================================================
Estimates how many hours it will take to reach the target gain
based on hourly ATR as a percentage of price.

Formula: estimated_hours = TARGET_GAIN_PCT / (ATR_pct_per_hour × efficiency)
  efficiency = 0.8 (accounts for non-directional movement)

If estimated_hours > MAX_HOURS, the trade is flagged as not viable.

This gives the trade quality scorer another dimension: even if the
confluence score is high, if it would take 20 hours to reach TP,
the trade might not be worth the capital lock-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("engine.duration_estimator")

MAX_HOURS = 5.0           # max acceptable hold time
TARGET_GAIN_PCT = 5.2     # TP2 target
EFFICIENCY = 0.8          # 80% of ATR is directional


@dataclass
class DurationEstimate:
    estimated_hours: float
    atr_pct_per_hour: float
    viable: bool           # False if estimated_hours > MAX_HOURS
    target_gain_pct: float = TARGET_GAIN_PCT
    max_hours: float = MAX_HOURS

    def to_dict(self) -> dict:
        return {
            "estimated_hours": self.estimated_hours,
            "atr_pct_per_hour": self.atr_pct_per_hour,
            "viable": self.viable,
            "target_gain_pct": self.target_gain_pct,
            "max_hours": self.max_hours,
        }


# ----------------------------------------------------------------------
class TradeDurationEstimator:
    """Estimates hours-to-TP based on ATR."""

    def __init__(self,
                 max_hours: float = MAX_HOURS,
                 target_gain_pct: float = TARGET_GAIN_PCT,
                 efficiency: float = EFFICIENCY) -> None:
        self.max_hours = float(max_hours)
        self.target_gain_pct = float(target_gain_pct)
        self.efficiency = float(efficiency)

    # ----------------------------------------------------------------
    def estimate(self, df: pd.DataFrame,
                   timeframe: str = "1h") -> DurationEstimate:
        """Estimate hours to reach target gain.

        Args:
            df: OHLCV DataFrame
            timeframe: bar timeframe ("1h", "15min", "4h", "1d")

        Returns:
            DurationEstimate with estimated_hours + viability
        """
        if len(df) < 15:
            return DurationEstimate(
                estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False,
            )

        atr_series = atr(df, 14)
        if atr_series.isna().iloc[-1]:
            return DurationEstimate(
                estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False,
            )

        atr_val = float(atr_series.iloc[-1])
        price = float(df["close"].iloc[-1])
        if price <= 0 or atr_val <= 0:
            return DurationEstimate(
                estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False,
            )

        # ATR as % of price per bar
        atr_pct_per_bar = (atr_val / price) * 100

        # Convert to per-hour based on timeframe
        hours_per_bar = self._timeframe_to_hours(timeframe)
        atr_pct_per_hour = atr_pct_per_bar / hours_per_bar if hours_per_bar > 0 else atr_pct_per_bar

        if atr_pct_per_hour <= 0:
            return DurationEstimate(
                estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False,
            )

        # Estimate hours to target
        effective_atr = atr_pct_per_hour * self.efficiency
        estimated = self.target_gain_pct / effective_atr if effective_atr > 0 else 999.0

        return DurationEstimate(
            estimated_hours=round(estimated, 2),
            atr_pct_per_hour=round(atr_pct_per_hour, 4),
            viable=estimated <= self.max_hours,
            target_gain_pct=self.target_gain_pct,
            max_hours=self.max_hours,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _timeframe_to_hours(tf: str) -> float:
        """Convert timeframe string to hours per bar."""
        tf = tf.lower().strip()
        if tf in ("1m", "m1", "1min"):
            return 1 / 60
        if tf in ("5m", "m5", "5min"):
            return 5 / 60
        if tf in ("15m", "m15", "15min"):
            return 15 / 60
        if tf in ("30m", "m30", "30min"):
            return 30 / 60
        if tf in ("1h", "h1", "60m"):
            return 1.0
        if tf in ("4h", "h4"):
            return 4.0
        if tf in ("1d", "d1", "1day"):
            return 24.0
        if tf in ("1w", "w1", "1week"):
            return 24.0 * 7
        # H20 fix: raise ValueError for unknown timeframes instead of
        # silently defaulting to 1h, which produces incorrect duration
        # estimates without any signal to the operator.
        raise ValueError(f"unknown timeframe: {tf!r} — expected one of "
                         f"1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w")
