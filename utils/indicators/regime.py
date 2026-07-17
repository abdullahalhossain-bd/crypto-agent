"""utils/indicators/regime.py
=====================================================================
Volatility + Trend Regime Detection (Improvements #9 + #10)
=====================================================================
Volatility regimes: LOW, NORMAL, HIGH, EXTREME
Trend regimes: TRENDING, RANGING, BREAKOUT, REVERSAL,
                CONSOLIDATION, ACCUMULATION, DISTRIBUTION
"""
from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.trend import ema
from utils.indicators.volatility import atr, atr_pct, bollinger_bands
from utils.indicators.momentum import rsi


class VolatilityRegime(str, Enum):
    LOW = "low_volatility"
    NORMAL = "normal"
    HIGH = "high_volatility"
    EXTREME = "extreme_volatility"


class TrendRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


@cached()
def volatility_regime(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Detect volatility regime from ATR percentile.

    LOW: ATR < 20th percentile
    NORMAL: 20-80th percentile
    HIGH: 80-95th percentile
    EXTREME: > 95th percentile
    """
    atr_p = atr_pct(df, period=14)
    percentile = atr_p.rolling(period).rank(pct=True)

    regime = pd.Series("normal", index=df.index)
    regime[percentile < 0.20] = VolatilityRegime.LOW.value
    regime[percentile >= 0.80] = VolatilityRegime.HIGH.value
    regime[percentile >= 0.95] = VolatilityRegime.EXTREME.value
    return regime


@cached()
def trend_regime(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Detect trend regime using ADX + price action + BBands.

    TRENDING: ADX > 25, clear EMA stacking
    RANGING: ADX < 20, price oscillating
    BREAKOUT: BBands expanding after consolidation
    REVERSAL: recent BoS/ChoCH (price broke key level)
    CONSOLIDATION: BBands narrow, low ADX
    ACCUMULATION: range-bound after downtrend, volume increasing
    DISTRIBUTION: range-bound after uptrend, volume increasing
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df.get("volume", pd.Series(1, index=df.index))

    # ADX-like trend strength (simplified — use rolling slope)
    ema_val = ema(close, period)
    slope = ema_val.diff(5) / ema_val.shift(5).replace(0, np.nan)
    adx_like = slope.abs() * 1000  # scale up

    # BBands width for consolidation/breakout detection
    _, mid, _, width = bollinger_bands(close, period=20)
    bb_width = width
    avg_width = bb_width.rolling(period).mean()
    width_ratio = bb_width / avg_width.replace(0, np.nan)

    # RSI for accumulation/distribution
    rsi_val = rsi(close)

    # Volume trend
    vol_ma = vol.rolling(period).mean()
    vol_increasing = vol > vol_ma * 1.2

    # Recent price action
    recent_high = high.rolling(period).max()
    recent_low = low.rolling(period).min()
    price_pos = (close - recent_low) / (recent_high - recent_low).replace(0, np.nan)

    regime = pd.Series("ranging", index=df.index)

    # Strong trend
    trending = (adx_like > 25) & (width_ratio > 0.8)
    regime[trending] = TrendRegime.TRENDING.value

    # Consolidation (narrow bands, low ADX)
    consolidation = (width_ratio < 0.6) & (adx_like < 15)
    regime[consolidation] = TrendRegime.CONSOLIDATION.value

    # Breakout (bands expanding rapidly from consolidation)
    breakout = (width_ratio > 1.3) & (width_ratio.shift(1) < 0.8)
    regime[breakout] = TrendRegime.BREAKOUT.value

    # Accumulation: range-bound after downtrend, volume increasing
    was_downtrend = close < ema(close, 50).shift(20)
    accumulation = (rsi_val < 50) & was_downtrend & vol_increasing & (price_pos < 0.4)
    regime[accumulation] = TrendRegime.ACCUMULATION.value

    # Distribution: range-bound after uptrend, volume increasing
    was_uptrend = close > ema(close, 50).shift(20)
    distribution = (rsi_val > 50) & was_uptrend & vol_increasing & (price_pos > 0.6)
    regime[distribution] = TrendRegime.DISTRIBUTION.value

    # Reversal: trend was strong but now weakening sharply
    trend_weakening = (adx_like.shift(3) > 25) & (adx_like < adx_like.shift(3) * 0.6)
    regime[trend_weakening] = TrendRegime.REVERSAL.value

    return regime


@cached()
def regime_detection(df: pd.DataFrame, period: int = 50) -> Dict[str, str]:
    """Combined regime detection — returns dict with 'regime' + details.

    This is the main entry point used by the trading bot.
    """
    try:
        v_regime = volatility_regime(df, period).iloc[-1]
    except Exception:
        v_regime = "normal"
    try:
        t_regime = trend_regime(df, period).iloc[-1]
    except Exception:
        t_regime = "ranging"

    # Combine into a single regime label
    if v_regime == "extreme_volatility":
        regime = "crisis"
    elif t_regime == "trending":
        regime = "trend_up" if df["close"].iloc[-1] > ema(df["close"], 50).iloc[-1] else "trend_down"
    elif t_regime == "breakout":
        regime = "breakout"
    elif t_regime in ("consolidation", "ranging"):
        regime = "range"
    elif t_regime == "reversal":
        regime = "transition"
    elif t_regime == "accumulation":
        regime = "accumulation"
    elif t_regime == "distribution":
        regime = "distribution"
    else:
        regime = "unknown"

    return {
        "regime": regime,
        "volatility_regime": v_regime,
        "trend_regime": t_regime,
    }
