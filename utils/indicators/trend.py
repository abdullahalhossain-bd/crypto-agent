"""utils/indicators/trend.py
=====================================================================
Trend Indicators (13 indicators)
=====================================================================
EMA, SMA, WMA, HMA, VWMA, DEMA, TEMA, ZLEMA, KAMA, ALMA, T3,
SuperTrend, Ichimoku Cloud

All functions are NumPy-vectorized + cached.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.validation import assert_valid


# ----------------------------------------------------------------------
# Basic Moving Averages
# ----------------------------------------------------------------------
@cached()
def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=period, min_periods=period).mean()


@cached()
def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


@cached()
def wma(close: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average (linear weights)."""
    weights = np.arange(1, period + 1, dtype=float)
    weights = weights / weights.sum()
    return close.rolling(window=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )


@cached()
def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    pv = close * vol
    return pv.rolling(period).sum() / vol.rolling(period).sum()


@cached()
def hull_ma(close: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average — smoothed, low-lag."""
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    return wma(diff, sqrt_n)


@cached()
def dema(close: pd.Series, period: int = 20) -> pd.Series:
    """Double Exponential Moving Average."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


@cached()
def tema(close: pd.Series, period: int = 20) -> pd.Series:
    """Triple Exponential Moving Average."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


@cached()
def zlema(close: pd.Series, period: int = 20) -> pd.Series:
    """Zero-Lag EMA — removes lag using error correction."""
    lag = max(1, (period - 1) // 2)
    ema_data = 2 * close - close.shift(lag)
    return ema_data.ewm(span=period, adjust=False).mean()


@cached()
def kama(close: pd.Series, period: int = 10,
         fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average — adjusts speed to volatility."""
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama_values = pd.Series(index=close.index, dtype=float)
    kama_values.iloc[:period] = close.iloc[:period]
    for i in range(period, len(close)):
        prev = kama_values.iloc[i - 1]
        kama_values.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
    return kama_values


@cached()
def alma(close: pd.Series, period: int = 9,
         offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average — Gaussian-distributed weights."""
    m = offset * (period - 1)
    s = period / sigma
    weights = np.array([np.exp(-(i - m) ** 2 / (2 * s * s)) for i in range(period)])
    weights = weights / weights.sum()
    return close.rolling(period).apply(lambda x: np.dot(x, weights), raw=True)


@cached()
def t3_ma(close: pd.Series, period: int = 5,
          volume_factor: float = 0.7) -> pd.Series:
    """T3 Moving Average — Tillson's T3, smoothed triple EMA."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)
    c1 = -volume_factor ** 3
    c2 = 3 * volume_factor ** 2 + 3 * volume_factor ** 3
    c3 = -6 * volume_factor ** 2 - 3 * volume_factor - 3 * volume_factor ** 3
    c4 = 1 + 3 * volume_factor + volume_factor ** 3 + 3 * volume_factor ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ----------------------------------------------------------------------
# SuperTrend
# ----------------------------------------------------------------------
@cached()
def supertrend(df: pd.DataFrame, period: int = 10,
               multiplier: float = 3.0) -> pd.Series:
    """SuperTrend — ATR-based trend-following overlay.

    Returns a Series where:
        - Value above price = downtrend (resistance)
        - Value below price = uptrend (support)
    """
    from utils.indicators.volatility import atr
    df = assert_valid(df)
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(index=df.index, dtype=float)
    close = df["close"]
    direction = 1  # 1=up, -1=down
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction = -1
        if direction == 1:
            st.iloc[i] = max(lower_band.iloc[i], st.iloc[i - 1]) \
                if not pd.isna(st.iloc[i - 1]) and close.iloc[i - 1] > st.iloc[i - 1] \
                else lower_band.iloc[i]
        else:
            st.iloc[i] = min(upper_band.iloc[i], st.iloc[i - 1]) \
                if not pd.isna(st.iloc[i - 1]) and close.iloc[i - 1] < st.iloc[i - 1] \
                else upper_band.iloc[i]
    return st


# ----------------------------------------------------------------------
# Ichimoku Cloud
# ----------------------------------------------------------------------
@cached()
def ichimoku(df: pd.DataFrame,
             conversion_period: int = 9,
             base_period: int = 26,
             span_b_period: int = 52,
             displacement: int = 26) -> dict:
    """Ichimoku Cloud — 5 lines: Tenkan, Kijun, SenkouA, SenkouB, Chikou.

    Returns dict of 5 Series.
    """
    df = assert_valid(df)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    def _donchian(h, l, p):
        return (h.rolling(p).max() + l.rolling(p).min()) / 2

    tenkan = _donchian(high, low, conversion_period)
    kijun = _donchian(high, low, base_period)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = _donchian(high, low, span_b_period).shift(displacement)
    chikou = close.shift(-displacement)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


# ----------------------------------------------------------------------
# ADX / DMI (Trend strength)
# ----------------------------------------------------------------------
@cached()
def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0-100)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    from utils.indicators.volatility import atr as _atr
    tr = _atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


@cached()
def dmi(df: pd.DataFrame, period: int = 14) -> tuple:
    """Directional Movement Index — returns (+DI, -DI, ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    from utils.indicators.volatility import atr as _atr
    tr = _atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx_val


@cached()
def slope(series: pd.Series, period: int = 5) -> pd.Series:
    """Slope of a series over N bars (rate of change)."""
    return series.diff(period) / period


@cached()
def highest(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling highest (max)."""
    return series.rolling(period).max()


@cached()
def lowest(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling lowest (min)."""
    return series.rolling(period).min()


@cached()
def trend_score(close: pd.Series, period: int = 20) -> pd.Series:
    """Trend Score: composite of EMA stacking + slope.

    Returns [-1, 1] where:
        +1 = strong uptrend
        -1 = strong downtrend
         0 = no trend
    """
    ema_fast = ema(close, max(period // 2, 5))
    ema_slow = ema(close, period)
    ema_slower = ema(close, period * 2)
    bull_stack = (ema_fast > ema_slow) & (ema_slow > ema_slower)
    bear_stack = (ema_fast < ema_slow) & (ema_slow < ema_slower)
    slope_val = (ema_slow.diff(5) / ema_slow.shift(5).replace(0, np.nan)).fillna(0)
    score = pd.Series(0.0, index=close.index)
    score[bull_stack] = np.minimum(slope_val[bull_stack] * 100, 1.0)
    score[bear_stack] = np.maximum(slope_val[bear_stack] * 100, -1.0)
    return score
