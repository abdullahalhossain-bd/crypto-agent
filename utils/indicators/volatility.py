"""utils/indicators/volatility.py
=====================================================================
Volatility Indicators (9 indicators)
=====================================================================
ATR, NATR, Bollinger Bands, Bollinger Width, Bollinger %B,
Keltner Channel, Donchian Channel, Chaikin Volatility, StdDev
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.trend import ema, sma


@cached()
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's smoothing."""
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


@cached()
def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as % of close price."""
    return atr(df, period) / df["close"].replace(0, np.nan) * 100


@cached()
def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalized ATR — ATR / close * 100."""
    return atr(df, period) / df["close"].replace(0, np.nan) * 100


@cached()
def bollinger_bands(close: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands — returns (upper, middle, lower, width)."""
    middle = sma(close, period)
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


@cached()
def bollinger_width(close: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band Width (just the width)."""
    _, _, _, w = bollinger_bands(close, period, std_dev)
    return w


@cached()
def bollinger_pct_b(close: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> pd.Series:
    """Bollinger %B — where price sits within the bands (0-1)."""
    upper, middle, lower, _ = bollinger_bands(close, period, std_dev)
    return (close - lower) / (upper - lower).replace(0, np.nan)


@cached()
def keltner_channel(df: pd.DataFrame, period: int = 20,
                    multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel — EMA + ATR bands. Returns (upper, middle, lower)."""
    middle = ema(df["close"], period)
    atr_val = atr(df, period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


@cached()
def donchian_channel(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel — high/low range. Returns (upper, middle, lower)."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


@cached()
def chaikin_volatility(df: pd.DataFrame, period: int = 10,
                       roc_period: int = 10) -> pd.Series:
    """Chaikin Volatility — EMA of high-low range, rate of change."""
    hl = df["high"] - df["low"]
    ema_hl = ema(hl, period)
    return (ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period).replace(0, np.nan) * 100


@cached()
def stddev(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Standard Deviation."""
    return close.rolling(period).std()


@cached()
def historical_volatility(close: pd.Series, period: int = 20,
                          annualize: bool = True) -> pd.Series:
    """Historical volatility (annualized). Default: 252*24*4 for 15min bars."""
    returns = close.pct_change()
    hv = returns.rolling(period).std()
    if annualize:
        hv = hv * np.sqrt(252 * 24 * 4)  # 15min bars
    return hv


@cached()
def parkinson_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Parkinson volatility — uses high-low range, more efficient."""
    ln_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    sq = ln_hl ** 2
    return np.sqrt(sq.rolling(period).mean() / (4 * np.log(2)))
