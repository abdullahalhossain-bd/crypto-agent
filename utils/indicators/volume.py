"""utils/indicators/volume.py
=====================================================================
Volume Indicators (10 indicators)
=====================================================================
OBV, VWAP, CMF, MFI, Accumulation/Distribution, Ease of Movement,
Volume Oscillator, Force Index, Negative Volume Index, Positive Volume Index
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.trend import ema, sma


@cached()
def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume signed by price direction."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    direction = close.diff().apply(np.sign)
    return (direction * vol).fillna(0).cumsum()


@cached()
def vwap(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """Volume-Weighted Average Price.

    period=None: cumulative VWAP (session)
    period=N: rolling VWAP over N bars
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    typical = (high + low + close) / 3
    pv = typical * vol
    if period is None:
        return pv.cumsum() / vol.cumsum().replace(0, np.nan)
    return pv.rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)


from typing import Optional  # noqa: E402


@cached()
def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — CLV × volume, rolling mean."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_vol = clv * vol
    return mf_vol.rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)


@cached()
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI (0-100)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    tp = (high + low + close) / 3
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


@cached()
def adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line — cumulative CLV × volume."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv * vol).fillna(0).cumsum()


@cached()
def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ease of Movement — price movement per unit volume."""
    high = df["high"]
    low = df["low"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    mid_move = ((high + low) / 2).diff()
    box_ratio = (vol / 1e9) / (high - low).replace(0, np.nan)
    emv = mid_move / box_ratio.replace(0, np.nan)
    return sma(emv, period)


@cached()
def volume_oscillator(df: pd.DataFrame, fast: int = 5,
                      slow: int = 20) -> pd.Series:
    """Volume Oscillator — fast SMA - slow SMA of volume."""
    vol = df.get("volume", pd.Series(0, index=df.index))
    return sma(vol, fast) - sma(vol, slow)


@cached()
def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Force Index — price change × volume, smoothed."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    fi = close.diff() * vol
    return ema(fi, period)


@cached()
def negative_volume_index(df: pd.DataFrame) -> pd.Series:
    """Negative Volume Index — cum close change when volume decreases."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    vol_down = vol < vol.shift(1)
    nvi = pd.Series(0.0, index=df.index)
    cum = 1000.0  # start at 1000
    for i in range(1, len(df)):
        if vol_down.iloc[i]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / max(close.iloc[i - 1], 1e-10)
            cum *= (1 + pct)
        nvi.iloc[i] = cum
    return nvi


@cached()
def positive_volume_index(df: pd.DataFrame) -> pd.Series:
    """Positive Volume Index — cum close change when volume increases."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    vol_up = vol > vol.shift(1)
    pvi = pd.Series(0.0, index=df.index)
    cum = 1000.0
    for i in range(1, len(df)):
        if vol_up.iloc[i]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / max(close.iloc[i - 1], 1e-10)
            cum *= (1 + pct)
        pvi.iloc[i] = cum
    return pvi


@cached()
def pvt(df: pd.DataFrame) -> pd.Series:
    """Price-Volume Trend — cumulative volume × % price change."""
    close = df["close"]
    vol = df.get("volume", pd.Series(1, index=df.index))
    pct = close.pct_change().fillna(0)
    return (pct * vol).cumsum()


@cached()
def rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative Volume — current volume / rolling avg volume."""
    vol = df.get("volume", pd.Series(1, index=df.index))
    avg = vol.rolling(period).mean()
    return vol / avg.replace(0, np.nan)
