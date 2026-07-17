"""utils/indicators/momentum.py
=====================================================================
Momentum Indicators (13 indicators)
=====================================================================
RSI, Stochastic RSI, Stochastic, MACD, PPO, ROC, Momentum,
TRIX, TSI, CCI, Williams %R, Ultimate Oscillator, Awesome Oscillator
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.trend import ema, sma


@cached()
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder's smoothing."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@cached()
def stoch_rsi(close: pd.Series, rsi_period: int = 14,
              stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI — RSI of RSI, normalized 0-100."""
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    return (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100


@cached()
def stochastic(df: pd.DataFrame, k_period: int = 14,
               d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator — %K and %D lines."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


@cached()
def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD — returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@cached()
def ppo(close: pd.Series, fast: int = 12, slow: int = 26,
        signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Percentage Price Oscillator — MACD in percentage terms."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    ppo_line = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100
    signal_line = ema(ppo_line, signal)
    histogram = ppo_line - signal_line
    return ppo_line, signal_line, histogram


@cached()
def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change — percentage change N bars ago."""
    return (close - close.shift(period)) / close.shift(period).replace(0, np.nan) * 100


@cached()
def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Momentum — difference N bars ago (absolute)."""
    return close - close.shift(period)


@cached()
def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """TRIX — triple-smoothed EMA rate of change."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return e3.pct_change() * 100


@cached()
def tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index — double-smoothed momentum."""
    m = close.diff()
    abs_m = m.abs()
    ema1 = m.ewm(span=long, adjust=False).mean()
    ema2 = ema1.ewm(span=short, adjust=False).mean()
    abs_ema1 = abs_m.ewm(span=long, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False).mean()
    return 100 * (ema2 / abs_ema2.replace(0, np.nan))


@cached()
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


@cached()
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — overbought/oversold (-100 to 0)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


@cached()
def ultimate_oscillator(df: pd.DataFrame, c1: int = 7, c2: int = 14,
                        c3: int = 28) -> pd.Series:
    """Ultimate Oscillator — weighted average of 3 BP/TR ratios."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    bp = close - np.minimum(low, prev_close)
    tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)
    avg1 = bp.rolling(c1).sum() / tr.rolling(c1).sum().replace(0, np.nan)
    avg2 = bp.rolling(c2).sum() / tr.rolling(c2).sum().replace(0, np.nan)
    avg3 = bp.rolling(c3).sum() / tr.rolling(c3).sum().replace(0, np.nan)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


@cached()
def awesome_oscillator(df: pd.DataFrame, fast: int = 5,
                       slow: int = 34) -> pd.Series:
    """Awesome Oscillator — SMA midpoint difference."""
    mid = (df["high"] + df["low"]) / 2
    return sma(mid, fast) - sma(mid, slow)
