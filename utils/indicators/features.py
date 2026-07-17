"""utils/indicators/features.py
=====================================================================
AI Feature Engineering (Improvement #11)
=====================================================================
14 features purpose-built for ML model input:
    - EMA Distance, Price Position, RSI Normalized, ATR%, BB Width,
      Volume Ratio, Momentum Score, Trend Score, Volatility Score,
      Candle Body %, Upper Wick %, Lower Wick %, Daily Range %, Gap %
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.trend import ema, sma
from utils.indicators.momentum import rsi, macd
from utils.indicators.volatility import atr, atr_pct, bollinger_bands


@cached()
def ema_distance(close: pd.Series, period: int = 20) -> pd.Series:
    """Distance from price to EMA, normalized by price.

    0 = price == EMA
    Positive = price above EMA (bullish)
    Negative = price below EMA (bearish)
    """
    return (close - ema(close, period)) / close.replace(0, np.nan) * 100


@cached()
def price_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Where is price within the recent range? 0=bottom, 1=top, 0.5=middle."""
    high = df["high"].rolling(period).max()
    low = df["low"].rolling(period).min()
    return (df["close"] - low) / (high - low).replace(0, np.nan)


@cached()
def rsi_normalized(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI scaled to [-1, 1] (0 = neutral)."""
    return (rsi(close, period) - 50) / 50


@cached()
def atr_percentage(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as % of price (volatility normalized)."""
    return atr_pct(df, period)


@cached()
def bb_width(close: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger Band Width (volatility regime indicator)."""
    _, _, _, w = bollinger_bands(close, period)
    return w


@cached()
def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / rolling avg volume."""
    vol = df.get("volume", pd.Series(1, index=df.index))
    return vol / vol.rolling(period).mean().replace(0, np.nan)


@cached()
def momentum_score(close: pd.Series, period: int = 10) -> pd.Series:
    """Composite momentum: ROC normalized to [-1, 1].

    Uses sigmoid-like scaling so extreme values saturate.
    """
    roc = (close - close.shift(period)) / close.shift(period).replace(0, np.nan)
    return 2 / (1 + np.exp(-roc * 50)) - 1  # sigmoid scaled


@cached()
def trend_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Composite trend score: EMA slope × ADX direction.

    Returns [-1, 1] where:
        +1 = strong uptrend
        -1 = strong downtrend
         0 = no trend
    """
    close = df["close"]
    ema_val = ema(close, period)
    slope = ema_val.diff(5) / ema_val.shift(5).replace(0, np.nan)
    # ADX-like measure: rolling trend strength
    high = df["high"]
    low = df["low"]
    adx_strength = (high - low).rolling(period).mean() / close.replace(0, np.nan)
    direction = np.sign(slope)
    return direction * np.minimum(adx_strength * 10, 1.0)


@cached()
def volatility_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volatility score: ATR percentile rank [0, 1].

    1 = extreme volatility (top of distribution)
    0 = low volatility
    """
    atr_val = atr(df, period)
    return atr_val.rolling(period * 5).rank(pct=True)


@cached()
def candle_body_pct(df: pd.DataFrame) -> pd.Series:
    """Body as % of range (candle strength)."""
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return body / range_


@cached()
def upper_wick_pct(df: pd.DataFrame) -> pd.Series:
    """Upper wick as % of range (rejection from top)."""
    upper = df["high"] - np.maximum(df["open"], df["close"])
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return upper / range_


@cached()
def lower_wick_pct(df: pd.DataFrame) -> pd.Series:
    """Lower wick as % of range (rejection from bottom)."""
    lower = np.minimum(df["open"], df["close"]) - df["low"]
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return lower / range_


@cached()
def daily_range_pct(df: pd.DataFrame) -> pd.Series:
    """High - Low as % of close (intraday range)."""
    return (df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100


@cached()
def gap_pct(df: pd.DataFrame) -> pd.Series:
    """Gap between previous close and current open, as % of close."""
    return (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan) * 100


@cached()
def feature_vector(df: pd.DataFrame, period: int = 20) -> dict:
    """Compute all 14 AI features at once.

    Returns a dict of feature_name -> latest value (scalar).
    For full Series, use the individual functions.
    """
    close = df["close"]
    return {
        "ema_distance": float(ema_distance(close, period).iloc[-1]) if len(close) > 0 else 0.0,
        "price_position": float(price_position(df, period).iloc[-1]) if len(df) > 0 else 0.0,
        "rsi_normalized": float(rsi_normalized(close).iloc[-1]) if len(close) > 0 else 0.0,
        "atr_pct": float(atr_percentage(df).iloc[-1]) if len(df) > 0 else 0.0,
        "bb_width": float(bb_width(close, period).iloc[-1]) if len(close) > 0 else 0.0,
        "volume_ratio": float(volume_ratio(df, period).iloc[-1]) if len(df) > 0 else 1.0,
        "momentum_score": float(momentum_score(close).iloc[-1]) if len(close) > 0 else 0.0,
        "trend_score": float(trend_score(df, period).iloc[-1]) if len(df) > 0 else 0.0,
        "volatility_score": float(volatility_score(df, period).iloc[-1]) if len(df) > 0 else 0.0,
        "candle_body_pct": float(candle_body_pct(df).iloc[-1]) if len(df) > 0 else 0.0,
        "upper_wick_pct": float(upper_wick_pct(df).iloc[-1]) if len(df) > 0 else 0.0,
        "lower_wick_pct": float(lower_wick_pct(df).iloc[-1]) if len(df) > 0 else 0.0,
        "daily_range_pct": float(daily_range_pct(df).iloc[-1]) if len(df) > 0 else 0.0,
        "gap_pct": float(gap_pct(df).iloc[-1]) if len(df) > 0 else 0.0,
    }


@cached()
def confidence_scores(df: pd.DataFrame, period: int = 20) -> dict:
    """Compute per-category confidence scores [0, 1].

    EMA Confidence: EMA stacking + price above/below EMA
    MACD Confidence: histogram direction + strength
    RSI Confidence: position relative to 30/70
    Trend Confidence: ADX-like strength
    Volume Confidence: volume ratio
    """
    close = df["close"]
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)

    # EMA confidence: stacked + price on correct side
    bull_stack = (ema9 > ema21) & (ema21 > ema50) & (close > ema9)
    bear_stack = (ema9 < ema21) & (ema21 < ema50) & (close < ema9)
    ema_conf = pd.Series(0.0, index=df.index)
    ema_conf[bull_stack] = 0.8
    ema_conf[bear_stack] = 0.8
    ema_conf[(ema9 > ema21) & ~bull_stack] = 0.4
    ema_conf[(ema9 < ema21) & ~bear_stack] = 0.4

    # MACD confidence
    macd_line, signal_line, hist = macd(close)
    macd_conf = (np.sign(hist) * np.minimum(np.abs(hist) / (close * 0.001 + 1e-10), 1.0)).abs()

    # RSI confidence
    rsi_val = rsi(close)
    rsi_conf = pd.Series(0.5, index=df.index)
    rsi_conf[rsi_val > 70] = 0.8  # strong bull
    rsi_conf[rsi_val < 30] = 0.8  # strong bear (mean reversion)
    rsi_conf[(rsi_val >= 50) & (rsi_val <= 70)] = 0.6
    rsi_conf[(rsi_val >= 30) & (rsi_val < 50)] = 0.4

    # Trend confidence (simple: rolling directional strength)
    returns = close.pct_change()
    trend_conf = (returns.rolling(period).mean() / returns.rolling(period).std().replace(0, np.nan)).abs()
    trend_conf = trend_conf.fillna(0.5).clip(0, 1)

    # Volume confidence
    vol = df.get("volume", pd.Series(1, index=df.index))
    vol_conf = (vol / vol.rolling(period).mean().replace(0, np.nan)).clip(0, 2) / 2

    return {
        "ema_confidence": float(ema_conf.iloc[-1]) if len(ema_conf) > 0 else 0.0,
        "macd_confidence": float(macd_conf.iloc[-1]) if len(macd_conf) > 0 else 0.0,
        "rsi_confidence": float(rsi_conf.iloc[-1]) if len(rsi_conf) > 0 else 0.0,
        "trend_confidence": float(trend_conf.iloc[-1]) if len(trend_conf) > 0 else 0.0,
        "volume_confidence": float(vol_conf.iloc[-1]) if len(vol_conf) > 0 else 0.0,
    }
