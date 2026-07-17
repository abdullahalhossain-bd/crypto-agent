"""utils/indicators/candles.py
=====================================================================
Candlestick Pattern Detection (Improvement #7)
=====================================================================
11 patterns: Doji, Hammer, Hanging Man, Shooting Star,
Bullish Engulfing, Bearish Engulfing, Morning Star, Evening Star,
Harami, Three White Soldiers, Three Black Crows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.indicators.caching import cached


def _candle_features(df: pd.DataFrame) -> dict:
    """Compute candle body, shadows, range."""
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    body = (close - open_).abs()
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    range_ = (high - low).replace(0, np.nan)
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "body": body, "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow, "range": range_,
        "is_bull": close > open_,
        "is_bear": close < open_,
    }


@cached()
def detect_doji(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """Doji: open and close nearly equal (body < threshold * range)."""
    f = _candle_features(df)
    return f["body"] < threshold * f["range"]


@cached()
def detect_hammer(df: pd.DataFrame, shadow_ratio: float = 2.0) -> pd.Series:
    """Hammer: small body at top, long lower shadow (bullish reversal)."""
    f = _candle_features(df)
    return (f["lower_shadow"] > shadow_ratio * f["body"]) & \
           (f["upper_shadow"] < f["body"]) & \
           f["is_bull"]


@cached()
def detect_hanging_man(df: pd.DataFrame, shadow_ratio: float = 2.0) -> pd.Series:
    """Hanging Man: same shape as hammer but in uptrend (bearish reversal)."""
    f = _candle_features(df)
    # Need to detect uptrend context — use rolling close slope
    close = df["close"]
    uptrend = close > close.shift(5)
    return (f["lower_shadow"] > shadow_ratio * f["body"]) & \
           (f["upper_shadow"] < f["body"]) & \
           uptrend


@cached()
def detect_shooting_star(df: pd.DataFrame, shadow_ratio: float = 2.0) -> pd.Series:
    """Shooting Star: small body at bottom, long upper shadow (bearish reversal)."""
    f = _candle_features(df)
    return (f["upper_shadow"] > shadow_ratio * f["body"]) & \
           (f["lower_shadow"] < f["body"]) & \
           f["is_bear"]


@cached()
def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing: bearish bar followed by bullish bar that engulfs it."""
    f = _candle_features(df)
    prev_bear = f["is_bear"].shift(1)
    curr_bull = f["is_bull"]
    engulfs = (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    return prev_bear & curr_bull & engulfs


@cached()
def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish Engulfing: bullish bar followed by bearish bar that engulfs it."""
    f = _candle_features(df)
    prev_bull = f["is_bull"].shift(1)
    curr_bear = f["is_bear"]
    engulfs = (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))
    return prev_bull & curr_bear & engulfs


@cached()
def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Morning Star: 3-bar bullish reversal pattern.

    Bar 1: large bearish candle
    Bar 2: small body (indecision) — gaps down
    Bar 3: large bullish candle — closes above midpoint of bar 1
    """
    f = _candle_features(df)
    bar1_bear = f["is_bear"].shift(2) & (f["body"].shift(2) > f["body"].rolling(20).mean().shift(2))
    bar2_small = f["body"].shift(1) < f["range"].shift(1) * 0.3
    bar3_bull = f["is_bull"] & (f["body"] > f["range"] * 0.5)
    closes_above_mid = df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2
    return bar1_bear & bar2_small & bar3_bull & closes_above_mid


@cached()
def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Evening Star: 3-bar bearish reversal pattern (mirror of morning star)."""
    f = _candle_features(df)
    bar1_bull = f["is_bull"].shift(2) & (f["body"].shift(2) > f["body"].rolling(20).mean().shift(2))
    bar2_small = f["body"].shift(1) < f["range"].shift(1) * 0.3
    bar3_bear = f["is_bear"] & (f["body"] > f["range"] * 0.5)
    closes_below_mid = df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2
    return bar1_bull & bar2_small & bar3_bear & closes_below_mid


@cached()
def detect_harami(df: pd.DataFrame) -> pd.DataFrame:
    """Harami: 2-bar reversal pattern. Large bar followed by small bar inside its range.

    Returns DataFrame with: bullish_harami, bearish_harami
    """
    f = _candle_features(df)
    prev_large = f["body"].shift(1) > f["body"] * 2
    inside_range = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    bullish_harami = f["is_bear"].shift(1) & f["is_bull"] & prev_large & inside_range
    bearish_harami = f["is_bull"].shift(1) & f["is_bear"] & prev_large & inside_range
    return pd.DataFrame({
        "bullish_harami": bullish_harami,
        "bearish_harami": bearish_harami,
    }, index=df.index)


@cached()
def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers: 3 consecutive bullish bars, each closing higher."""
    f = _candle_features(df)
    b1 = f["is_bull"].shift(2) & (f["body"].shift(2) > f["range"].shift(2) * 0.5)
    b2 = f["is_bull"].shift(1) & (f["body"].shift(1) > f["range"].shift(1) * 0.5) & \
         (df["close"].shift(1) > df["close"].shift(2))
    b3 = f["is_bull"] & (f["body"] > f["range"] * 0.5) & \
         (df["close"] > df["close"].shift(1))
    return b1 & b2 & b3


@cached()
def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three Black Crows: 3 consecutive bearish bars, each closing lower."""
    f = _candle_features(df)
    b1 = f["is_bear"].shift(2) & (f["body"].shift(2) > f["range"].shift(2) * 0.5)
    b2 = f["is_bear"].shift(1) & (f["body"].shift(1) > f["range"].shift(1) * 0.5) & \
         (df["close"].shift(1) < df["close"].shift(2))
    b3 = f["is_bear"] & (f["body"] > f["range"] * 0.5) & \
         (df["close"] < df["close"].shift(1))
    return b1 & b2 & b3


@cached()
def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all 11 candlestick patterns at once.

    Returns DataFrame with one column per pattern (boolean).
    """
    harami = detect_harami(df)
    return pd.DataFrame({
        "doji": detect_doji(df),
        "hammer": detect_hammer(df),
        "hanging_man": detect_hanging_man(df),
        "shooting_star": detect_shooting_star(df),
        "bullish_engulfing": detect_bullish_engulfing(df),
        "bearish_engulfing": detect_bearish_engulfing(df),
        "morning_star": detect_morning_star(df),
        "evening_star": detect_evening_star(df),
        "bullish_harami": harami["bullish_harami"],
        "bearish_harami": harami["bearish_harami"],
        "three_white_soldiers": detect_three_white_soldiers(df),
        "three_black_crows": detect_three_black_crows(df),
    }, index=df.index)
