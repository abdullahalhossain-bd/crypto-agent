"""utils/indicators/structure.py
=====================================================================
Market Structure Indicators (Improvement #5)
=====================================================================
Swing High, Swing Low, HH/HL/LH/LL detection, Break of Structure (BoS),
Change of Character (ChoCH), Market Structure Shift (MSS),
Liquidity Sweep Detection
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from utils.indicators.caching import cached


@cached()
def swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Detect swing highs and lows.

    A swing high = bar whose high is highest in [-lookback, +lookback] window.
    A swing low = bar whose low is lowest in same window.

    Returns DataFrame with columns: swing_high, swing_low, swing_high_price, swing_low_price
    """
    high = df["high"]
    low = df["low"]
    n = len(df)
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    sh_price = pd.Series(np.nan, index=df.index, dtype=float)
    sl_price = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(lookback, n - lookback):
        window_high = high.iloc[i - lookback:i + lookback + 1]
        window_low = low.iloc[i - lookback:i + lookback + 1]
        if high.iloc[i] == window_high.max() and high.iloc[i] > window_high.iloc[:lookback].max():
            swing_high.iloc[i] = True
            sh_price.iloc[i] = high.iloc[i]
        if low.iloc[i] == window_low.min() and low.iloc[i] < window_low.iloc[:lookback].min():
            swing_low.iloc[i] = True
            sl_price.iloc[i] = low.iloc[i]

    return pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_high_price": sh_price,
        "swing_low_price": sl_price,
    }, index=df.index)


@cached()
def market_structure(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Detect HH/HL/LH/LL sequence from swing points.

    Returns DataFrame with:
        - structure: 'HH', 'HL', 'LH', 'LL', '' (empty if not a swing)
        - trend: 'up' if HH+HL, 'down' if LH+LL, 'range' otherwise
    """
    swings = swing_highs_lows(df, lookback)
    structure = pd.Series("", index=df.index)
    trend = pd.Series("range", index=df.index)

    prev_sh = np.nan
    prev_sl = np.nan
    current_trend = "range"
    for i in range(len(df)):
        if swings["swing_high"].iloc[i]:
            cur = swings["swing_high_price"].iloc[i]
            if not np.isnan(prev_sh):
                if cur > prev_sh:
                    structure.iloc[i] = "HH"
                    current_trend = "up"
                else:
                    structure.iloc[i] = "LH"
                    current_trend = "down"
            prev_sh = cur
        elif swings["swing_low"].iloc[i]:
            cur = swings["swing_low_price"].iloc[i]
            if not np.isnan(prev_sl):
                if cur > prev_sl:
                    structure.iloc[i] = "HL"
                    current_trend = "up"
                else:
                    structure.iloc[i] = "LL"
                    current_trend = "down"
            prev_sl = cur
        trend.iloc[i] = current_trend

    return pd.DataFrame({
        "structure": structure,
        "trend": trend,
        "swing_high": swings["swing_high"],
        "swing_low": swings["swing_low"],
        "swing_high_price": swings["swing_high_price"],
        "swing_low_price": swings["swing_low_price"],
    }, index=df.index)


@cached()
def break_of_structure(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect Break of Structure (BoS).

    BoS = price breaks above the last swing high (in uptrend continuation)
    or below the last swing low (in downtrend continuation).

    Returns Series of booleans (True on bars where BoS occurred).
    """
    swings = swing_highs_lows(df, lookback)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    bos = pd.Series(False, index=df.index)
    last_sh = np.nan
    last_sl = np.nan
    for i in range(len(df)):
        if swings["swing_high"].iloc[i]:
            last_sh = swings["swing_high_price"].iloc[i]
        if swings["swing_low"].iloc[i]:
            last_sl = swings["swing_low_price"].iloc[i]
        if not np.isnan(last_sh) and close.iloc[i] > last_sh and high.iloc[i] > last_sh:
            bos.iloc[i] = True
        if not np.isnan(last_sl) and close.iloc[i] < last_sl and low.iloc[i] < last_sl:
            bos.iloc[i] = True
    return bos


@cached()
def change_of_character(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect Change of Character (ChoCH).

    ChoCH = first break against the prevailing trend.
    In uptrend: price breaks below last swing low (potential reversal down)
    In downtrend: price breaks above last swing high (potential reversal up)

    Returns Series of booleans.
    """
    ms = market_structure(df, lookback)
    swings = swing_highs_lows(df, lookback)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    choch = pd.Series(False, index=df.index)
    last_sh = np.nan
    last_sl = np.nan
    prev_trend = "range"
    for i in range(len(df)):
        if swings["swing_high"].iloc[i]:
            last_sh = swings["swing_high_price"].iloc[i]
        if swings["swing_low"].iloc[i]:
            last_sl = swings["swing_low_price"].iloc[i]
        cur_trend = ms["trend"].iloc[i]
        # ChoCH: trend changed AND price broke opposite swing
        if cur_trend != prev_trend and prev_trend != "range":
            if cur_trend == "down" and not np.isnan(last_sl) and close.iloc[i] < last_sl:
                choch.iloc[i] = True
            elif cur_trend == "up" and not np.isnan(last_sh) and close.iloc[i] > last_sh:
                choch.iloc[i] = True
        prev_trend = cur_trend
    return choch


@cached()
def market_structure_shift(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect Market Structure Shift (MSS).

    MSS = BoS + ChoCH combined — strong reversal signal.

    Returns Series of booleans.
    """
    bos = break_of_structure(df, lookback)
    choch = change_of_character(df, lookback)
    return bos | choch


@cached()
def liquidity_sweep(df: pd.DataFrame, lookback: int = 20,
                    tolerance_pct: float = 0.001) -> pd.Series:
    """Detect liquidity sweep (stop hunt).

    A liquidity sweep = price briefly pierces a recent swing high/low
    then reverses. This is a classic institutional pattern.

    Returns Series of booleans (True on bars where sweep occurred).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    n = len(df)

    sweeps = pd.Series(False, index=df.index)
    for i in range(lookback, n):
        # Recent swing high/low (excluding current bar)
        window_high = high.iloc[i - lookback:i].max()
        window_low = low.iloc[i - lookback:i].min()
        tolerance = tolerance_pct * close.iloc[i]

        # Bullish sweep: price dips below recent low, then closes back above
        if low.iloc[i] < window_low - tolerance and close.iloc[i] > window_low:
            sweeps.iloc[i] = True
        # Bearish sweep: price pokes above recent high, then closes back below
        if high.iloc[i] > window_high + tolerance and close.iloc[i] < window_high:
            sweeps.iloc[i] = True
    return sweeps


@cached()
def swing_points(df: pd.DataFrame, lookback: int = 5) -> dict:
    """Convenience: return list of swing high/low prices + indices."""
    swings = swing_highs_lows(df, lookback)
    sh_idx = df.index[swings["swing_high"]].tolist()
    sh_prices = swings["swing_high_price"].dropna().tolist()
    sl_idx = df.index[swings["swing_low"]].tolist()
    sl_prices = swings["swing_low_price"].dropna().tolist()
    return {
        "swing_high_indices": sh_idx,
        "swing_high_prices": sh_prices,
        "swing_low_indices": sl_idx,
        "swing_low_prices": sl_prices,
    }
