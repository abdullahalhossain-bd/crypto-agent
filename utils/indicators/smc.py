"""utils/indicators/smc.py
=====================================================================
Smart Money Concepts (SMC) Indicators (Improvement #6)
=====================================================================
Fair Value Gap (FVG), Order Block, Breaker Block, Mitigation Block,
Premium/Discount Zone, Equal Highs/Lows, Liquidity Pool, Imbalance Detection
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from utils.indicators.caching import cached
from utils.indicators.structure import swing_highs_lows


@cached()
def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps (FVG).

    Bullish FVG: bar[i-1].high < bar[i+1].low (gap up)
    Bearish FVG: bar[i-1].low > bar[i+1].high (gap down)

    Returns DataFrame with: bullish_fvg, bearish_fvg, fvg_top, fvg_bottom
    """
    high = df["high"]
    low = df["low"]
    n = len(df)
    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)
    fvg_top = pd.Series(np.nan, index=df.index, dtype=float)
    fvg_bottom = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(1, n - 1):
        # Bullish FVG: gap between bar[i-1].high and bar[i+1].low
        if low.iloc[i + 1] > high.iloc[i - 1]:
            bullish.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i + 1]
            fvg_bottom.iloc[i] = high.iloc[i - 1]
        # Bearish FVG: gap between bar[i-1].low and bar[i+1].high
        if high.iloc[i + 1] < low.iloc[i - 1]:
            bearish.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i - 1]
            fvg_bottom.iloc[i] = high.iloc[i + 1]

    return pd.DataFrame({
        "bullish_fvg": bullish,
        "bearish_fvg": bearish,
        "fvg_top": fvg_top,
        "fvg_bottom": fvg_bottom,
    }, index=df.index)


@cached()
def detect_order_block(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Detect Order Blocks.

    Bullish OB = last down candle before a strong up move
    Bearish OB = last up candle before a strong down move

    Returns DataFrame with: bullish_ob, bearish_ob, ob_high, ob_low
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    n = len(df)

    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)
    ob_high = pd.Series(np.nan, index=df.index, dtype=float)
    ob_low = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(2, n - 1):
        # Bullish OB: bar[i] is down candle, bar[i+1] is strong up
        if close.iloc[i] < open_.iloc[i] and close.iloc[i + 1] > high.iloc[i]:
            # Check for strong move (close above previous 3 highs)
            if i + 1 < n and close.iloc[i + 1] > high.iloc[max(0, i - 2):i].max():
                bullish.iloc[i] = True
                ob_high.iloc[i] = high.iloc[i]
                ob_low.iloc[i] = low.iloc[i]
        # Bearish OB: bar[i] is up candle, bar[i+1] is strong down
        if close.iloc[i] > open_.iloc[i] and close.iloc[i + 1] < low.iloc[i]:
            if i + 1 < n and close.iloc[i + 1] < low.iloc[max(0, i - 2):i].min():
                bearish.iloc[i] = True
                ob_high.iloc[i] = high.iloc[i]
                ob_low.iloc[i] = low.iloc[i]

    return pd.DataFrame({
        "bullish_ob": bullish,
        "bearish_ob": bearish,
        "ob_high": ob_high,
        "ob_low": ob_low,
    }, index=df.index)


@cached()
def detect_breaker_block(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect Breaker Blocks.

    A breaker block = an order block that failed (price broke through it).
    These act as resistance/support on the retest.

    Returns DataFrame with: bullish_breaker, bearish_breaker, breaker_high, breaker_low
    """
    obs = detect_order_block(df, lookback)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    n = len(df)

    bullish_breaker = pd.Series(False, index=df.index)
    bearish_breaker = pd.Series(False, index=df.index)
    bh = pd.Series(np.nan, index=df.index, dtype=float)
    bl = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(1, n):
        # If a bearish OB was invalidated by price breaking above its high
        if obs["bearish_ob"].iloc[i - 1] and close.iloc[i] > obs["ob_high"].iloc[i - 1]:
            bullish_breaker.iloc[i - 1] = True
            bh.iloc[i - 1] = obs["ob_high"].iloc[i - 1]
            bl.iloc[i - 1] = obs["ob_low"].iloc[i - 1]
        # If a bullish OB was invalidated by price breaking below its low
        if obs["bullish_ob"].iloc[i - 1] and close.iloc[i] < obs["ob_low"].iloc[i - 1]:
            bearish_breaker.iloc[i - 1] = True
            bh.iloc[i - 1] = obs["ob_high"].iloc[i - 1]
            bl.iloc[i - 1] = obs["ob_low"].iloc[i - 1]

    return pd.DataFrame({
        "bullish_breaker": bullish_breaker,
        "bearish_breaker": bearish_breaker,
        "breaker_high": bh,
        "breaker_low": bl,
    }, index=df.index)


@cached()
def detect_mitigation_block(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Mitigation Blocks.

    A mitigation block = price returns to an unfilled order block and
    "mitigates" the position. Often a reversal point.

    Returns DataFrame with: mitigation_detected, mitigation_price
    """
    obs = detect_order_block(df)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    n = len(df)

    mitigated = pd.Series(False, index=df.index)
    mit_price = pd.Series(np.nan, index=df.index, dtype=float)

    # Track open (unfilled) OBs
    open_bull_obs = []  # list of (index, high, low)
    open_bear_obs = []
    for i in range(n):
        if obs["bullish_ob"].iloc[i]:
            open_bull_obs.append((i, obs["ob_high"].iloc[i], obs["ob_low"].iloc[i]))
        if obs["bearish_ob"].iloc[i]:
            open_bear_obs.append((i, obs["ob_high"].iloc[i], obs["ob_low"].iloc[i]))
        # Check if price returned to any open bullish OB
        for idx, ob_h, ob_l in open_bull_obs[-5:]:
            if idx < i and low.iloc[i] <= ob_h and close.iloc[i] >= ob_l:
                mitigated.iloc[i] = True
                mit_price.iloc[i] = close.iloc[i]
                break
        # Check bearish OBs
        for idx, ob_h, ob_l in open_bear_obs[-5:]:
            if idx < i and high.iloc[i] >= ob_l and close.iloc[i] <= ob_h:
                mitigated.iloc[i] = True
                mit_price.iloc[i] = close.iloc[i]
                break
    return pd.DataFrame({
        "mitigation_detected": mitigated,
        "mitigation_price": mit_price,
    }, index=df.index)


@cached()
def premium_discount_zone(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Compute premium/discount zones.

    Premium zone = upper 50% of recent range (above 50% equilibrium)
    Discount zone = lower 50% (below equilibrium)

    Returns DataFrame with: equilibrium, premium_zone, discount_zone, zone
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    range_high = high.rolling(lookback).max()
    range_low = low.rolling(lookback).min()
    equilibrium = (range_high + range_low) / 2
    premium = close > equilibrium
    discount = close < equilibrium

    zone = pd.Series("equilibrium", index=df.index)
    zone[premium] = "premium"
    zone[discount] = "discount"

    return pd.DataFrame({
        "equilibrium": equilibrium,
        "range_high": range_high,
        "range_low": range_low,
        "premium_zone": premium,
        "discount_zone": discount,
        "zone": zone,
    }, index=df.index)


@cached()
def detect_equal_highs_lows(df: pd.DataFrame, lookback: int = 20,
                            tolerance_pct: float = 0.001) -> pd.DataFrame:
    """Detect Equal Highs and Equal Lows (liquidity magnets).

    Equal Highs = two swing highs at nearly the same price
    Equal Lows = two swing lows at nearly the same price

    Returns DataFrame with: equal_high, equal_low, eh_price, el_price
    """
    swings = swing_highs_lows(df, lookback=5)
    close = df["close"]
    n = len(df)

    eq_high = pd.Series(False, index=df.index)
    eq_low = pd.Series(False, index=df.index)
    eh_price = pd.Series(np.nan, index=df.index, dtype=float)
    el_price = pd.Series(np.nan, index=df.index, dtype=float)

    # Collect swing prices
    sh_prices = []
    sl_prices = []
    sh_indices = []
    sl_indices = []
    for i in range(n):
        if swings["swing_high"].iloc[i]:
            price = swings["swing_high_price"].iloc[i]
            for prev_p, prev_i in zip(sh_prices[-5:], sh_indices[-5:]):
                if abs(price - prev_p) / max(prev_p, 1e-10) < tolerance_pct:
                    eq_high.iloc[i] = True
                    eh_price.iloc[i] = price
                    break
            sh_prices.append(price)
            sh_indices.append(i)
        if swings["swing_low"].iloc[i]:
            price = swings["swing_low_price"].iloc[i]
            for prev_p, prev_i in zip(sl_prices[-5:], sl_indices[-5:]):
                if abs(price - prev_p) / max(prev_p, 1e-10) < tolerance_pct:
                    eq_low.iloc[i] = True
                    el_price.iloc[i] = price
                    break
            sl_prices.append(price)
            sl_indices.append(i)

    return pd.DataFrame({
        "equal_high": eq_high,
        "equal_low": eq_low,
        "eh_price": eh_price,
        "el_price": el_price,
    }, index=df.index)


@cached()
def detect_liquidity_pool(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Detect liquidity pools (areas of concentrated stop losses).

    Liquidity pool above = recent swing high cluster
    Liquidity pool below = recent swing low cluster

    Returns DataFrame with: liq_above, liq_below, liq_above_price, liq_below_price
    """
    swings = swing_highs_lows(df, lookback=5)
    high = df["high"]
    low = df["low"]

    liq_above = pd.Series(False, index=df.index)
    liq_below = pd.Series(False, index=df.index)
    la_price = pd.Series(np.nan, index=df.index, dtype=float)
    lb_price = pd.Series(np.nan, index=df.index, dtype=float)

    # Rolling max of swing highs in lookback
    sh_prices = swings["swing_high_price"]
    sl_prices = swings["swing_low_price"]
    recent_sh = sh_prices.rolling(lookback).max()
    recent_sl = sl_prices.rolling(lookback).min()

    liq_above = high >= recent_sh * 0.999
    liq_below = low <= recent_sl * 1.001
    la_price = recent_sh
    lb_price = recent_sl

    return pd.DataFrame({
        "liq_above": liq_above,
        "liq_below": liq_below,
        "liq_above_price": la_price,
        "liq_below_price": lb_price,
    }, index=df.index)


@cached()
def detect_imbalance(df: pd.DataFrame) -> pd.Series:
    """Detect imbalance zones (large directional candles).

    Imbalance = bar where body > 70% of range, indicating strong directional flow.

    Returns Series of: 1 (bullish imbalance), -1 (bearish imbalance), 0 (none)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    ratio = body / range_
    imbalance = pd.Series(0, index=df.index)
    imbalance[(ratio > 0.7) & (close > open_)] = 1
    imbalance[(ratio > 0.7) & (close < open_)] = -1
    return imbalance
