"""
Triple-Barrier Labels — López de Prado Labeling Method
========================================================

The standard "forward returns" labeling is crude — it doesn't capture
the PATH price took. A stock might have ended +5% but first dropped
-10% (hitting your stop) before recovering.

Triple-Barrier Labels solve this by simulating 3 barriers:
  1. Upper barrier: Take-profit level (e.g., +2%)
  2. Lower barrier: Stop-loss level (e.g., -1%)
  3. Vertical barrier: Time limit (e.g., 5 bars)

The label is determined by WHICH barrier was hit FIRST:
  +1 = Upper barrier hit first (profit target)
  -1 = Lower barrier hit first (stop loss)
   0 = Vertical barrier hit first (time expired, no touch)

This produces path-dependent labels that match real trading — your
stop loss would have triggered even if price eventually recovered.

Source: López de Prado (2018) "Advances in Financial Machine Learning" ch.3
        ml4t-3e (review #18)

Usage:
    from triple_barrier import TripleBarrier, compute_labels

    labels = compute_labels(
        df,                    # OHLCV DataFrame
        upper_pct=0.02,       # +2% take-profit
        lower_pct=0.01,       # -1% stop-loss
        max_holding=5,        # 5-bar vertical barrier
    )
    # labels: Series of +1 / -1 / 0 for each bar
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    upper_pct: float = 0.02       # Take-profit as percentage (0.02 = +2%)
    lower_pct: float = 0.01       # Stop-loss as percentage (0.01 = -1%)
    max_holding: int = 5          # Vertical barrier (number of bars)
    use_atr: bool = False         # Use ATR-based dynamic barriers
    atr_period: int = 14          # ATR calculation period
    atr_upper_mult: float = 2.0   # Upper barrier = entry + ATR × mult
    atr_lower_mult: float = 1.0   # Lower barrier = entry - ATR × mult


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def compute_labels(
    df: pd.DataFrame,
    upper_pct: float = 0.02,
    lower_pct: float = 0.01,
    max_holding: int = 5,
    use_atr: bool = False,
    atr_period: int = 14,
    atr_upper_mult: float = 2.0,
    atr_lower_mult: float = 1.0,
) -> pd.Series:
    """
    Compute triple-barrier labels for each bar.

    For each starting bar, simulates forward until one of 3 barriers is hit:
      - Upper: entry × (1 + upper_pct) → label = +1
      - Lower: entry × (1 - lower_pct) → label = -1
      - Vertical: max_holding bars later → label = 0

    Args:
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close' columns
        upper_pct: Take-profit percentage (fixed mode)
        lower_pct: Stop-loss percentage (fixed mode)
        max_holding: Maximum bars to hold (vertical barrier)
        use_atr: If True, use ATR-based dynamic barriers instead of fixed %
        atr_period: ATR lookback period
        atr_upper_mult: Upper barrier = entry + ATR × this
        atr_lower_mult: Lower barrier = entry - ATR × this

    Returns:
        pd.Series of labels (+1, -1, 0) indexed same as df
    """
    n = len(df)
    labels = pd.Series(0, index=df.index, dtype=int)

    # Precompute ATR if needed
    if use_atr:
        atr = compute_atr(df, atr_period)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    for i in range(n - 1):
        entry_price = close[i]

        if use_atr and not np.isnan(atr.iloc[i]):
            upper_barrier = entry_price + atr.iloc[i] * atr_upper_mult
            lower_barrier = entry_price - atr.iloc[i] * atr_lower_mult
        else:
            upper_barrier = entry_price * (1 + upper_pct)
            lower_barrier = entry_price * (1 - lower_pct)

        # Simulate forward
        end_idx = min(i + max_holding, n - 1)
        label = 0  # Default: vertical barrier hit

        for j in range(i + 1, end_idx + 1):
            # Check if high touched upper barrier
            if high[j] >= upper_barrier:
                label = 1
                break
            # Check if low touched lower barrier
            if low[j] <= lower_barrier:
                label = -1
                break

        labels.iloc[i] = label

    # Last bar can't have forward-looking label
    labels.iloc[-1] = 0

    return labels


def compute_meta_labels(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    upper_pct: float = 0.02,
    lower_pct: float = 0.01,
    max_holding: int = 5,
) -> pd.Series:
    """
    Compute meta-labels for secondary model (López de Prado ch.3).

    Meta-labeling separates the DIRECTION decision (primary model)
    from the SIZE/BET decision (secondary model).

    1. Primary model gives direction (buy/sell)
    2. Triple-barrier gives outcome (was the direction correct?)
    3. Meta-label = 1 if direction was correct, 0 if not

    This lets you train a separate model to answer:
    "Given the primary signal says BUY, should I actually bet on it?"

    Args:
        df: OHLCV DataFrame
        primary_signal: Series of +1 (buy), -1 (sell), 0 (no trade)
        upper_pct, lower_pct, max_holding: Barrier parameters

    Returns:
        pd.Series of meta-labels (1 = bet, 0 = don't bet)
    """
    tb_labels = compute_labels(df, upper_pct, lower_pct, max_holding)

    meta_labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df)):
        signal = primary_signal.iloc[i]

        if signal == 0:
            meta_labels.iloc[i] = 0  # No trade → no meta-label
        elif signal == 1:  # Buy signal
            # Correct if upper barrier hit (+1) or no touch (0)
            meta_labels.iloc[i] = 1 if tb_labels.iloc[i] >= 0 else 0
        elif signal == -1:  # Sell signal
            # Correct if lower barrier hit (-1) or no touch (0)
            meta_labels.iloc[i] = 1 if tb_labels.iloc[i] <= 0 else 0

    return meta_labels


def label_distribution(labels: pd.Series) -> dict:
    """Get label distribution statistics."""
    total = len(labels)
    if total == 0:
        return {}

    counts = labels.value_counts().to_dict()
    return {
        "total": total,
        "long_win": counts.get(1, 0),
        "long_win_pct": round(counts.get(1, 0) / total, 4),
        "short_win": counts.get(-1, 0),
        "short_win_pct": round(counts.get(-1, 0) / total, 4),
        "neutral": counts.get(0, 0),
        "neutral_pct": round(counts.get(0, 0) / total, 4),
        "actionable_pct": round((counts.get(1, 0) + counts.get(-1, 0)) / total, 4),
    }
