"""
Verified Market-Data Snapshot Module — Anti-Confabulation
===========================================================

LLMs confabulate exact numbers. An analyst might cite "RSI at 35.2" or
"Bollinger lower band at $182.50" that the actual data doesn't support.

This module computes a DETERMINISTIC ground-truth snapshot (latest OHLCV
row + common indicators) that analysts are told to treat as the source
of truth for any exact numeric claim.

No LLM involved — pure deterministic computation. The snapshot is injected
into analyst prompts so they reference verified numbers, not hallucinated ones.

Source: TradingAgents v0.3.1 (review #30) — market_data_validator.py
Fixes: Issue #830 — LLM confabulation of exact indicator values

Usage:
    from verified_snapshot import build_verified_snapshot

    snapshot = build_verified_snapshot(df, symbol="BTCUSDT", curr_date="2024-06-15")
    print(snapshot)
    # Outputs: latest OHLCV + EMA10/SMA50/SMA200/RSI/Bollinger/MACD/ATR

    # Inject into LLM prompt
    prompt = f"## Verified Market Data (use these EXACT numbers)\\n{snapshot}\\n\\n## Your Analysis\\nBased on the verified data above, analyze..."
"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Fixed indicator set so the snapshot is the same shape every run
DEFAULT_SNAPSHOT_INDICATORS = (
    "close_10_ema",       # 10-period EMA
    "close_50_sma",       # 50-period SMA
    "close_200_sma",      # 200-period SMA
    "rsi",                # 14-period RSI
    "boll",               # Bollinger middle band
    "boll_ub",            # Bollinger upper band
    "boll_lb",            # Bollinger lower band
    "macd",               # MACD line
    "macds",              # MACD signal line
    "macdh",              # MACD histogram
    "atr",                # 14-period ATR
)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI manually (no external dependency)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute ATR manually."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger bands (middle, upper, lower)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower


def _verified_rows(df: pd.DataFrame, curr_date: Optional[str]) -> pd.DataFrame:
    """
    OHLCV on or before curr_date, date-sorted.

    Re-applies the cutoff defensively — this is a verification path,
    so it must not trust its input to be pre-filtered.
    """
    # Normalize columns
    df = df.copy()
    for col in df.columns:
        lower = col.lower()
        if lower in ('open', 'high', 'low', 'close', 'volume', 'date', 'timestamp'):
            df.rename(columns={col: lower}, inplace=True)

    date_col = 'date' if 'date' in df.columns else 'timestamp' if 'timestamp' in df.columns else None

    if date_col and curr_date:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        cutoff = pd.to_datetime(curr_date)
        df = df[df[date_col] <= cutoff].sort_values(date_col)

    if df.empty:
        raise ValueError("No OHLCV rows available after filtering.")

    return df


def _fmt(value) -> str:
    """Format a value for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_snapshot(
    df: pd.DataFrame,
    symbol: str = "",
    curr_date: Optional[str] = None,
    look_back_days: int = 30,
) -> str:
    """
    Render a ground-truth snapshot for LLM consumption.

    Computes: latest OHLCV row + 11 indicators + recent closes.

    This is INJECTED into analyst prompts so they reference verified
    numbers instead of confabulated ones.

    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol for context
        curr_date: Cutoff date (look-ahead prevention)
        look_back_days: How many recent closes to include

    Returns:
        Text block with verified data for prompt injection
    """
    data = _verified_rows(df, curr_date)

    if len(data) < 50:
        logger.warning(f"Snapshot needs at least 50 bars, got {len(data)}")
        return f"## Verified Snapshot for {symbol}\nInsufficient data ({len(data)} bars)."

    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume'] if 'volume' in data.columns else pd.Series([0] * len(data))

    # Latest OHLCV
    latest = data.iloc[-1]
    date_col = 'date' if 'date' in data.columns else 'timestamp' if 'timestamp' in data.columns else None
    latest_date = str(latest[date_col])[:10] if date_col else "N/A"

    # Compute indicators
    ema10 = close.ewm(span=10, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))
    rsi = compute_rsi(close)
    boll_mid, boll_upper, boll_lower = compute_bollinger(close)
    macd_line, macd_signal, macd_hist = compute_macd(close)
    atr = compute_atr(high, low, close)

    # Recent closes (for trend context)
    recent_n = min(look_back_days, len(close))
    recent_closes = close.iloc[-recent_n:].tolist()

    # Build text block
    lines = [
        f"## Verified Market-Data Snapshot — {symbol}",
        f"**Date**: {latest_date}",
        f"**Bars analyzed**: {len(data)}",
        "",
        "### Latest OHLCV (DO NOT use other numbers — these are verified)",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Open | {_fmt(latest.get('open'))} |",
        f"| High | {_fmt(latest.get('high'))} |",
        f"| Low | {_fmt(latest.get('low'))} |",
        f"| Close | {_fmt(latest.get('close'))} |",
        f"| Volume | {int(latest.get('volume', 0)):,} |",
        "",
        "### Indicators (computed deterministically — cite these exact values)",
        f"| Indicator | Value |",
        f"|-----------|-------|",
        f"| EMA(10) | {_fmt(ema10.iloc[-1])} |",
        f"| SMA(50) | {_fmt(sma50.iloc[-1])} |",
        f"| SMA(200) | {_fmt(sma200.iloc[-1])} |",
        f"| RSI(14) | {_fmt(rsi.iloc[-1])} |",
        f"| Bollinger Mid | {_fmt(boll_mid.iloc[-1])} |",
        f"| Bollinger Upper | {_fmt(boll_upper.iloc[-1])} |",
        f"| Bollinger Lower | {_fmt(boll_lower.iloc[-1])} |",
        f"| MACD Line | {_fmt(macd_line.iloc[-1])} |",
        f"| MACD Signal | {_fmt(macd_signal.iloc[-1])} |",
        f"| MACD Histogram | {_fmt(macd_hist.iloc[-1])} |",
        f"| ATR(14) | {_fmt(atr.iloc[-1])} |",
        "",
        f"### Recent Closes (last {recent_n} bars)",
        f"{' → '.join(f'{c:.2f}' for c in recent_closes[-10:])}",
        "",
        "### Price Position",
        f"- Price vs EMA10: {'Above' if close.iloc[-1] > ema10.iloc[-1] else 'Below'}",
        f"- Price vs SMA50: {'Above' if close.iloc[-1] > sma50.iloc[-1] else 'Below'}",
        f"- RSI Regime: {_rsi_regime(rsi.iloc[-1])}",
        f"- Bollinger %B: {_boll_pct(close.iloc[-1], boll_upper.iloc[-1], boll_lower.iloc[-1])}",
        f"- ATR as % of price: {(atr.iloc[-1] / close.iloc[-1] * 100):.2f}%",
        "",
        "⚠️ **IMPORTANT**: Use ONLY these verified numbers in your analysis.",
        "   Do NOT cite indicator values from memory or estimation.",
    ]

    return "\n".join(lines)


def _rsi_regime(rsi_val: float) -> str:
    """Classify RSI regime."""
    if np.isnan(rsi_val):
        return "N/A"
    if rsi_val >= 70:
        return "Overbought"
    if rsi_val >= 55:
        return "Bullish"
    if rsi_val >= 45:
        return "Neutral"
    if rsi_val >= 30:
        return "Bearish"
    return "Oversold"


def _boll_pct(price: float, upper: float, lower: float) -> str:
    """Compute Bollinger %B position."""
    if np.isnan(upper) or np.isnan(lower) or upper == lower:
        return "N/A"
    pct = (price - lower) / (upper - lower) * 100
    if pct >= 100:
        return f"{pct:.0f}% (above upper band)"
    if pct <= 0:
        return f"{pct:.0f}% (below lower band)"
    return f"{pct:.0f}%"
