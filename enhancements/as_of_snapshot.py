"""enhancements.as_of_snapshot
=====================================================================
Inspired by OpenAlice's snapshot module.

The honest "what did it look like at time T" primitive.

Unlike the quant calculator (latest scalars, dateless), a snapshot
returns:
  - DATED OHLCV bars up to (never past) `as_of` — NO LOOKAHEAD
  - The most-recent ACTUAL bar at/before `as_of` as the "current as of T"
  - Compact technical state (SMA20/50, RSI14, period high/low)
  - Freshness contract surfaced loudly — warns if data is stale

This is the load-bearing read for retrospective / time-machine
analysis. The no-lookahead guarantee is the命门 (lifeline) of an
honest retro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from utils.indicators import rsi, sma
from utils.logger import get_logger

log = get_logger("enhancements.as_of_snapshot")


@dataclass
class SnapshotBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass
class SnapshotResult:
    symbol: str
    interval: str
    as_of: str
    is_latest_actual: bool
    stale_bars: int
    freshness_warning: Optional[str] = None
    latest: Optional[dict[str, Any]] = None
    window_bars: int = 0
    levels: dict[str, Any] = field(default_factory=dict)
    bars: list[SnapshotBar] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "as_of": self.as_of,
            "is_latest_actual": self.is_latest_actual,
            "stale_bars": self.stale_bars,
            "freshness_warning": self.freshness_warning,
            "latest": self.latest,
            "window_bars": self.window_bars,
            "levels": self.levels,
            "bars": [{"date": b.date, "open": b.open, "high": b.high,
                       "low": b.low, "close": b.close, "volume": b.volume}
                      for b in self.bars],
        }


# ----------------------------------------------------------------------
class AsOfSnapshot:
    """Point-in-time market snapshot with no lookahead."""

    def __init__(self, count: int = 90, bars_out: int = 0) -> None:
        self.count = int(count)
        self.bars_out = int(bars_out)

    # ----------------------------------------------------------------
    def take(self, df: pd.DataFrame, symbol: str,
              as_of: Optional[str] = None,
              interval: str = "1d") -> SnapshotResult:
        """Take a snapshot as of `as_of` (YYYY-MM-DD or ISO).

        If as_of is None, uses the last bar in df.
        """
        if df.empty:
            return SnapshotResult(symbol=symbol, interval=interval,
                                    as_of=as_of or "unknown",
                                    is_latest_actual=False, stale_bars=0,
                                    freshness_warning="empty dataframe")
        # Filter: only bars at or before as_of (NO LOOKAHEAD)
        if as_of:
            try:
                as_of_dt = pd.to_datetime(as_of, utc=True)
                df_time = df.copy()
                if "time" in df.columns:
                    df_time["time"] = pd.to_datetime(df_time["time"], utc=True)
                    df_filtered = df_time[df_time["time"] <= as_of_dt].copy()
                else:
                    df_filtered = df_time.copy()
            except Exception as e:  # noqa: BLE001
                log.warning("as_of parse failed: %r — using full df", e)
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()
            as_of = str(df_filtered["time"].iloc[-1]) if "time" in df_filtered.columns else "latest"

        if df_filtered.empty:
            return SnapshotResult(symbol=symbol, interval=interval,
                                    as_of=as_of, is_latest_actual=False,
                                    stale_bars=0,
                                    freshness_warning="no bars at or before as_of")

        # Take the last `count` bars
        window = df_filtered.tail(self.count).reset_index(drop=True)
        # Latest actual bar
        latest_bar = window.iloc[-1]
        # Freshness: compare latest bar time to the original df's latest
        is_latest = True
        stale = 0
        warning = None
        if "time" in df.columns and "time" in window.columns:
            try:
                original_latest = pd.to_datetime(df["time"].iloc[-1], utc=True)
                window_latest = pd.to_datetime(window["time"].iloc[-1], utc=True)
                if window_latest < original_latest:
                    is_latest = False
                    # Count how many bars are stale
                    stale_mask = pd.to_datetime(df["time"], utc=True) > window_latest
                    stale = int(stale_mask.sum())
                    warning = (f"Data is STALE: last bar is {window_latest}, "
                               f"but newer data exists ({stale} bars ahead)")
            except Exception:  # noqa: BLE001
                pass

        # Compute levels (SMA20, SMA50, RSI14, period high/low)
        levels: dict[str, Any] = {}
        if len(window) >= 20:
            sma20 = sma(window["close"], 20)
            levels["sma20"] = float(sma20.iloc[-1]) if not sma20.isna().iloc[-1] else None
        else:
            levels["sma20"] = None
        if len(window) >= 50:
            sma50 = sma(window["close"], 50)
            levels["sma50"] = float(sma50.iloc[-1]) if not sma50.isna().iloc[-1] else None
        else:
            levels["sma50"] = None
        if len(window) >= 15:
            rsi14 = rsi(window["close"], 14)
            levels["rsi14"] = float(rsi14.iloc[-1]) if not rsi14.isna().iloc[-1] else None
        else:
            levels["rsi14"] = None
        levels["period_high"] = float(window["high"].max())
        levels["period_low"] = float(window["low"].min())
        close = float(latest_bar["close"])
        if levels["period_high"] > 0:
            levels["distance_from_high_pct"] = float(
                (close - levels["period_high"]) / levels["period_high"] * 100
            )
        if levels["period_low"] > 0:
            levels["distance_from_low_pct"] = float(
                (close - levels["period_low"]) / levels["period_low"] * 100
            )

        # Latest bar summary
        prev_close = float(window["close"].iloc[-2]) if len(window) > 1 else None
        change_pct = ((close - prev_close) / prev_close * 100) if prev_close else None
        day_amplitude = None
        if prev_close and prev_close > 0:
            day_amplitude = float(
                (float(latest_bar["high"]) - float(latest_bar["low"])) / prev_close * 100
            )
        latest_summary = {
            "date": str(latest_bar.get("time", "")),
            "close": close,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "day_high": float(latest_bar["high"]),
            "day_low": float(latest_bar["low"]),
            "day_amplitude_pct": day_amplitude,
        }

        # Bars out (optional)
        bars_out_list: list[SnapshotBar] = []
        if self.bars_out > 0:
            bars_df = window.tail(self.bars_out)
            for _, row in bars_df.iterrows():
                bars_out_list.append(SnapshotBar(
                    date=str(row.get("time", "")),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)) if "volume" in row else None,
                ))

        return SnapshotResult(
            symbol=symbol, interval=interval,
            as_of=str(as_of),
            is_latest_actual=is_latest,
            stale_bars=stale,
            freshness_warning=warning,
            latest=latest_summary,
            window_bars=len(window),
            levels=levels,
            bars=bars_out_list,
        )
