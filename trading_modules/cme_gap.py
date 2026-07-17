"""
CME Gap Detector — institutional gap analysis for BTC
=====================================================

CME (Chicago Mercantile Exchange) Bitcoin futures close Friday evening
and reopen Sunday evening, creating visible gaps on the CME chart.
These gaps have a ~95% fill rate — price tends to revisit them.

This module detects:
    1. Open CME gaps       — unfilled gaps from prior weekends
    2. Filled gaps         — gaps that have been revisited
    3. Gap direction       — "up gap" (open > prior close) or "down gap"
    4. Gap size            — % distance
    5. Nearest gap magnet  — closest unfilled gap to current price

Note: requires daily (D1) candles with timestamps covering weekend
closures. Works best with CME futures data, but can be approximated
with any daily BTC chart that has weekend gaps.

Usage:
    from trading_modules.cme_gap import CMEGapDetector
    detector = CMEGapDetector()
    result = detector.analyze(df_d1)
    if result.nearest_open_gap:
        print(f"Magnet: {result.nearest_open_gap['price']:.2f} "
              f"({result.nearest_open_gap['direction']})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CMEGapResult:
    open_gaps: list[dict] = field(default_factory=list)
    filled_gaps: list[dict] = field(default_factory=list)
    nearest_open_gap: Optional[dict] = None
    largest_open_gap: Optional[dict] = None
    gap_fill_probability: float = 0.95   # historical fill rate
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "open_gaps": self.open_gaps,
            "filled_gaps_count": len(self.filled_gaps),
            "nearest_open_gap": self.nearest_open_gap,
            "largest_open_gap": self.largest_open_gap,
            "gap_fill_probability": self.gap_fill_probability,
            "notes": self.notes,
        }


class CMEGapDetector:
    """Detect CME futures gaps in daily BTC data.

    Parameters:
        min_gap_pct: minimum gap size as % of price (default 0.5)
        weekend_only: if True, only detect gaps across weekend closures (default True)
    """

    def __init__(
        self, min_gap_pct: float = 0.5, weekend_only: bool = True,
    ) -> None:
        self.min_gap_pct = float(min_gap_pct)
        self.weekend_only = bool(weekend_only)

    def analyze(self, df: pd.DataFrame) -> CMEGapResult:
        if df is None or len(df) < 5 or "time" not in df.columns:
            return CMEGapResult(notes=["insufficient data"])

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        if len(df) < 5:
            return CMEGapResult(notes=["insufficient rows after cleaning"])

        open_gaps: list[dict] = []
        filled_gaps: list[dict] = []

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            prev_close = float(prev["close"])
            curr_open = float(curr["open"])
            prev_time = prev["time"]
            curr_time = curr["time"]

            # If weekend_only, only consider gaps across weekend (Fri → Sun)
            if self.weekend_only:
                # Detect if curr is the first bar after a weekend
                # Weekend: Saturday + Sunday (crypto markets trade 24/7, but
                # CME futures close Friday ~17:00 ET and reopen Sunday ~18:00 ET)
                # We detect this as a gap of >1 calendar day between consecutive D1 bars
                gap_days = (curr_time - prev_time).total_seconds() / 86400
                if gap_days < 2.0:  # less than 2 days → not a weekend gap
                    continue

            # Compute gap
            gap_size_pct = abs(curr_open - prev_close) / prev_close * 100 if prev_close != 0 else 0
            if gap_size_pct < self.min_gap_pct:
                continue

            direction = "up" if curr_open > prev_close else "down"
            gap_low = min(prev_close, curr_open)
            gap_high = max(prev_close, curr_open)

            # Has this gap been filled? Check if any subsequent bar traded
            # back into the [gap_low, gap_high] range
            subsequent = df.iloc[i + 1:]
            filled = False
            fill_time = None
            if not subsequent.empty:
                in_gap = (
                    (subsequent["low"] <= gap_high) &
                    (subsequent["high"] >= gap_low)
                )
                if in_gap.any():
                    filled = True
                    fill_idx = in_gap.idxmax()
                    fill_time = df.loc[fill_idx, "time"]

            gap_dict = {
                "gap_date": curr_time.strftime("%Y-%m-%d"),
                "direction": direction,
                "prev_close": round(prev_close, 2),
                "curr_open": round(curr_open, 2),
                "gap_low": round(gap_low, 2),
                "gap_high": round(gap_high, 2),
                "size_pct": round(gap_size_pct, 2),
                "filled": filled,
                "fill_date": fill_time.strftime("%Y-%m-%d") if fill_time else None,
            }

            if filled:
                filled_gaps.append(gap_dict)
            else:
                open_gaps.append(gap_dict)

        # Find nearest and largest open gaps relative to current price
        last_close = float(df["close"].iloc[-1])
        nearest_open_gap = None
        largest_open_gap = None
        if open_gaps:
            for g in open_gaps:
                g["distance_pct"] = round(
                    abs(last_close - (g["gap_low"] + g["gap_high"]) / 2) / last_close * 100, 2
                )
            nearest_open_gap = min(open_gaps, key=lambda g: g["distance_pct"])
            largest_open_gap = max(open_gaps, key=lambda g: g["size_pct"])

        notes: list[str] = []
        notes.append(f"{len(open_gaps)} open gaps, {len(filled_gaps)} filled gaps")
        if nearest_open_gap:
            notes.append(
                f"nearest open gap: {nearest_open_gap['direction']} gap "
                f"{nearest_open_gap['size_pct']:.2f}% on {nearest_open_gap['gap_date']} "
                f"({nearest_open_gap['distance_pct']:.2f}% away)"
            )
        if largest_open_gap:
            notes.append(
                f"largest open gap: {largest_open_gap['size_pct']:.2f}% on {largest_open_gap['gap_date']}"
            )

        return CMEGapResult(
            open_gaps=open_gaps,
            filled_gaps=filled_gaps,
            nearest_open_gap=nearest_open_gap,
            largest_open_gap=largest_open_gap,
            gap_fill_probability=0.95,
            notes=notes,
        )


__all__ = ["CMEGapDetector", "CMEGapResult"]
