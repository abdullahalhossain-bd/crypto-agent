"""enhancements.trade_simulator
=====================================================================
Inspired by OpenAlice's simulate module.

The "if I'd entered here, with this exit, what happened" backtest.

A path-dependent walk over dated bars from an entry date to `as_of`
(no lookahead past it), applying ONE built-in exit rule. Answers the
recurring retro question: "buy BTC on the news, does a trailing stop
save me?"

Built-in exit rules:
    trailing_stop(pct)  — exit when close falls `pct` from running peak
    ma_break(period)    — exit on first close below `period`-bar SMA
    stop(pct)           — exit when close falls `pct` below entry
    target(pct)         — exit when close rises `pct` above entry
    hold                — never exit, measure entry → as_of

Reports: entry/exit (date·price·reason), return, MFE/MAE,
peak/trough, and a sampled path for narration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from utils.indicators import sma
from utils.logger import get_logger

log = get_logger("enhancements.trade_simulator")


# Exit rule types
ExitRule = Union[
    dict,  # {"type": "trailing_stop", "pct": 0.05}
    str,   # "hold"
]


@dataclass
class SimulateResult:
    symbol: str
    as_of: str
    rule: dict
    entry: dict[str, Any] = field(default_factory=dict)
    exit: Optional[dict[str, Any]] = None
    open: bool = True
    bars_held: int = 0
    return_pct: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    peak: dict[str, Any] = field(default_factory=dict)
    trough: dict[str, Any] = field(default_factory=dict)
    path: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "rule": self.rule,
            "entry": self.entry,
            "exit": self.exit,
            "open": self.open,
            "bars_held": self.bars_held,
            "return_pct": self.return_pct,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "peak": self.peak,
            "trough": self.trough,
            "path": list(self.path),
            "note": self.note,
        }


# ----------------------------------------------------------------------
class TradeSimulator:
    """Path-dependent what-if backtester."""

    def simulate(
        self,
        df: pd.DataFrame,
        symbol: str,
        entry_date: str,
        exit_rule: ExitRule,
        interval: str = "1d",
        as_of: Optional[str] = None,
        direction: str = "long",
    ) -> SimulateResult:
        """Simulate a trade from entry_date to as_of with the given exit rule.

        Args:
            df: OHLCV dataframe with 'time' column
            symbol: symbol name
            entry_date: YYYY-MM-DD — enter at close of first bar on/after this date
            exit_rule: dict like {"type": "trailing_stop", "pct": 0.05} or "hold"
            interval: bar interval label
            as_of: cutoff date (no lookahead past it). Default: last bar.
            direction: "long" or "short"
        """
        if df.empty or "time" not in df.columns:
            return SimulateResult(symbol=symbol, as_of=as_of or "unknown",
                                    rule=self._parse_rule(exit_rule),
                                    note="empty df or no time column")
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        # Filter to as_of (no lookahead)
        if as_of:
            as_of_dt = pd.to_datetime(as_of, utc=True)
            df = df[df["time"] <= as_of_dt]
        if df.empty:
            return SimulateResult(symbol=symbol, as_of=as_of or "unknown",
                                    rule=self._parse_rule(exit_rule),
                                    note="no bars at or before as_of")
        effective_as_of = str(df["time"].iloc[-1])
        # Find entry bar
        entry_dt = pd.to_datetime(entry_date, utc=True)
        entry_mask = df["time"] >= entry_dt
        if not entry_mask.any():
            return SimulateResult(symbol=symbol, as_of=effective_as_of,
                                    rule=self._parse_rule(exit_rule),
                                    note=f"no bars on/after entry_date {entry_date}")
        entry_idx = int(entry_mask.idxmax())
        entry_bar = df.iloc[entry_idx]
        entry_price = float(entry_bar["close"])
        entry_time = str(entry_bar["time"])
        # Walk forward from entry
        forward = df.iloc[entry_idx:].reset_index(drop=True)
        if len(forward) < 2:
            return SimulateResult(
                symbol=symbol, as_of=effective_as_of,
                rule=self._parse_rule(exit_rule),
                entry={"date": entry_time, "price": entry_price},
                note="not enough bars after entry",
            )
        rule = self._parse_rule(exit_rule)
        # Pre-compute MA if needed
        ma_series: Optional[pd.Series] = None
        if rule["type"] == "ma_break":
            period = rule.get("period", 20)
            ma_series = sma(forward["close"], period)

        # Walk forward
        running_peak = entry_price
        running_trough = entry_price
        mfe = 0.0
        mae = 0.0
        exit_bar: Optional[dict] = None
        path: list[dict] = []
        for i in range(1, len(forward)):
            bar = forward.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            bar_time = str(bar["time"])
            # MFE / MAE (intrabar)
            if direction == "long":
                fav = (high - entry_price) / entry_price
                adv = (entry_price - low) / entry_price
            else:
                fav = (entry_price - low) / entry_price
                adv = (high - entry_price) / entry_price
            mfe = max(mfe, fav)
            mae = max(mae, adv)
            running_peak = max(running_peak, close)
            running_trough = min(running_trough, close)
            # Sample path (≤ 20 points)
            if i % max(1, len(forward) // 20) == 0:
                path.append({"date": bar_time, "close": close})
            # Check exit rule
            exit_reason = self._check_exit(rule, close, entry_price,
                                             running_peak, ma_series, i)
            if exit_reason:
                exit_bar = {
                    "date": bar_time,
                    "price": close,
                    "reason": exit_reason,
                }
                break
        # If no exit triggered
        if exit_bar is None:
            last_bar = forward.iloc[-1]
            exit_price = float(last_bar["close"])
            exit_bar = None
            is_open = True
            ret = (exit_price - entry_price) / entry_price if direction == "long" \
                else (entry_price - exit_price) / entry_price
        else:
            exit_price = exit_bar["price"]
            is_open = False
            ret = (exit_price - entry_price) / entry_price if direction == "long" \
                else (entry_price - exit_price) / entry_price

        peak_idx = int(forward["close"].idxmax())
        trough_idx = int(forward["close"].idxmin())
        return SimulateResult(
            symbol=symbol, as_of=effective_as_of, rule=rule,
            entry={"date": entry_time, "price": entry_price},
            exit=exit_bar, open=is_open,
            bars_held=len(forward) - 1,
            return_pct=float(ret),
            mfe_pct=float(mfe),
            mae_pct=float(mae),
            peak={"date": str(forward.iloc[peak_idx]["time"]),
                   "price": float(forward.iloc[peak_idx]["close"])},
            trough={"date": str(forward.iloc[trough_idx]["time"]),
                     "price": float(forward.iloc[trough_idx]["close"])},
            path=path,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _parse_rule(rule: ExitRule) -> dict:
        if isinstance(rule, str):
            return {"type": rule}
        if isinstance(rule, dict):
            return rule
        return {"type": "hold"}

    # ----------------------------------------------------------------
    @staticmethod
    def _check_exit(rule: dict, close: float, entry: float,
                      peak: float, ma: Optional[pd.Series],
                      i: int) -> Optional[str]:
        """Check if exit rule triggered. Returns reason string or None."""
        rtype = rule.get("type", "hold")
        if rtype == "hold":
            return None
        if rtype == "trailing_stop":
            pct = rule.get("pct", 0.05)
            if peak > 0 and (peak - close) / peak >= pct:
                return f"trailing_stop {pct:.0%}"
        elif rtype == "stop":
            pct = rule.get("pct", 0.05)
            if entry > 0 and (entry - close) / entry >= pct:
                return f"stop {pct:.0%}"
        elif rtype == "target":
            pct = rule.get("pct", 0.10)
            if entry > 0 and (close - entry) / entry >= pct:
                return f"target {pct:.0%}"
        elif rtype == "ma_break":
            if ma is not None and i < len(ma):
                ma_val = ma.iloc[i]
                if not pd.isna(ma_val) and close < float(ma_val):
                    period = rule.get("period", 20)
                    return f"ma_break ({period})"
        return None
