"""engine.candlestick.pattern_statistics
=====================================================================
Day 133 — Pattern reliability database.

Tracks historical statistics for every candlestick pattern, per
symbol, per timeframe, per market state. The ML layer can query
this DB to learn which patterns actually work in which contexts.

Schema (per row):
  - pattern_type
  - symbol
  - timeframe
  - market_state (TREND/RANGE/CHOPPY)
  - direction (bullish/bearish)
  - n_trades
  - win_rate
  - avg_pnl_pct
  - avg_rr (risk:reward)
  - avg_hold_bars
  - max_drawdown_pct
  - last_updated

The DB is persisted to JSON so it accumulates over time.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("candlestick.stats")


@dataclass
class PatternStats:
    pattern: str
    symbol: str
    timeframe: str
    market_state: str
    direction: str
    n_trades: int = 0
    n_wins: int = 0
    avg_pnl_pct: float = 0.0
    avg_rr: float = 0.0
    avg_hold_bars: int = 0
    max_drawdown_pct: float = 0.0
    last_updated: float = 0.0

    @property
    def win_rate(self) -> float:
        return float(self.n_wins / self.n_trades) if self.n_trades > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        return d


# ----------------------------------------------------------------------
class PatternStatisticsDB:
    """Persistent per-pattern statistics database."""

    def __init__(self, path: str = "data/pattern_statistics.json",
                 flush_every: int = 20) -> None:
        self.path = path
        # key: (pattern, symbol, timeframe, market_state, direction)
        self._stats: dict[str, PatternStats] = {}
        # Guards all reads/writes of `_stats`. record_trade() runs on the
        # live decision path and query()/best_patterns_for()/all_stats()
        # can run concurrently from a backtest, dashboard, or analytics
        # thread. Without this lock, a query iterating `_stats.values()`
        # while record_trade() mutates it can raise
        # "RuntimeError: dictionary changed size during iteration".
        self._lock = threading.RLock()
        # Batch disk writes instead of writing synchronously on every
        # single trade. `record_trade()` can sit on the live execution
        # path; blocking on disk I/O per-trade adds unnecessary latency
        # and disk wear under high trade frequency. We flush every
        # `flush_every` recorded trades, and always expose `flush()` for
        # callers (e.g. shutdown hooks) that need a guaranteed write.
        self.flush_every = max(1, int(flush_every))
        self._dirty_count = 0
        self._load()

    # ----------------------------------------------------------------
    @staticmethod
    def _key(pattern: str, symbol: str, timeframe: str,
             market_state: str, direction: str) -> str:
        return f"{pattern}|{symbol}|{timeframe}|{market_state}|{direction}"

    # ----------------------------------------------------------------
    def record_trade(
        self,
        pattern: str,
        symbol: str,
        timeframe: str,
        market_state: str,
        direction: str,
        pnl_pct: float,
        rr: float,
        hold_bars: int,
        drawdown_pct: float,
    ) -> None:
        key = self._key(pattern, symbol, timeframe, market_state, direction)
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = PatternStats(
                    pattern=pattern, symbol=symbol, timeframe=timeframe,
                    market_state=market_state, direction=direction,
                )
                self._stats[key] = s
            s.n_trades += 1
            if pnl_pct > 0:
                s.n_wins += 1
            # Running average
            n = s.n_trades
            s.avg_pnl_pct = ((s.avg_pnl_pct * (n - 1)) + pnl_pct) / n
            s.avg_rr = ((s.avg_rr * (n - 1)) + rr) / n
            s.avg_hold_bars = int(((s.avg_hold_bars * (n - 1)) + hold_bars) / n)
            s.max_drawdown_pct = max(s.max_drawdown_pct, drawdown_pct)
            s.last_updated = time.time()
            self._dirty_count += 1
            should_flush = self._dirty_count >= self.flush_every
        if should_flush:
            self.flush()

    # ----------------------------------------------------------------
    def query(
        self,
        pattern: str,
        symbol: str,
        timeframe: str,
        market_state: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> list[PatternStats]:
        """Return all matching stats. Wildcards = None."""
        out = []
        with self._lock:
            candidates = list(self._stats.values())
        for s in candidates:
            if s.pattern != pattern or s.symbol != symbol or s.timeframe != timeframe:
                continue
            if market_state is not None and s.market_state != market_state:
                continue
            if direction is not None and s.direction != direction:
                continue
            out.append(s)
        return out

    def best_patterns_for(
        self,
        symbol: str,
        timeframe: str,
        market_state: str,
        min_trades: int = 10,
        min_win_rate: float = 0.50,
    ) -> list[PatternStats]:
        """Return patterns with good historical performance in this context."""
        out = []
        with self._lock:
            candidates = list(self._stats.values())
        for s in candidates:
            if (s.symbol == symbol and s.timeframe == timeframe
                    and s.market_state == market_state
                    and s.n_trades >= min_trades
                    and s.win_rate >= min_win_rate):
                out.append(s)
        out.sort(key=lambda s: (s.win_rate, s.n_trades), reverse=True)
        return out

    # ----------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        with self._lock:
            stats_snapshot = list(self._stats.values())
        return {
            "n_records": len(stats_snapshot),
            "total_trades": sum(s.n_trades for s in stats_snapshot),
            "patterns_tracked": len(set(s.pattern for s in stats_snapshot)),
        }

    def all_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            stats_snapshot = list(self._stats.values())
        return [s.to_dict() for s in stats_snapshot]

    # ----------------------------------------------------------------
    def flush(self) -> None:
        """Write current stats to disk synchronously. Safe to call
        explicitly (e.g. on shutdown) to guarantee no buffered trades
        are lost between batched auto-flushes."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        try:
            with self._lock:
                payload = [s.to_dict() for s in self._stats.values()]
                self._dirty_count = 0
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            log.warning("pattern stats save failed: %r", e)

    # Backward-compatible alias for the old (misleadingly-named,
    # actually-synchronous) method name.
    def _save_async(self) -> None:
        self.flush()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                key = self._key(d["pattern"], d["symbol"], d["timeframe"],
                                d["market_state"], d["direction"])
                self._stats[key] = PatternStats(
                    pattern=d["pattern"], symbol=d["symbol"],
                    timeframe=d["timeframe"], market_state=d["market_state"],
                    direction=d["direction"],
                    n_trades=d.get("n_trades", 0),
                    n_wins=d.get("n_wins", 0),
                    avg_pnl_pct=d.get("avg_pnl_pct", 0.0),
                    avg_rr=d.get("avg_rr", 0.0),
                    avg_hold_bars=d.get("avg_hold_bars", 0),
                    max_drawdown_pct=d.get("max_drawdown_pct", 0.0),
                    last_updated=d.get("last_updated", 0.0),
                )
            log.info("Pattern stats loaded: %d records", len(self._stats))
        except Exception as e:  # noqa: BLE001
            log.warning("pattern stats load failed: %r", e)