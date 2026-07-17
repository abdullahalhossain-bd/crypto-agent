"""engine.data_feed
=====================================================================
Day 1 — Market Data Layer.

Responsibilities:
  - Translate "give me N candles of BTCUSD M15" into a clean pandas
    DataFrame with a stable schema:
        time | open | high | low | close | volume | (tick_volume)
  - Cache last fetch per symbol so we don't hammer MT5 on every loop
  - Stream-safe: detect new bars vs. in-progress bar
  - Degrade gracefully to synthetic data when MT5 is unavailable
    (Linux/CI), so the paper-trading loop is fully demoable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from brokers.mt5_connector import MT5Connector
from utils.logger import get_logger

log = get_logger("engine.data_feed")

# Canonical column order produced by every fetch
CANONICAL_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    last_bar_time: Optional[pd.Timestamp] = None
    fetched_at: float = 0.0


class DataFeed:
    """OHLCV retrieval wrapper around an `MT5Connector`.

    If `connector` is None (MT5 unavailable), `synthetic_fallback` is
    invoked with (symbol, timeframe, count) and must return a pandas
    DataFrame in the canonical schema. This keeps the paper-trading
    loop runnable on Linux/CI for demos and integration tests.
    """

    def __init__(
        self,
        connector: Optional[MT5Connector],
        lookback_candles: int = 1000,
        cache_ttl_s: float = 1.0,
        synthetic_fallback: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
    ) -> None:
        self.conn = connector
        self.lookback = int(lookback_candles)
        self.cache_ttl_s = float(cache_ttl_s)
        self.synthetic_fallback = synthetic_fallback
        self._cache: dict[str, _CacheEntry] = {}

    # ----------------------------------------------------------------
    # Cache helpers
    # ----------------------------------------------------------------
    def _cache_fresh(self, key: str) -> bool:
        """True iff we have a non-stale cache entry for `key`.

        M10 fix: use time.monotonic() instead of time.time() so cache
        freshness isn't affected by system clock jumps (NTP adjustments,
        DST changes, manual operator changes). The fetched_at timestamp
        is now stored in monotonic time.
        """
        entry = self._cache.get(key)
        if entry is None or entry.df.empty:
            return False
        return (time.monotonic() - entry.fetched_at) < self.cache_ttl_s

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        count: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch up to `count` (default self.lookback) OHLCV bars."""
        count = count or self.lookback
        cache_key = f"{symbol}:{timeframe}"

        if use_cache and self._cache_fresh(cache_key):
            return self._cache[cache_key].df.copy()

        # ----- Pick data source: live MT5 or synthetic fallback -----
        if self.conn is not None:
            raw = self.conn.fetch_candles(symbol, timeframe, count, as_dataframe=True)
            df = self._normalize(raw)
        elif self.synthetic_fallback is not None:
            df = self.synthetic_fallback(symbol, timeframe, count)
            # Ensure canonical schema + ascending time
            df = self._normalize_if_needed(df)
        else:
            log.warning("No data source for %s — returning empty", symbol)
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        self._cache[cache_key] = _CacheEntry(
            df=df,
            last_bar_time=df["time"].iloc[-1] if not df.empty else None,
            fetched_at=time.monotonic(),  # M10 fix: monotonic for cache freshness
        )
        log.debug("fetch ok symbol=%s tf=%s bars=%d last=%s",
                  symbol, timeframe, len(df), df["time"].iloc[-1] if not df.empty else "—")
        return df.copy()

    def latest_bar_time(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        entry = self._cache.get(f"{symbol}:{timeframe}")
        return entry.last_bar_time if entry else None

    def has_new_bar(self, symbol: str, timeframe: str) -> bool:
        """Compare cache vs. live tick-time to detect a fresh bar."""
        entry = self._cache.get(f"{symbol}:{timeframe}")
        if entry is None or entry.last_bar_time is None:
            return True
        df = self.fetch(symbol, timeframe, count=2, use_cache=False)
        if df.empty:
            return False
        return df["time"].iloc[-1] > entry.last_bar_time

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _normalize_if_needed(df: pd.DataFrame) -> pd.DataFrame:
        """If df already has canonical columns + datetime time, return as-is;
        otherwise run the MT5-style normalizer on it."""
        if df is None or df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        if (set(CANONICAL_COLUMNS).issubset(df.columns)
                and pd.api.types.is_datetime64_any_dtype(df["time"])
                and df["time"].is_monotonic_increasing):
            return df.reset_index(drop=True)[CANONICAL_COLUMNS]
        return DataFeed._normalize(df)

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        """Coerce MT5 output into canonical schema, drop NaNs, sort asc."""
        if raw is None or raw.empty:
            log.warning("Empty raw rates returned — returning empty DataFrame")
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = raw.copy()
        # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
        rename = {
            "tick_volume": "volume",
            "real_volume": "real_volume",
        }
        df = df.rename(columns=rename)
        if "volume" not in df.columns and "real_volume" in df.columns:
            df["volume"] = df["real_volume"]

        # Keep canonical columns only
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        df = df[keep]

        # Time → tz-aware UTC pandas Timestamp, ascending sort
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True,
                                        errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        # Drop any bar missing ANY of open/high/low/close. A partially-NaN
        # bar (e.g. close missing but open/high/low present, seen during
        # DST gaps and partial broker outages) is just as unusable as a
        # fully-NaN bar — it silently poisons every downstream indicator
        # (SMA/RSI/ATR) if allowed through.
        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"], how="any")
        dropped = before - len(df)
        if dropped:
            log.warning("data_feed: dropped %d bar(s) with partial/missing OHLC "
                       "(%.2f%% of fetch)", dropped, 100.0 * dropped / max(before, 1))
        df = df.reset_index(drop=True)

        # Final schema validation
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(f"data_feed schema violation: missing {missing}")
        return df