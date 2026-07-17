"""
Fibonacci Retracement + Confluence Analyzer
============================================

Automatic Fibonacci retracement drawing with confluence cluster detection.

Computes:
    1. Auto swing high/low detection (pure numpy, no scipy)
    2. Fib retracement levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    3. Fib extensions: 127.2%, 161.8%, 261.8%
    4. Confluence detection — is current price near a Fib level (ATR-scaled)
    5. Cluster detection — multiple Fib levels collapsing into a tight band

Direction convention:
- BUY  : Fib drawn low(0%)→high(100%). Retracements between (entry zones for
         long, expecting bounce UP). Extensions above (profit targets).
- SELL : Fib drawn high(0%)→low(100%). Retracements between (entry zones for
         short, expecting rejection DOWN). Extensions below (profit targets).

Usage:
    from trading_modules.fibonacci import FibonacciAnalyzer
    analyzer = FibonacciAnalyzer(swing_window=10, atr_period=14)
    result = analyzer.analyze(df_m15, current_price=65200.0, direction="BUY")
    if result.at_nearest_level:
        print(f"Price at Fib {result.nearest_level_name}")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIB_RETRACEMENT_RATIOS: dict[str, float] = {
    "0.0": 0.0, "23.6": 0.236, "38.2": 0.382, "50.0": 0.5,
    "61.8": 0.618, "78.6": 0.786, "100.0": 1.0,
}
FIB_EXTENSION_RATIOS: dict[str, float] = {
    "127.2": 1.272, "161.8": 1.618, "261.8": 2.618,
}


@dataclass
class FibResult:
    swing_high: Optional[float]
    swing_low: Optional[float]
    fib_levels: dict[str, float]
    nearest_level_name: Optional[str]
    nearest_level_price: Optional[float]
    distance_to_nearest: float
    at_nearest_level: bool
    confluence_cluster: Optional[dict]

    def to_dict(self) -> dict[str, Any]:
        dist = self.distance_to_nearest
        dist_out: Optional[float] = None if not math.isfinite(dist) else round(dist, 6)
        return {
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "fib_levels": {k: round(v, 6) for k, v in self.fib_levels.items()},
            "nearest_level_name": self.nearest_level_name,
            "nearest_level_price": self.nearest_level_price,
            "distance_to_nearest": dist_out,
            "at_nearest_level": self.at_nearest_level,
            "confluence_cluster": self.confluence_cluster,
        }


class FibonacciAnalyzer:
    """Automatic Fibonacci retracement + confluence analyzer.

    Parameters:
        swing_window: half-window for swing detection (default 10)
        atr_period: ATR lookback for confluence scaling (default 14)
        confluence_atr_multiple: |distance| <= this * ATR = at level (default 0.3)
        cluster_threshold_atr: adjacent levels within this * ATR = same cluster (default 0.5)
    """

    def __init__(
        self, swing_window: int = 10, atr_period: int = 14,
        confluence_atr_multiple: float = 0.3, cluster_threshold_atr: float = 0.5,
    ) -> None:
        if swing_window < 1:
            raise ValueError(f"swing_window must be >= 1, got {swing_window}")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if confluence_atr_multiple <= 0:
            raise ValueError(f"confluence_atr_multiple must be > 0")
        if cluster_threshold_atr <= 0:
            raise ValueError(f"cluster_threshold_atr must be > 0")
        self.swing_window = swing_window
        self.atr_period = atr_period
        self.confluence_atr_multiple = confluence_atr_multiple
        self.cluster_threshold_atr = cluster_threshold_atr

    def analyze(
        self, df: pd.DataFrame, current_price: float, direction: str = "BUY",
    ) -> FibResult:
        direction = (direction or "BUY").upper()
        if direction not in ("BUY", "SELL"):
            direction = "BUY"
        empty = self._empty_result()
        if df is None or len(df) < 2 * self.swing_window + 1:
            return empty
        df = self._normalize_columns(df)
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return empty

        swing_high = self._detect_last_swing_high(df)
        swing_low = self._detect_last_swing_low(df)
        if swing_high is None or swing_low is None:
            return empty
        sh_price, sh_idx = swing_high
        sl_price, sl_idx = swing_low
        if sh_price <= sl_price:
            return empty

        atr = self._atr(df, self.atr_period)
        if not math.isfinite(atr) or atr <= 0:
            atr = 0.0

        fib_levels = self._compute_levels(
            swing_high=sh_price, swing_low=sl_price, direction=direction,
        )
        nearest_name, nearest_price, distance = self._find_nearest_level(
            levels=fib_levels, current_price=current_price, direction=direction,
        )
        at_nearest = False
        if nearest_price is not None and atr > 0:
            at_nearest = distance <= self.confluence_atr_multiple * atr

        cluster = self._detect_confluence_cluster(
            levels=fib_levels, current_price=current_price, atr=atr,
        )

        return FibResult(
            swing_high=sh_price, swing_low=sl_price,
            fib_levels=fib_levels,
            nearest_level_name=nearest_name,
            nearest_level_price=nearest_price,
            distance_to_nearest=distance,
            at_nearest_level=at_nearest,
            confluence_cluster=cluster,
        )

    @staticmethod
    def _empty_result() -> FibResult:
        return FibResult(
            swing_high=None, swing_low=None, fib_levels={},
            nearest_level_name=None, nearest_level_price=None,
            distance_to_nearest=float("inf"),
            at_nearest_level=False, confluence_cluster=None,
        )

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            lower = str(col).lower()
            if lower in ("open", "high", "low", "close", "volume", "time"):
                rename_map[col] = lower
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        required = ("open", "high", "low", "close")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df

    def _detect_last_swing_high(self, df: pd.DataFrame) -> Optional[tuple[float, int]]:
        highs = df["high"].to_numpy(dtype=float)
        n = len(highs)
        w = self.swing_window
        if n < 2 * w + 1:
            return None
        for i in range(n - 1 - w, w - 1, -1):
            center = highs[i]
            if not np.isfinite(center):
                continue
            left = highs[i - w:i]
            right = highs[i + 1:i + 1 + w]
            if np.all(center > left) and np.all(center > right):
                return float(center), i
        return None

    def _detect_last_swing_low(self, df: pd.DataFrame) -> Optional[tuple[float, int]]:
        lows = df["low"].to_numpy(dtype=float)
        n = len(lows)
        w = self.swing_window
        if n < 2 * w + 1:
            return None
        for i in range(n - 1 - w, w - 1, -1):
            center = lows[i]
            if not np.isfinite(center):
                continue
            left = lows[i - w:i]
            right = lows[i + 1:i + 1 + w]
            if np.all(center < left) and np.all(center < right):
                return float(center), i
        return None

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> float:
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(period, min_periods=1).mean()
        if atr_series.empty:
            return float("nan")
        val = atr_series.iloc[-1]
        return float(val) if pd.notna(val) else float("nan")

    @staticmethod
    def _compute_levels(
        swing_high: float, swing_low: float, direction: str,
    ) -> dict[str, float]:
        rng = swing_high - swing_low
        levels: dict[str, float] = {}
        if direction == "BUY":
            for name, r in FIB_RETRACEMENT_RATIOS.items():
                levels[name] = round(swing_low + r * rng, 6)
            for name, r in FIB_EXTENSION_RATIOS.items():
                levels[name] = round(swing_low + r * rng, 6)
        else:
            for name, r in FIB_RETRACEMENT_RATIOS.items():
                levels[name] = round(swing_high - r * rng, 6)
            for name, r in FIB_EXTENSION_RATIOS.items():
                levels[name] = round(swing_high - r * rng, 6)
        return levels

    @staticmethod
    def _find_nearest_level(
        levels: dict[str, float], current_price: float, direction: str,
    ) -> tuple[Optional[str], Optional[float], float]:
        if not levels:
            return None, None, float("inf")

        def _nearest_in_subset(items: list[tuple[str, float]]):
            best_name, best_price, best_dist = None, None, float("inf")
            for name, price in items:
                d = abs(price - current_price)
                if d < best_dist:
                    best_dist = d; best_name = name; best_price = price
            return best_name, best_price, best_dist

        if direction == "BUY":
            subset = [(n, p) for n, p in levels.items() if p <= current_price]
        else:
            subset = [(n, p) for n, p in levels.items() if p >= current_price]
        if subset:
            return _nearest_in_subset(subset)
        return _nearest_in_subset(list(levels.items()))

    def _detect_confluence_cluster(
        self, levels: dict[str, float], current_price: float, atr: float,
    ) -> Optional[dict]:
        if not levels or atr <= 0:
            return None
        threshold = self.cluster_threshold_atr * atr
        sorted_levels = sorted(levels.items(), key=lambda kv: kv[1])
        clusters: list[list[tuple[str, float]]] = []
        current: list[tuple[str, float]] = [sorted_levels[0]]
        for name, price in sorted_levels[1:]:
            prev_price = current[-1][1]
            if abs(price - prev_price) <= threshold:
                current.append((name, price))
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [(name, price)]
        if len(current) >= 2:
            clusters.append(current)
        if not clusters:
            return None

        def _cluster_distance(cluster):
            low = min(p for _, p in cluster)
            high = max(p for _, p in cluster)
            if low <= current_price <= high:
                return 0.0
            return min(abs(current_price - low), abs(current_price - high))

        best_cluster = min(clusters, key=_cluster_distance)
        prices = [p for _, p in best_cluster]
        return {
            "cluster_low": round(min(prices), 6),
            "cluster_high": round(max(prices), 6),
            "levels_in_cluster": [
                {"name": n, "price": round(p, 6)} for n, p in best_cluster
            ],
            "cluster_size": len(best_cluster),
            "distance_to_cluster": round(_cluster_distance(best_cluster), 6),
        }


__all__ = ["FibonacciAnalyzer", "FibResult"]
