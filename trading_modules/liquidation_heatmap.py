"""
Liquidation Heatmap (heuristic)
================================

Crypto markets have visible liquidation cascades — when price moves
sharply, leveraged positions get liquidated, amplifying the move. This
module estimates liquidation zones from price + volume action (since
real liquidation data requires exchange API access).

Heuristic approach:
    1. Identify recent swing highs and lows (where stops cluster)
    2. Estimate leverage zones — common crypto leverage: 10x, 25x, 50x, 100x
    3. For each leverage level, compute the liquidation price for longs
       (below entry) and shorts (above entry)
    4. Cluster liquidation prices → "magnets" that price tends to visit

The heatmap shows where price is likely to spike to trigger cascading
liquidations, then reverse.

Usage:
    from trading_modules.liquidation_heatmap import LiquidationHeatmap
    lh = LiquidationHeatmap()
    result = lh.analyze(df_m15, current_price=65000)
    # result.clusters = list of {price, intensity, side, leverage}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    price: float
    intensity: float           # 0..1, higher = more liquidations expected
    side: str                  # "long_liq" (longs getting liquidated) / "short_liq"
    leverage: int              # 10 / 25 / 50 / 100
    distance_pct: float        # % distance from current price


@dataclass
class LiquidationResult:
    clusters: list[LiquidationCluster] = field(default_factory=list)
    nearest_long_liq: Optional[LiquidationCluster] = None   # below price
    nearest_short_liq: Optional[LiquidationCluster] = None  # above price
    magnet_below: Optional[float] = None                    # strongest long-liq magnet
    magnet_above: Optional[float] = None                    # strongest short-liq magnet
    cascade_risk: str = "low"                                # "low"/"medium"/"high"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clusters": [
                {
                    "price": round(c.price, 2),
                    "intensity": round(c.intensity, 2),
                    "side": c.side,
                    "leverage": c.leverage,
                    "distance_pct": round(c.distance_pct, 2),
                }
                for c in self.clusters
            ],
            "nearest_long_liq": self.nearest_long_liq.to_dict() if self.nearest_long_liq else None,
            "nearest_short_liq": self.nearest_short_liq.to_dict() if self.nearest_short_liq else None,
            "magnet_below": self.magnet_below,
            "magnet_above": self.magnet_above,
            "cascade_risk": self.cascade_risk,
            "notes": self.notes,
        }


class LiquidationHeatmap:
    """Heuristic liquidation zone estimator.

    Parameters:
        leverages: leverage levels to model (default [10, 25, 50, 100])
        swing_window: window for swing detection (default 10)
        cluster_atr_multiple: cluster liquidation prices within this * ATR (default 0.5)
        min_intensity: only show clusters above this intensity (default 0.1)
        atr_period: ATR lookback (default 14)
    """

    def __init__(
        self, leverages: list[int] = None,
        swing_window: int = 10,
        cluster_atr_multiple: float = 0.5,
        min_intensity: float = 0.1,
        atr_period: int = 14,
    ) -> None:
        self.leverages = leverages if leverages is not None else [10, 25, 50, 100]
        self.swing_window = swing_window
        self.cluster_atr_multiple = cluster_atr_multiple
        self.min_intensity = min_intensity
        self.atr_period = atr_period

    def analyze(
        self, df: pd.DataFrame, current_price: float,
    ) -> LiquidationResult:
        if df is None or len(df) < 2 * self.swing_window + 5:
            return LiquidationResult(notes=["insufficient data"])

        atr_series = self._atr(df, self.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr <= 0 or not np.isfinite(atr):
            atr = current_price * 0.01  # fallback: 1% of price

        # Detect recent swing highs and lows
        swing_highs = self._swing_highs(df["high"].to_numpy(dtype=float), self.swing_window)
        swing_lows = self._swing_lows(df["low"].to_numpy(dtype=float), self.swing_window)

        # Compute liquidation prices for each swing + leverage
        # Long liquidation: when price falls by (1/leverage) below the entry price
        #   liq_price = entry × (1 - 1/leverage + maintenance_margin)
        #   Using simplified: liq_price = entry × (1 - 0.9/leverage)
        # Short liquidation: when price rises by (1/leverage) above the entry
        #   liq_price = entry × (1 + 0.9/leverage)
        raw_points: list[tuple[float, float, str, int]] = []
        # (price, intensity, side, leverage)
        for sh in swing_highs:
            for lev in self.leverages:
                # Shorts entered near swing high; their liquidation is above
                liq_price = sh * (1 + 0.9 / lev)
                # Intensity scales by leverage (higher leverage = more liquidations)
                intensity = 0.3 + 0.7 * (lev / 100.0)
                raw_points.append((liq_price, intensity, "short_liq", lev))
        for sl in swing_lows:
            for lev in self.leverages:
                # Longs entered near swing low; their liquidation is below
                liq_price = sl * (1 - 0.9 / lev)
                intensity = 0.3 + 0.7 * (lev / 100.0)
                raw_points.append((liq_price, intensity, "long_liq", lev))

        # Cluster nearby liquidation prices (within cluster_atr_multiple × ATR)
        raw_points.sort(key=lambda p: p[0])
        clusters: list[LiquidationCluster] = []
        current_group: list[tuple[float, float, str, int]] = []
        cluster_tol = atr * self.cluster_atr_multiple

        for point in raw_points:
            if not current_group:
                current_group.append(point)
                continue
            if abs(point[0] - current_group[-1][0]) <= cluster_tol:
                current_group.append(point)
            else:
                if current_group:
                    self._add_cluster(current_group, current_price, clusters)
                current_group = [point]
        if current_group:
            self._add_cluster(current_group, current_price, clusters)

        # Filter by min_intensity
        clusters = [c for c in clusters if c.intensity >= self.min_intensity]
        clusters.sort(key=lambda c: c.intensity, reverse=True)

        # Find nearest long-liq (below price) and short-liq (above price)
        long_liqs = [c for c in clusters if c.side == "long_liq" and c.price < current_price]
        short_liqs = [c for c in clusters if c.side == "short_liq" and c.price > current_price]

        nearest_long = max(long_liqs, key=lambda c: c.price) if long_liqs else None
        nearest_short = min(short_liqs, key=lambda c: c.price) if short_liqs else None
        magnet_below = nearest_long.price if nearest_long else None
        magnet_above = nearest_short.price if nearest_short else None

        # Cascade risk: high if both magnets within 2% of price
        cascade_risk = "low"
        if magnet_below and magnet_above:
            db = abs(current_price - magnet_below) / current_price
            da = abs(magnet_above - current_price) / current_price
            if db < 0.01 and da < 0.01:
                cascade_risk = "high"
            elif db < 0.02 and da < 0.02:
                cascade_risk = "medium"

        notes: list[str] = []
        if magnet_below:
            notes.append(f"magnet below: ${magnet_below:.2f} (long liquidations)")
        if magnet_above:
            notes.append(f"magnet above: ${magnet_above:.2f} (short liquidations)")
        notes.append(f"cascade risk: {cascade_risk}")
        notes.append(f"{len(clusters)} liquidation clusters detected")

        return LiquidationResult(
            clusters=clusters[:20],  # top 20
            nearest_long_liq=nearest_long,
            nearest_short_liq=nearest_short,
            magnet_below=magnet_below,
            magnet_above=magnet_above,
            cascade_risk=cascade_risk,
            notes=notes,
        )

    def _add_cluster(
        self,
        group: list[tuple[float, float, str, int]],
        current_price: float,
        out: list[LiquidationCluster],
    ) -> None:
        """Aggregate a group of raw points into a single LiquidationCluster."""
        if not group:
            return
        avg_price = float(np.mean([p[0] for p in group]))
        total_intensity = float(sum(p[1] for p in group))
        # Side: majority vote
        sides = [p[2] for p in group]
        side = "long_liq" if sides.count("long_liq") >= sides.count("short_liq") else "short_liq"
        # Dominant leverage: max in group
        lev = max(p[3] for p in group)
        distance_pct = abs(avg_price - current_price) / current_price * 100
        out.append(LiquidationCluster(
            price=avg_price,
            intensity=min(1.0, total_intensity),
            side=side,
            leverage=lev,
            distance_pct=float(distance_pct),
        ))

    def _swing_highs(self, highs: np.ndarray, k: int) -> list[float]:
        swings = []
        n = len(highs)
        for i in range(k, n - k):
            window = highs[i - k:i + k + 1]
            if highs[i] == window.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swings.append(float(highs[i]))
        # Last 5 unique swings
        seen: set[float] = set()
        unique: list[float] = []
        for s in reversed(swings):
            if s not in seen:
                seen.add(s); unique.append(s)
            if len(unique) >= 5:
                break
        return unique

    def _swing_lows(self, lows: np.ndarray, k: int) -> list[float]:
        swings = []
        n = len(lows)
        for i in range(k, n - k):
            window = lows[i - k:i + k + 1]
            if lows[i] == window.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swings.append(float(lows[i]))
        seen: set[float] = set()
        unique: list[float] = []
        for s in reversed(swings):
            if s not in seen:
                seen.add(s); unique.append(s)
            if len(unique) >= 5:
                break
        return unique

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()


__all__ = ["LiquidationHeatmap", "LiquidationResult", "LiquidationCluster"]
