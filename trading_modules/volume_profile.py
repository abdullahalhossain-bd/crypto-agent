"""
Volume Profile Analyzer — "Where has the market actually done business?"
======================================================================

Distributes traded volume across *price levels* rather than across *time*.
Reveals where institutional participants actually transacted.

Computes:
    1. POC (Point of Control)  — highest-volume price level
    2. VAH (Value Area High)   — upper bound of the 70% value area
    3. VAL (Value Area Low)    — lower bound of the 70% value area
    4. HVN (High Volume Node)  — institutional interest / S-R magnets
    5. LVN (Low Volume Node)   — vacuum zones, price moves fast through
    6. Price-VP interaction    — rejection from HVN, fast move through LVN

Usage:
    from trading_modules.volume_profile import VolumeProfileAnalyzer
    analyzer = VolumeProfileAnalyzer(num_bins=100, value_area_pct=0.70)
    vp = analyzer.analyze(df_m15)
    if vp.rejection_from_hvn:
        # fade the move — institutions defended this level
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    hvn_levels: list[float] = field(default_factory=list)
    lvn_levels: list[float] = field(default_factory=list)
    price_at_hvn: bool = False
    price_at_lvn: bool = False
    price_in_value_area: bool = False
    rejection_from_hvn: bool = False
    rejection_from_lvn: bool = False
    fast_move_through_lvn: bool = False
    current_price: Optional[float] = None
    mean_bin_volume: float = 0.0
    total_volume: float = 0.0
    value_area_volume: float = 0.0
    num_bins_used: int = 0

    def to_dict(self) -> dict:
        return {
            "poc": self.poc, "vah": self.vah, "val": self.val,
            "hvn_levels": [round(p, 6) for p in self.hvn_levels],
            "lvn_levels": [round(p, 6) for p in self.lvn_levels],
            "price_at_hvn": self.price_at_hvn,
            "price_at_lvn": self.price_at_lvn,
            "price_in_value_area": self.price_in_value_area,
            "rejection_from_hvn": self.rejection_from_hvn,
            "rejection_from_lvn": self.rejection_from_lvn,
            "fast_move_through_lvn": self.fast_move_through_lvn,
            "current_price": self.current_price,
            "mean_bin_volume": round(self.mean_bin_volume, 6),
            "total_volume": round(self.total_volume, 6),
            "value_area_volume": round(self.value_area_volume, 6),
            "num_bins_used": self.num_bins_used,
        }


class VolumeProfileAnalyzer:
    """Build a Volume Profile over an OHLCV lookback window."""

    STRATEGY_HINTS: dict[str, str] = {
        "rejection_from_hvn":    "Fade / mean-reversion — institutions defended an HVN.",
        "fast_move_through_lvn": "Momentum / breakout continuation through an LVN.",
        "price_in_value_area":   "Range / mean-reversion — market inside value area.",
        "price_at_hvn":          "Caution / fade — heavy institutional interest.",
        "price_at_lvn":          "Breakout continuation — expect fast moves.",
    }

    def __init__(
        self, num_bins: int = 100, value_area_pct: float = 0.70,
        hvn_threshold: float = 1.5, lvn_threshold: float = 0.5,
        min_rows: int = 30, rejection_lookback: int = 3,
        rejection_wick_ratio: float = 0.5, fast_move_body_ratio: float = 0.6,
        price_tolerance_bins: int = 1, binning_method: str = "range",
    ) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if not 0.0 < value_area_pct < 1.0:
            raise ValueError(f"value_area_pct must be in (0,1), got {value_area_pct}")
        if hvn_threshold <= 1.0:
            raise ValueError(f"hvn_threshold must be > 1.0, got {hvn_threshold}")
        if lvn_threshold >= 1.0:
            raise ValueError(f"lvn_threshold must be < 1.0, got {lvn_threshold}")
        if binning_method not in ("range", "close"):
            raise ValueError(f"binning_method must be 'range' or 'close'")

        self.num_bins = int(num_bins)
        self.value_area_pct = float(value_area_pct)
        self.hvn_threshold = float(hvn_threshold)
        self.lvn_threshold = float(lvn_threshold)
        self.min_rows = int(min_rows)
        self.rejection_lookback = int(rejection_lookback)
        self.rejection_wick_ratio = float(rejection_wick_ratio)
        self.fast_move_body_ratio = float(fast_move_body_ratio)
        self.price_tolerance_bins = int(price_tolerance_bins)
        self.binning_method = binning_method

    def analyze(self, df: pd.DataFrame) -> VolumeProfileResult:
        if df is None:
            return VolumeProfileResult()
        required = ("open", "high", "low", "close", "volume")
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("VP analyze: missing columns %s", missing)
            return VolumeProfileResult()

        work = df[list(required)].dropna().copy()
        work = work[work["volume"] >= 0].reset_index(drop=True)
        if len(work) < self.min_rows:
            return VolumeProfileResult()

        high = work["high"].to_numpy(dtype=float)
        low = work["low"].to_numpy(dtype=float)
        close = work["close"].to_numpy(dtype=float)
        vol = work["volume"].to_numpy(dtype=float)

        price_low = float(np.nanmin(low))
        price_high = float(np.nanmax(high))
        if not np.isfinite(price_low) or not np.isfinite(price_high) or price_high <= price_low:
            return VolumeProfileResult()
        total_vol = float(vol.sum())
        if total_vol <= 0.0:
            return VolumeProfileResult()

        bin_edges = np.linspace(price_low, price_high, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = float(bin_edges[1] - bin_edges[0])

        if self.binning_method == "close":
            bin_volumes, _ = np.histogram(close, bins=bin_edges, weights=vol)
            bin_volumes = bin_volumes.astype(float)
        else:
            bin_volumes = self._distribute_volume_by_range(
                low=low, high=high, volume=vol, bin_edges=bin_edges,
            )

        poc_idx = int(np.argmax(bin_volumes))
        poc = float(bin_centers[poc_idx])

        vah, val, va_volume, _ = self._compute_value_area(
            bin_volumes=bin_volumes, poc_idx=poc_idx, bin_edges=bin_edges,
        )

        mean_vol = float(bin_volumes.mean())
        hvn_mask = bin_volumes >= self.hvn_threshold * mean_vol
        lvn_mask = bin_volumes <= self.lvn_threshold * mean_vol
        hvn_levels = [float(c) for c, m in zip(bin_centers, hvn_mask) if m]
        lvn_levels = [float(c) for c, m in zip(bin_centers, lvn_mask) if m]

        current_price = float(close[-1])
        tol = self.price_tolerance_bins * bin_width
        price_at_hvn = any(abs(current_price - lvl) <= tol for lvl in hvn_levels)
        price_at_lvn = any(abs(current_price - lvl) <= tol for lvl in lvn_levels)
        price_in_va = (val is not None and vah is not None and val <= current_price <= vah)

        lookback = max(1, min(self.rejection_lookback, len(work)))
        recent = work.iloc[-lookback:].reset_index(drop=True)
        rej_hvn = self._detect_rejection(recent, hvn_levels, tol)
        rej_lvn = self._detect_rejection(recent, lvn_levels, tol)
        fast_lvn = self._detect_fast_move(recent, lvn_levels)

        return VolumeProfileResult(
            poc=poc, vah=vah, val=val,
            hvn_levels=hvn_levels, lvn_levels=lvn_levels,
            price_at_hvn=price_at_hvn, price_at_lvn=price_at_lvn,
            price_in_value_area=price_in_va,
            rejection_from_hvn=rej_hvn, rejection_from_lvn=rej_lvn,
            fast_move_through_lvn=fast_lvn,
            current_price=current_price, mean_bin_volume=mean_vol,
            total_volume=total_vol, value_area_volume=va_volume,
            num_bins_used=int(self.num_bins),
        )

    @staticmethod
    def _distribute_volume_by_range(
        low: np.ndarray, high: np.ndarray, volume: np.ndarray, bin_edges: np.ndarray,
    ) -> np.ndarray:
        num_bins = len(bin_edges) - 1
        bin_lo = bin_edges[:-1]
        bin_hi = bin_edges[1:]
        overlap = np.minimum(high[:, None], bin_hi[None, :]) - \
                  np.maximum(low[:, None], bin_lo[None, :])
        overlap = np.clip(overlap, 0.0, None)
        candle_range = high - low
        zero_range = candle_range <= 0
        if zero_range.any():
            idx = np.searchsorted(bin_edges, low, side="right") - 1
            idx = np.clip(idx, 0, num_bins - 1)
            rows = np.where(zero_range)[0]
            overlap[rows] = 0.0
            overlap[rows, idx[rows]] = 1.0
        row_sum = overlap.sum(axis=1, keepdims=True)
        row_sum_safe = np.where(row_sum > 0, row_sum, 1.0)
        overlap_frac = overlap / row_sum_safe
        return (overlap_frac * volume[:, None]).sum(axis=0)

    def _compute_value_area(
        self, bin_volumes: np.ndarray, poc_idx: int, bin_edges: np.ndarray,
    ) -> tuple[Optional[float], Optional[float], float, set[int]]:
        num_bins = len(bin_volumes)
        total_vol = float(bin_volumes.sum())
        if total_vol <= 0:
            return None, None, 0.0, set()
        target = total_vol * self.value_area_pct
        va_bins: set[int] = {poc_idx}
        acc = float(bin_volumes[poc_idx])
        up = poc_idx + 1
        down = poc_idx - 1
        while acc < target and (up < num_bins or down >= 0):
            up_vol = float(bin_volumes[up]) if up < num_bins else -np.inf
            down_vol = float(bin_volumes[down]) if down >= 0 else -np.inf
            if up_vol < 0 and down_vol < 0:
                break
            if up_vol >= down_vol and up < num_bins:
                va_bins.add(up); acc += float(bin_volumes[up]); up += 1
            elif down >= 0:
                va_bins.add(down); acc += float(bin_volumes[down]); down -= 1
            else:
                break
        va_min_idx = min(va_bins)
        va_max_idx = max(va_bins)
        vah = float(bin_edges[va_max_idx + 1])
        val = float(bin_edges[va_min_idx])
        return vah, val, acc, va_bins

    def _detect_rejection(
        self, recent: pd.DataFrame, levels: list[float], tol: float,
    ) -> bool:
        if not levels:
            return False
        for _, row in recent.iterrows():
            o, c = float(row["open"]), float(row["close"])
            h, l = float(row["high"]), float(row["low"])
            rng = h - l
            if rng <= 0:
                continue
            upper_wick_ratio = (h - max(o, c)) / rng
            lower_wick_ratio = (min(o, c) - l) / rng
            for lvl in levels:
                if (abs(h - lvl) <= tol and c < lvl
                        and upper_wick_ratio >= self.rejection_wick_ratio):
                    return True
                if (abs(l - lvl) <= tol and c > lvl
                        and lower_wick_ratio >= self.rejection_wick_ratio):
                    return True
        return False

    def _detect_fast_move(
        self, recent: pd.DataFrame, levels: list[float],
    ) -> bool:
        if not levels:
            return False
        levels_arr = np.asarray(levels, dtype=float)
        for _, row in recent.iterrows():
            o, c = float(row["open"]), float(row["close"])
            h, l = float(row["high"]), float(row["low"])
            rng = h - l
            if rng <= 0:
                continue
            body = abs(c - o)
            if body / rng < self.fast_move_body_ratio:
                continue
            crossed = ((o - levels_arr) * (c - levels_arr)) < 0
            if crossed.any():
                return True
        return False


__all__ = ["VolumeProfileAnalyzer", "VolumeProfileResult"]
