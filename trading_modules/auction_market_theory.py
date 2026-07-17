"""
Auction Market Theory (AMT) Analyzer
====================================

Implements institutional auction-market concepts pioneered by J. Peter
Steidlmayer and Jim Dalton (Market Profile):

    1. Initial Balance (IB)  — first hour's high-low range
    2. IB High / IB Low      — reference levels for the day
    3. Value Area Rotation   — does the value area shift up/down/sideways?
    4. Opening Auction       — first 30-min auction behavior
    5. Acceptance vs Rejection — price accepted inside VA or rejected at edge
    6. Single Prints         — lonely prints = trend legs (low revisits)
    7. Excess High/Low       — sharp rejection wicks beyond VA = excess

The analyzer groups bars by trading day (UTC) and computes per-day
auction metrics + a comparison with the previous day's value area.

Usage:
    from trading_modules.auction_market_theory import AMTAnalyzer
    analyzer = AMTAnalyzer()
    amt = analyzer.analyze(df_m15)
    if amt.at_va_high_rejection:
        # price rejected at upper VA edge — short signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AMTResult:
    # Initial Balance
    ib_high: Optional[float] = None
    ib_low: Optional[float] = None
    ib_range: Optional[float] = None
    # Value Area (today)
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    # Value Area Rotation (vs prior day)
    va_rotation: str = "unknown"  # "up" / "down" / "overlap" / "sideways" / "unknown"
    prior_vah: Optional[float] = None
    prior_val: Optional[float] = None
    # Price interaction
    price_at_ib_high: bool = False
    price_at_ib_low: bool = False
    price_in_va: bool = False
    at_va_high_rejection: bool = False
    at_va_low_rejection: bool = False
    # Single prints
    single_print_high: Optional[float] = None
    single_print_low: Optional[float] = None
    # Excess
    excess_high: Optional[float] = None
    excess_low: Optional[float] = None
    # Opening auction
    opening_auction_direction: str = "neutral"  # "up"/"down"/"neutral"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ib_high": self.ib_high, "ib_low": self.ib_low,
            "ib_range": self.ib_range,
            "poc": self.poc, "vah": self.vah, "val": self.val,
            "va_rotation": self.va_rotation,
            "prior_vah": self.prior_vah, "prior_val": self.prior_val,
            "price_at_ib_high": self.price_at_ib_high,
            "price_at_ib_low": self.price_at_ib_low,
            "price_in_va": self.price_in_va,
            "at_va_high_rejection": self.at_va_high_rejection,
            "at_va_low_rejection": self.at_va_low_rejection,
            "single_print_high": self.single_print_high,
            "single_print_low": self.single_print_low,
            "excess_high": self.excess_high,
            "excess_low": self.excess_low,
            "opening_auction_direction": self.opening_auction_direction,
            "notes": self.notes,
        }


class AMTAnalyzer:
    """
    Auction Market Theory analyzer.

    Parameters:
        ib_bars: # of bars in the Initial Balance period (default 4 = 1 hour on M15)
        value_area_pct: target fraction of volume in value area (default 0.70)
        va_rejection_wick_ratio: wick/range threshold for VA-edge rejection (default 0.5)
        single_print_window: bars per price-level for single-print detection (default 1)
        excess_atr_multiple: wick beyond VA by this * ATR = excess (default 0.5)
        atr_period: ATR lookback (default 14)
        session_start_hour_utc: hour the trading day starts (default 0)
    """

    def __init__(
        self, ib_bars: int = 4, value_area_pct: float = 0.70,
        va_rejection_wick_ratio: float = 0.5,
        single_print_window: int = 1,
        excess_atr_multiple: float = 0.5,
        atr_period: int = 14,
        session_start_hour_utc: int = 0,
    ) -> None:
        self.ib_bars = ib_bars
        self.value_area_pct = value_area_pct
        self.va_rejection_wick_ratio = va_rejection_wick_ratio
        self.single_print_window = single_print_window
        self.excess_atr_multiple = excess_atr_multiple
        self.atr_period = atr_period
        self.session_start_hour_utc = session_start_hour_utc

    def analyze(self, df: pd.DataFrame) -> AMTResult:
        if df is None or "time" not in df.columns or len(df) < self.ib_bars + 5:
            return AMTResult(notes=["insufficient data"])

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        if df.empty:
            return AMTResult(notes=["no valid timestamps"])

        # Trading-day assignment
        df["day"] = (df["time"] - pd.Timedelta(hours=self.session_start_hour_utc)).dt.floor("D")

        # Today = last day in df
        today = df["day"].iloc[-1]
        today_df = df[df["day"] == today].reset_index(drop=True)
        if len(today_df) < self.ib_bars:
            return AMTResult(notes=["today has fewer than ib_bars bars"])

        # ── Initial Balance ─────────────────────────────────────
        ib = today_df.head(self.ib_bars)
        ib_high = float(ib["high"].max())
        ib_low = float(ib["low"].min())
        ib_range = ib_high - ib_low

        # ── Opening auction direction (first IB bar's body) ─────
        first = today_df.iloc[0]
        if float(first["close"]) > float(first["open"]):
            opening_auction_direction = "up"
        elif float(first["close"]) < float(first["open"]):
            opening_auction_direction = "down"
        else:
            opening_auction_direction = "neutral"

        # ── Volume Profile for today (POC/VAH/VAL) ──────────────
        poc, vah, val = self._compute_value_area(today_df)

        # ── Prior day's VA for rotation analysis ────────────────
        prior_days = df[df["day"] < today]["day"].unique()
        prior_vah = prior_val = None
        if len(prior_days) > 0:
            prior_day = prior_days[-1]
            prior_df = df[df["day"] == prior_day]
            _, prior_vah, prior_val = self._compute_value_area(prior_df)

        # Value Area Rotation
        if prior_vah is None or prior_val is None or vah is None or val is None:
            va_rotation = "unknown"
        elif val > prior_vah:
            va_rotation = "up"          # today's VA entirely above prior
        elif vah < prior_val:
            va_rotation = "down"
        elif vah > prior_vah and val > prior_val:
            va_rotation = "up"          # shifted higher with overlap
        elif vah < prior_vah and val < prior_val:
            va_rotation = "down"
        else:
            va_rotation = "overlap"     # mostly overlapping

        # ── Price interaction ───────────────────────────────────
        last = today_df.iloc[-1]
        last_close = float(last["close"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_open = float(last["open"])
        rng = last_high - last_low

        # ATR for excess / proximity scaling
        atr_series = self._atr(df, self.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr <= 0 or not np.isfinite(atr):
            atr = 1.0

        price_at_ib_high = abs(last_close - ib_high) <= 0.3 * atr
        price_at_ib_low = abs(last_close - ib_low) <= 0.3 * atr
        price_in_va = (
            vah is not None and val is not None and val <= last_close <= vah
        )

        # VA-edge rejections
        at_va_high_rejection = False
        at_va_low_rejection = False
        if rng > 0:
            upper_wick_ratio = (last_high - max(last_open, last_close)) / rng
            lower_wick_ratio = (min(last_open, last_close) - last_low) / rng
            if (vah is not None and abs(last_high - vah) <= 0.3 * atr
                    and last_close < vah and upper_wick_ratio >= self.va_rejection_wick_ratio):
                at_va_high_rejection = True
            if (val is not None and abs(last_low - val) <= 0.3 * atr
                    and last_close > val and lower_wick_ratio >= self.va_rejection_wick_ratio):
                at_va_low_rejection = True

        # ── Single prints (trend legs) ──────────────────────────
        # A price level that appears in only one bar's range over the day
        single_print_high, single_print_low = self._detect_single_prints(today_df)

        # ── Excess high/low ─────────────────────────────────────
        excess_high = None
        excess_low = None
        if vah is not None:
            # Any bar whose high exceeded VAH by >= excess_atr_multiple * ATR
            # AND closed back inside VA
            for _, row in today_df.iterrows():
                h = float(row["high"]); c = float(row["close"])
                if h > vah + self.excess_atr_multiple * atr and c < vah:
                    excess_high = h
                    break
        if val is not None:
            for _, row in today_df.iterrows():
                l = float(row["low"]); c = float(row["close"])
                if l < val - self.excess_atr_multiple * atr and c > val:
                    excess_low = l
                    break

        notes: list[str] = []
        notes.append(f"IB=[{ib_low:.2f}, {ib_high:.2f}] range={ib_range:.2f}")
        if poc is not None:
            notes.append(f"POC={poc:.2f} VA=[{val:.2f}, {vah:.2f}]")
        notes.append(f"VA rotation: {va_rotation}")
        notes.append(f"opening auction: {opening_auction_direction}")
        if at_va_high_rejection:
            notes.append("rejected at VAH — bearish")
        if at_va_low_rejection:
            notes.append("rejected at VAL — bullish")
        if single_print_high is not None:
            notes.append(f"single-print high @ {single_print_high:.2f} (trend leg up)")
        if single_print_low is not None:
            notes.append(f"single-print low @ {single_print_low:.2f} (trend leg down)")
        if excess_high is not None:
            notes.append(f"excess high @ {excess_high:.2f}")
        if excess_low is not None:
            notes.append(f"excess low @ {excess_low:.2f}")

        return AMTResult(
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_range,
            poc=poc, vah=vah, val=val,
            va_rotation=va_rotation,
            prior_vah=prior_vah, prior_val=prior_val,
            price_at_ib_high=price_at_ib_high,
            price_at_ib_low=price_at_ib_low,
            price_in_va=price_in_va,
            at_va_high_rejection=at_va_high_rejection,
            at_va_low_rejection=at_va_low_rejection,
            single_print_high=single_print_high,
            single_print_low=single_print_low,
            excess_high=excess_high,
            excess_low=excess_low,
            opening_auction_direction=opening_auction_direction,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _compute_value_area(
        self, df: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute POC, VAH, VAL via simple volume-by-price histogram."""
        if df.empty or "volume" not in df.columns:
            return None, None, None
        try:
            price_low = float(df["low"].min())
            price_high = float(df["high"].max())
            if price_high <= price_low:
                return None, None, None
            num_bins = 50
            bin_edges = np.linspace(price_low, price_high, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            # Distribute each bar's volume across the bins its [low, high] range overlaps
            bin_lo = bin_edges[:-1]; bin_hi = bin_edges[1:]
            lows = df["low"].to_numpy(dtype=float)
            highs = df["high"].to_numpy(dtype=float)
            vols = df["volume"].to_numpy(dtype=float)
            overlap = np.minimum(highs[:, None], bin_hi[None, :]) - \
                      np.maximum(lows[:, None], bin_lo[None, :])
            overlap = np.clip(overlap, 0.0, None)
            row_sum = overlap.sum(axis=1, keepdims=True)
            row_sum_safe = np.where(row_sum > 0, row_sum, 1.0)
            bin_volumes = (overlap / row_sum_safe * vols[:, None]).sum(axis=0)

            poc_idx = int(np.argmax(bin_volumes))
            poc = float(bin_centers[poc_idx])
            total_vol = float(bin_volumes.sum())
            if total_vol <= 0:
                return poc, None, None
            target = total_vol * self.value_area_pct
            va_bins = {poc_idx}
            acc = float(bin_volumes[poc_idx])
            up = poc_idx + 1; down = poc_idx - 1
            n = len(bin_volumes)
            while acc < target and (up < n or down >= 0):
                up_v = float(bin_volumes[up]) if up < n else -1
                dn_v = float(bin_volumes[down]) if down >= 0 else -1
                if up_v < 0 and dn_v < 0:
                    break
                if up_v >= dn_v and up < n:
                    va_bins.add(up); acc += float(bin_volumes[up]); up += 1
                elif down >= 0:
                    va_bins.add(down); acc += float(bin_volumes[down]); down -= 1
                else:
                    break
            vah = float(bin_edges[max(va_bins) + 1])
            val = float(bin_edges[min(va_bins)])
            return poc, vah, val
        except Exception as e:
            logger.warning(f"AMT value area computation failed: {e}")
            return None, None, None

    def _detect_single_prints(
        self, df: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float]]:
        """Find the highest and lowest single-print levels in the day.

        A price bin is a 'single print' if only one bar's range covers it.
        Returns (highest_single_print, lowest_single_print).
        """
        if df.empty:
            return None, None
        try:
            price_low = float(df["low"].min())
            price_high = float(df["high"].max())
            if price_high <= price_low:
                return None, None
            num_bins = 50
            bin_edges = np.linspace(price_low, price_high, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_lo = bin_edges[:-1]; bin_hi = bin_edges[1:]
            lows = df["low"].to_numpy(dtype=float)
            highs = df["high"].to_numpy(dtype=float)
            overlap = (np.minimum(highs[:, None], bin_hi[None, :]) -
                       np.maximum(lows[:, None], bin_lo[None, :])) > 0
            counts = overlap.sum(axis=0)
            single_mask = counts == 1
            if not single_mask.any():
                return None, None
            single_prices = bin_centers[single_mask]
            return float(single_prices.max()), float(single_prices.min())
        except Exception as e:
            logger.warning(f"AMT single-print detection failed: {e}")
            return None, None

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


__all__ = ["AMTAnalyzer", "AMTResult"]
