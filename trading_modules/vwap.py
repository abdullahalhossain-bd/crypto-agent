"""
VWAP Analyzer — Volume Weighted Average Price for institutional trading
=======================================================================

VWAP is the institutional benchmark. Large funds execute against VWAP to
minimise market impact; traders use VWAP as dynamic support/resistance and
a fairness reference for entries.

Computes:
    1. Standard VWAP — typical_price * volume / total_volume, resets daily
    2. Anchored VWAP — from a specific timestamp (week/month/swing anchor)
    3. VWAP bands — ±1 and ±2 std-dev bands around VWAP
    4. VWAP rejection — last bar wicked into VWAP and closed back opposite
    5. Price-VWAP relationship — above/below + distance in ATR units

Usage:
    from trading_modules.vwap import VWAPAnalyzer
    analyzer = VWAPAnalyzer()
    result = analyzer.analyze(df_15m)
    if result.bullish_rejection:
        # wicked below VWAP, closed back above — long-side signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class VWAPResult:
    vwap: float
    vwap_upper_1: float
    vwap_lower_1: float
    vwap_upper_2: float
    vwap_lower_2: float
    price_above_vwap: bool
    distance_atr: float
    bullish_rejection: bool
    bearish_rejection: bool
    anchored: bool

    def to_dict(self) -> dict:
        def _r(x):
            if x is None:
                return None
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return None
            if np.isnan(xf) or np.isinf(xf):
                return None
            return round(xf, 8)
        return {
            "vwap": _r(self.vwap),
            "vwap_upper_1": _r(self.vwap_upper_1),
            "vwap_lower_1": _r(self.vwap_lower_1),
            "vwap_upper_2": _r(self.vwap_upper_2),
            "vwap_lower_2": _r(self.vwap_lower_2),
            "price_above_vwap": bool(self.price_above_vwap),
            "distance_atr": _r(self.distance_atr),
            "bullish_rejection": bool(self.bullish_rejection),
            "bearish_rejection": bool(self.bearish_rejection),
            "anchored": bool(self.anchored),
        }


class VWAPAnalyzer:
    """Compute standard / anchored VWAP, sigma bands, rejections, price relationship."""

    REQUIRED_COLUMNS: Tuple[str, ...] = (
        "time", "open", "high", "low", "close", "volume",
    )

    def __init__(
        self, use_session_reset: bool = True, session_start_hour_utc: int = 0,
        band_multipliers: Tuple[float, float] = (1.0, 2.0),
        rejection_wick_ratio: float = 0.4, atr_period: int = 14,
    ) -> None:
        if not 0 <= int(session_start_hour_utc) <= 23:
            raise ValueError("session_start_hour_utc must be in [0, 23]")
        m1, m2 = float(band_multipliers[0]), float(band_multipliers[1])
        if m1 > m2:
            raise ValueError("band_multipliers must be ordered (small, large)")
        if not 0.0 <= float(rejection_wick_ratio) <= 1.0:
            raise ValueError("rejection_wick_ratio must be in [0, 1]")

        self.use_session_reset = bool(use_session_reset)
        self.session_start_hour_utc = int(session_start_hour_utc)
        self.band_multipliers = (m1, m2)
        self.rejection_wick_ratio = float(rejection_wick_ratio)
        self.atr_period = int(atr_period)

    def analyze(
        self, df: pd.DataFrame, anchor_time: Optional[pd.Timestamp] = None,
    ) -> VWAPResult:
        anchored = anchor_time is not None

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._empty_result(anchored=anchored)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            log.error("VWAPAnalyzer.analyze: missing columns %s", missing)
            return self._empty_result(anchored=anchored)

        work = df.copy()
        work["time"] = pd.to_datetime(work["time"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = (
            work.dropna(subset=["time", "high", "low", "close", "volume"])
            .sort_values("time").reset_index(drop=True)
        )
        if work.empty:
            return self._empty_result(anchored=anchored)

        work["typical_price"] = (work["high"] + work["low"] + work["close"]) / 3.0
        work["tp_vol"] = work["typical_price"] * work["volume"]
        work["tp2_vol"] = work["typical_price"] ** 2 * work["volume"]

        if anchored:
            vwap_series, std_series = self._anchored_vwap(work, anchor_time)
        elif self.use_session_reset:
            vwap_series, std_series = self._session_vwap(work)
        else:
            vwap_series, std_series = self._continuous_vwap(work)

        m1, m2 = self.band_multipliers
        work["vwap"] = vwap_series
        work["vwap_std"] = std_series
        work["vwap_upper_1"] = vwap_series + m1 * std_series
        work["vwap_lower_1"] = vwap_series - m1 * std_series
        work["vwap_upper_2"] = vwap_series + m2 * std_series
        work["vwap_lower_2"] = vwap_series - m2 * std_series
        work["atr"] = self._atr(work, self.atr_period)

        last = work.iloc[-1]
        close = float(last["close"])
        vwap_last = last["vwap"]
        atr_last = last["atr"]

        price_above = bool(pd.notna(vwap_last) and close > vwap_last)
        if pd.notna(vwap_last) and pd.notna(atr_last) and atr_last > 0:
            distance_atr = float((close - float(vwap_last)) / float(atr_last))
        else:
            distance_atr = float("nan")

        bull_rej, bear_rej = self._detect_rejection(last)

        return VWAPResult(
            vwap=self._f(vwap_last),
            vwap_upper_1=self._f(last["vwap_upper_1"]),
            vwap_lower_1=self._f(last["vwap_lower_1"]),
            vwap_upper_2=self._f(last["vwap_upper_2"]),
            vwap_lower_2=self._f(last["vwap_lower_2"]),
            price_above_vwap=price_above,
            distance_atr=distance_atr,
            bullish_rejection=bool(bull_rej),
            bearish_rejection=bool(bear_rej),
            anchored=bool(anchored),
        )

    def _session_vwap(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        session_id = self._session_id(df["time"])
        grp = df.groupby(session_id, sort=False)
        cum_v = grp["volume"].cumsum()
        cum_pv = grp["tp_vol"].cumsum()
        cum_tp2_v = grp["tp2_vol"].cumsum()
        safe_v = cum_v.replace(0.0, np.nan)
        vwap = cum_pv / safe_v
        var = (cum_tp2_v / safe_v) - vwap ** 2
        var = var.clip(lower=0.0)
        std = np.sqrt(var)
        return vwap, std

    def _continuous_vwap(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        cum_v = df["volume"].cumsum()
        cum_pv = df["tp_vol"].cumsum()
        cum_tp2_v = df["tp2_vol"].cumsum()
        safe_v = cum_v.replace(0.0, np.nan)
        vwap = cum_pv / safe_v
        var = (cum_tp2_v / safe_v) - vwap ** 2
        var = var.clip(lower=0.0)
        std = np.sqrt(var)
        return vwap, std

    def _anchored_vwap(
        self, df: pd.DataFrame, anchor_time: pd.Timestamp,
    ) -> Tuple[pd.Series, pd.Series]:
        anchor_ts = pd.to_datetime(anchor_time, utc=True, errors="coerce")
        nan_series = pd.Series(np.nan, index=df.index, dtype=float)
        if pd.isna(anchor_ts):
            return nan_series, nan_series.copy()
        mask = df["time"] >= anchor_ts
        if not mask.any():
            return nan_series, nan_series.copy()
        sub = df.loc[mask]
        cum_v = sub["volume"].cumsum()
        cum_pv = sub["tp_vol"].cumsum()
        cum_tp2_v = sub["tp2_vol"].cumsum()
        safe_v = cum_v.replace(0.0, np.nan)
        vwap_sub = cum_pv / safe_v
        var_sub = (cum_tp2_v / safe_v) - vwap_sub ** 2
        var_sub = var_sub.clip(lower=0.0)
        std_sub = np.sqrt(var_sub)
        vwap = nan_series.copy()
        std = nan_series.copy()
        vwap.loc[mask] = vwap_sub.values
        std.loc[mask] = std_sub.values
        return vwap, std

    def _session_id(self, times: pd.Series) -> pd.Series:
        shifted = times - pd.Timedelta(hours=self.session_start_hour_utc)
        return shifted.dt.floor("D")

    def _detect_rejection(self, row: pd.Series) -> Tuple[bool, bool]:
        vwap = row["vwap"]
        if pd.isna(vwap):
            return False, False
        o = float(row["open"]); h = float(row["high"])
        l = float(row["low"]); c = float(row["close"])
        rng = h - l
        if rng <= 0:
            return False, False
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        upper_wick_ratio = upper_wick / rng
        lower_wick_ratio = lower_wick / rng
        vwap = float(vwap)
        bull = (l < vwap and c > vwap and lower_wick_ratio >= self.rejection_wick_ratio)
        bear = (h > vwap and c < vwap and upper_wick_ratio >= self.rejection_wick_ratio)
        return bool(bull), bool(bear)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _f(x) -> float:
        if x is None:
            return float("nan")
        try:
            if pd.isna(x):
                return float("nan")
        except (TypeError, ValueError):
            pass
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    def _empty_result(self, anchored: bool) -> VWAPResult:
        nan = float("nan")
        return VWAPResult(
            vwap=nan, vwap_upper_1=nan, vwap_lower_1=nan,
            vwap_upper_2=nan, vwap_lower_2=nan,
            price_above_vwap=False, distance_atr=nan,
            bullish_rejection=False, bearish_rejection=False,
            anchored=bool(anchored),
        )


__all__ = ["VWAPAnalyzer", "VWAPResult"]
