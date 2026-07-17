"""ml.feature_store
=====================================================================
Day 20 — Feature store for the ML filter layer.

Builds per-bar feature vectors from raw OHLCV, suitable for feeding
to a classifier. Features are deliberately interpretable:

  - Lagged returns (1, 3, 5, 10, 20 bars)
  - Realised volatility (10, 20, 50 bar windows)
  - RSI level + RSI slope
  - SMA fast/slow ratio
  - Bollinger position (z-score of close vs mid band)
  - ATR ratio (current ATR vs 50-bar median ATR)
  - Volume z-score
  - Time-of-day + day-of-week (cyclical encoding)
  - Trend strength (linear regression slope of last 20 closes)

All features are computed with NO lookahead — feature[t] only uses
bars up to and including t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.indicators import atr, rsi, sma


# ----------------------------------------------------------------------
@dataclass
class FeatureVector:
    """One row of features for a single bar."""
    timestamp: pd.Timestamp
    features: dict[str, float] = field(default_factory=dict)
    label: Optional[float] = None  # set later by trainer

    def to_row(self) -> dict[str, Any]:
        out = {"timestamp": self.timestamp}
        out.update(self.features)
        if self.label is not None:
            out["label"] = self.label
        return out


# ----------------------------------------------------------------------
class FeatureStore:
    """Computes features from OHLCV DataFrames."""

    def __init__(self,
                 lag_windows: tuple[int, ...] = (1, 3, 5, 10, 20),
                 vol_windows: tuple[int, ...] = (10, 20, 50),
                 rsi_period: int = 14,
                 sma_fast: int = 20,
                 sma_slow: int = 50,
                 atr_period: int = 14,
                 trend_window: int = 20,
                 volume_window: int = 20) -> None:
        self.lag_windows = lag_windows
        self.vol_windows = vol_windows
        self.rsi_period = rsi_period
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.atr_period = atr_period
        self.trend_window = trend_window
        self.volume_window = volume_window

    # ----------------------------------------------------------------
    def build(self, df: pd.DataFrame,
              include_time: bool = True) -> pd.DataFrame:
        """Return a DataFrame of features aligned to `df`'s index.

        Rows where features cannot be computed (warmup) contain NaN.
        """
        if df.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=df.index)

        close = df["close"]
        # ---- Lagged log-returns ----
        log_ret = np.log(close / close.shift(1))
        for w in self.lag_windows:
            # Major #6 fix: removed dead `if False` branch.
            out[f"ret_{w}"] = log_ret.rolling(w).sum()

        # ---- Realised volatility ----
        for w in self.vol_windows:
            out[f"vol_{w}"] = log_ret.rolling(w).std()

        # ---- RSI ----
        r = rsi(close, self.rsi_period)
        out["rsi"] = r
        out["rsi_slope"] = r - r.shift(3)

        # ---- SMA ratio ----
        sma_f = sma(close, self.sma_fast)
        sma_s = sma(close, self.sma_slow)
        out["sma_ratio"] = sma_f / sma_s.replace(0, np.nan)
        out["sma_spread_pct"] = (sma_f - sma_s) / sma_s.replace(0, np.nan)

        # ---- Bollinger position (z-score) ----
        mid = close.rolling(self.sma_fast, min_periods=self.sma_fast).mean()
        std = close.rolling(self.sma_fast, min_periods=self.sma_fast).std()
        out["bb_z"] = (close - mid) / std.replace(0, np.nan)

        # ---- ATR ratio ----
        a = atr(df, self.atr_period)
        baseline_atr = a.rolling(50, min_periods=20).median()
        out["atr_ratio"] = a / baseline_atr.replace(0, np.nan)

        # ---- Volume z-score ----
        if "volume" in df.columns:
            vol = df["volume"]
            vol_mean = vol.rolling(self.volume_window, min_periods=self.volume_window).mean()
            vol_std = vol.rolling(self.volume_window, min_periods=self.volume_window).std()
            out["vol_z"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # ---- Trend strength (slope of linear regression of last N closes) ----
        out["trend_slope"] = close.rolling(self.trend_window).apply(
            self._trend_slope, raw=False
        )

        # ---- Time features (cyclical) ----
        if include_time and "time" in df.columns:
            t = pd.to_datetime(df["time"])
            hour = t.dt.hour + t.dt.minute / 60.0
            out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            dow = t.dt.dayofweek.astype(float)
            out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        return out

    # ----------------------------------------------------------------
    def build_for_bar(self, df: pd.DataFrame) -> FeatureVector:
        """Build features only for the LAST bar of df."""
        full = self.build(df, include_time=True)
        if full.empty:
            return FeatureVector(timestamp=df["time"].iloc[-1]
                                 if not df.empty else pd.Timestamp.utcnow(),
                                 features={})
        last_row = full.iloc[-1].to_dict()
        ts = df["time"].iloc[-1] if "time" in df.columns else pd.Timestamp.utcnow()
        # Convert NaNs to None for downstream safety
        clean = {k: (None if pd.isna(v) else float(v)) for k, v in last_row.items()}
        return FeatureVector(timestamp=ts, features=clean)

    # ----------------------------------------------------------------
    @staticmethod
    def _trend_slope(window: pd.Series) -> float:
        """Slope of linear regression on the window."""
        n = len(window)
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        y = window.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        # slope = cov(x, y) / var(x)
        denom = ((x - x.mean()) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x.mean()) * (y - y.mean())).sum() / denom)


# ----------------------------------------------------------------------
def label_forward_return(df: pd.DataFrame, horizon: int = 5,
                         threshold: float = 0.001) -> pd.Series:
    """Classify each bar as 1 (up), -1 (down), 0 (neutral) based on
    forward return over `horizon` bars.

    The last `horizon` bars have no forward data and are returned as NaN.
    """
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    # NaN where forward data is missing
    label = pd.Series(
        np.where(fwd.isna(), np.nan,
                 np.where(fwd > threshold, 1,
                          np.where(fwd < -threshold, -1, 0))),
        index=df.index, name="label",
    )
    return label
