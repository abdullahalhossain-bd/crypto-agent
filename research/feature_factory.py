"""research.feature_factory
=====================================================================
Day 41-44 — Feature Factory.

Generates a large catalogue of candidate features from raw OHLCV,
so the hypothesis generator can search for predictive combinations
without manually coding each one.

Feature families:
  - price          : returns, ratios, z-scores
  - volatility     : realised vol across windows, ATR ratios
  - microstructure : volume-weighted features, spread proxies
  - regime         : trend slope, ADX-proxy, autocorrelation
  - cross-asset    : (filled in if multi-symbol df provided)

Every feature is tagged with metadata (family, window, transform)
so the scorer can attribute predictive power back to its source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.indicators import atr, ema, rsi, sma


# ----------------------------------------------------------------------
@dataclass
class FeatureCandidate:
    """One generated feature with metadata."""
    name: str
    family: str          # price | volatility | microstructure | regime | cross_asset
    window: int
    transform: str       # raw | log_return | zscore | ratio | slope
    description: str = ""
    series: Optional[pd.Series] = None  # populated when built

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "window": self.window,
            "transform": self.transform,
            "description": self.description,
        }


# ----------------------------------------------------------------------
class FeatureFactory:
    def __init__(self,
                 price_windows: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
                 vol_windows: tuple[int, ...] = (5, 10, 20, 50),
                 microstructure_windows: tuple[int, ...] = (10, 20, 50),
                 regime_windows: tuple[int, ...] = (20, 50, 100)) -> None:
        self.price_windows = price_windows
        self.vol_windows = vol_windows
        self.microstructure_windows = microstructure_windows
        self.regime_windows = regime_windows

    # ----------------------------------------------------------------
    def generate(self, df: pd.DataFrame,
                 other_closes: Optional[dict[str, pd.Series]] = None
                 ) -> list[FeatureCandidate]:
        """Build the full feature catalogue for `df`."""
        candidates: list[FeatureCandidate] = []
        candidates.extend(self._price_features(df))
        candidates.extend(self._volatility_features(df))
        candidates.extend(self._microstructure_features(df))
        candidates.extend(self._regime_features(df))
        if other_closes:
            candidates.extend(self._cross_asset_features(df, other_closes))
        return candidates

    # ----------------------------------------------------------------
    def _price_features(self, df: pd.DataFrame) -> list[FeatureCandidate]:
        out: list[FeatureCandidate] = []
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        for w in self.price_windows:
            # Rolling return
            s = log_ret.rolling(w).sum()
            out.append(FeatureCandidate(
                name=f"ret_{w}", family="price", window=w,
                transform="log_return",
                description=f"{w}-bar log return",
                series=s,
            ))
            # Z-score of close vs rolling mean
            mean = close.rolling(w).mean()
            std = close.rolling(w).std()
            out.append(FeatureCandidate(
                name=f"close_z_{w}", family="price", window=w,
                transform="zscore",
                description=f"z-score of close vs {w}-bar mean",
                series=(close - mean) / std.replace(0, np.nan),
            ))
            # SMA ratio
            sm = sma(close, w)
            out.append(FeatureCandidate(
                name=f"sma_ratio_{w}", family="price", window=w,
                transform="ratio",
                description=f"close / sma({w})",
                series=close / sm.replace(0, np.nan),
            ))
        # RSI (single shot)
        for period in (7, 14, 21):
            out.append(FeatureCandidate(
                name=f"rsi_{period}", family="price", window=period,
                transform="raw",
                description=f"RSI({period})",
                series=rsi(close, period),
            ))
        return out

    # ----------------------------------------------------------------
    def _volatility_features(self, df: pd.DataFrame) -> list[FeatureCandidate]:
        out: list[FeatureCandidate] = []
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for w in self.vol_windows:
            # Realised vol
            out.append(FeatureCandidate(
                name=f"realised_vol_{w}", family="volatility", window=w,
                transform="raw",
                description=f"{w}-bar realised volatility",
                series=log_ret.rolling(w).std(),
            ))
            # Vol z-score (vs longer baseline)
            baseline = log_ret.rolling(100, min_periods=20).std()
            current = log_ret.rolling(w).std()
            out.append(FeatureCandidate(
                name=f"vol_z_{w}", family="volatility", window=w,
                transform="zscore",
                description=f"vol z-score vs 100-bar baseline",
                series=(current - baseline) / baseline.replace(0, np.nan),
            ))
        # ATR ratios
        a = atr(df, 14)
        a_baseline = a.rolling(50, min_periods=20).median()
        out.append(FeatureCandidate(
            name="atr_ratio_14", family="volatility", window=14,
            transform="ratio",
            description="ATR(14) / 50-bar median ATR",
            series=a / a_baseline.replace(0, np.nan),
        ))
        return out

    # ----------------------------------------------------------------
    def _microstructure_features(self, df: pd.DataFrame) -> list[FeatureCandidate]:
        out: list[FeatureCandidate] = []
        if "volume" not in df.columns:
            return out
        vol = df["volume"]
        close = df["close"]
        for w in self.microstructure_windows:
            # Volume z-score
            vmean = vol.rolling(w).mean()
            vstd = vol.rolling(w).std()
            out.append(FeatureCandidate(
                name=f"vol_z_{w}", family="microstructure", window=w,
                transform="zscore",
                description=f"volume z-score ({w}-bar)",
                series=(vol - vmean) / vstd.replace(0, np.nan),
            ))
            # VWAP deviation
            vwap = (close * vol).rolling(w).sum() / vol.rolling(w).sum()
            out.append(FeatureCandidate(
                name=f"vwap_dev_{w}", family="microstructure", window=w,
                transform="ratio",
                description=f"close / vwap({w}) - 1",
                series=(close / vwap.replace(0, np.nan)) - 1.0,
            ))
        # Volume-weighted RSI proxy
        # (volume * close change sign summed)
        delta = close.diff()
        signed = (delta * vol).rolling(14).sum()
        total = (vol * delta.abs()).rolling(14).sum()
        out.append(FeatureCandidate(
            name="vw_rsi_14", family="microstructure", window=14,
            transform="raw",
            description="volume-weighted RSI proxy",
            series=signed / total.replace(0, np.nan),
        ))
        return out

    # ----------------------------------------------------------------
    def _regime_features(self, df: pd.DataFrame) -> list[FeatureCandidate]:
        out: list[FeatureCandidate] = []
        close = df["close"]
        for w in self.regime_windows:
            # Linear regression slope
            def slope(window: pd.Series) -> float:
                n = len(window)
                if n < 2:
                    return np.nan
                x = np.arange(n, dtype=float)
                y = window.values
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    return np.nan
                denom = ((x[mask] - x[mask].mean()) ** 2).sum()
                if denom == 0:
                    return 0.0
                return float(((x[mask] - x[mask].mean())
                              * (y[mask] - y[mask].mean())).sum() / denom)
            s = close.rolling(w).apply(slope, raw=False)
            out.append(FeatureCandidate(
                name=f"trend_slope_{w}", family="regime", window=w,
                transform="slope",
                description=f"linear regression slope ({w}-bar)",
                series=s,
            ))
            # Autocorrelation (lag-1 of log returns)
            log_ret = np.log(close / close.shift(1))
            out.append(FeatureCandidate(
                name=f"autocorr_{w}", family="regime", window=w,
                transform="raw",
                description=f"lag-1 autocorrelation of returns ({w}-bar)",
                series=log_ret.rolling(w).apply(
                    lambda x: float(pd.Series(x).autocorr(lag=1))
                    if len(x) == w and not pd.Series(x).isna().any() else np.nan,
                    raw=False,
                ),
            ))
        return out

    # ----------------------------------------------------------------
    def _cross_asset_features(self, df: pd.DataFrame,
                              other_closes: dict[str, pd.Series]
                              ) -> list[FeatureCandidate]:
        out: list[FeatureCandidate] = []
        primary = df["close"]
        for other_name, other_close in other_closes.items():
            # Align
            aligned = pd.concat([primary, other_close], axis=1).dropna()
            aligned.columns = ["primary", "other"]
            if len(aligned) < 50:
                continue
            ret_p = np.log(aligned["primary"] / aligned["primary"].shift(1))
            ret_o = np.log(aligned["other"] / aligned["other"].shift(1))
            # Rolling correlation
            for w in (20, 50):
                corr = ret_p.rolling(w).corr(ret_o)
                out.append(FeatureCandidate(
                    name=f"corr_{other_name}_{w}", family="cross_asset",
                    window=w, transform="raw",
                    description=f"rolling corr with {other_name} ({w}-bar)",
                    series=corr,
                ))
            # Beta
            for w in (50,):
                cov = ret_p.rolling(w).cov(ret_o)
                var = ret_o.rolling(w).var()
                out.append(FeatureCandidate(
                    name=f"beta_{other_name}_{w}", family="cross_asset",
                    window=w, transform="ratio",
                    description=f"rolling beta vs {other_name} ({w}-bar)",
                    series=cov / var.replace(0, np.nan),
                ))
        return out

    # ----------------------------------------------------------------
    def as_dataframe(self, candidates: list[FeatureCandidate]) -> pd.DataFrame:
        """Combine a list of FeatureCandidates into a single DataFrame."""
        if not candidates:
            return pd.DataFrame()
        data = {c.name: c.series for c in candidates if c.series is not None}
        return pd.DataFrame(data)
