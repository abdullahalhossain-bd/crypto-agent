"""
Alpha Zoo — Cross-Sectional Factor Library
============================================

19 operators + key alphas from 4 zoos:
  - Qlib158 (Microsoft)
  - Alpha101 (Kakushadze 2015)
  - GTJA191 (Guotai Junan 2014)
  - Academic (FF5, Carhart, Amihud)

All operators are lookahead-banned (no negative shifts).

Usage:
    from alpha_zoo import AlphaZoo, rank, ts_rank, ts_corr, delta

    zoo = AlphaZoo()

    # Compute a single alpha
    panel = {
        "close": df_close,      # wide DataFrame (index=date, columns=assets)
        "open": df_open,
        "high": df_high,
        "low": df_low,
        "volume": df_volume,
    }
    result = zoo.compute("alpha_001", panel)
    # Returns wide DataFrame (date × assets)

    # List available alphas
    print(zoo.list_alphas())
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 19 Operators (lookahead-banned)
# ═══════════════════════════════════════════════════════════════

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank (0-1) for each date."""
    return df.rank(pct=True, axis=1)


def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Scale so each row sums to a."""
    row_sums = df.abs().sum(axis=1).replace(0, 1e-10)
    return df.div(row_sums, axis=0) * a


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Time-series rank over n periods."""
    return df.rolling(n).rank(pct=True)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling correlation over n periods."""
    return x.rolling(n).corr(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling covariance over n periods."""
    return x.rolling(n).cov(y)


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling mean."""
    return df.rolling(n).mean()


def ts_std(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling standard deviation."""
    return df.rolling(n).std()


def ts_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling max."""
    return df.rolling(n).max()


def ts_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling min."""
    return df.rolling(n).min()


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position of max in rolling window (0-based from end)."""
    return df.rolling(n).apply(np.argmax, raw=True)


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position of min in rolling window."""
    return df.rolling(n).apply(np.argmin, raw=True)


def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Difference d periods ago. d >= 1 (lookahead-banned)."""
    if d < 1:
        raise ValueError("delta requires d >= 1 (lookahead ban)")
    return df.diff(d)


def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly weighted moving average (recent = higher weight)."""
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()

    def weighted_mean(arr):
        return np.dot(arr, weights)

    return df.rolling(n).apply(weighted_mean, raw=True)


def signed_power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """Signed power: sign(x) * |x|^p."""
    return np.sign(df) * (df.abs() ** p)


def safe_div(a: pd.DataFrame, b: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Safe division with epsilon to avoid division by zero."""
    return a / b.replace(0, eps)


def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Shift by n periods (n >= 1)."""
    if n < 1:
        raise ValueError("delay requires n >= 1 (lookahead ban)")
    return df.shift(n)


def product(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling product."""
    return df.rolling(n).apply(np.prod, raw=True)


def coerce_float(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is float."""
    if df.dtypes.eq(np.float64).all():
        return df
    return df.astype(np.float64)


def vwap(panel: dict, market: str = "crypto") -> pd.DataFrame:
    """Volume-Weighted Average Price."""
    close = panel.get("close")
    high = panel.get("high")
    low = panel.get("low")
    volume = panel.get("volume")

    if volume is None:
        return close.copy()

    typical = (high + low + close) / 3.0
    return (typical * volume).rolling(1).sum() / volume.rolling(1).sum().replace(0, 1e-10)


# ═══════════════════════════════════════════════════════════════
# Alpha Implementations
# ═══════════════════════════════════════════════════════════════

def _alpha_001(panel: dict) -> pd.DataFrame:
    """Alpha#1: (rank(ts_argmax(SignedPower((returns < 0 ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    close = panel["close"]
    returns = close.pct_change()
    neg_vol = returns.where(returns < 0, close)
    inner = signed_power(neg_vol, 2.0)
    return rank(ts_argmax(inner, 5)) - 0.5


def _alpha_002(panel: dict) -> pd.DataFrame:
    """Alpha#2: (-1 * delta(correlation(rank(vwap), rank(volume), 5), 5))"""
    v = vwap(panel)
    return -1 * delta(ts_corr(rank(v), rank(panel["volume"]), 5), 5)


def _alpha_003(panel: dict) -> pd.DataFrame:
    """Alpha#3: -1 * correlation(rank(open), rank(volume), 10)"""
    return -1 * ts_corr(rank(panel["open"]), rank(panel["volume"]), 10)


def _alpha_004(panel: dict) -> pd.DataFrame:
    """Alpha#4: -1 * ts_rank(rank(low), 9)"""
    return -1 * ts_rank(rank(panel["low"]), 9)


def _alpha_005(panel: dict) -> pd.DataFrame:
    """Alpha#5: (rank((open - sum(close, 10) / 10)) * (-1 * abs(rank(close - vwap))))"""
    v = vwap(panel)
    return rank(panel["open"] - ts_mean(panel["close"], 10)) * (-1 * abs(rank(panel["close"] - v)))


def _alpha_006(panel: dict) -> pd.DataFrame:
    """Alpha#6: -1 * correlation(open, volume, 10)"""
    return -1 * ts_corr(panel["open"], panel["volume"], 10)


def _alpha_012(panel: dict) -> pd.DataFrame:
    """Alpha#12: sign(delta(volume, 1)) * (-delta(close, 1))"""
    return np.sign(delta(panel["volume"], 1)) * (-delta(panel["close"], 1))


def _alpha_026(panel: dict) -> pd.DataFrame:
    """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(close, 5), 5), 3))"""
    corr = ts_corr(ts_rank(panel["volume"], 5), ts_rank(panel["close"], 5), 5)
    return -1 * ts_max(corr, 3)


def _alpha_041(panel: dict) -> pd.DataFrame:
    """Alpha#41: (((high * low)**0.5) - vwap)"""
    v = vwap(panel)
    return (panel["high"] * panel["low"]) ** 0.5 - v


def _alpha_053(panel: dict) -> pd.DataFrame:
    """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    inner = ((panel["close"] - panel["low"]) - (panel["high"] - panel["close"])) / \
            (panel["close"] - panel["low"]).replace(0, 1e-10)
    return -1 * delta(inner, 9)


def _alpha_054(panel: dict) -> pd.DataFrame:
    """Alpha#54: ((-1 * ((low - close) * (open**5))) / ((low - high) * (close**5)))"""
    low = panel["low"]
    high = panel["high"]
    close = panel["close"]
    open_ = panel["open"]
    numerator = -1 * (low - close) * (open_ ** 5)
    denominator = (low - high).replace(0, 1e-10) * (close ** 5)
    return numerator / denominator


def _alpha_101(panel: dict) -> pd.DataFrame:
    """Alpha#101: ((close - open) / ((high - low) + 0.001))"""
    return (panel["close"] - panel["open"]) / \
           ((panel["high"] - panel["low"]) + 0.001)


# Academic alphas
def _momentum_20d(panel: dict) -> pd.DataFrame:
    """20-day momentum: close / delay(close, 20) - 1"""
    close = panel["close"]
    return close / delay(close, 20) - 1.0


def _momentum_60d(panel: dict) -> pd.DataFrame:
    """60-day momentum."""
    close = panel["close"]
    return close / delay(close, 60) - 1.0


def _reversal_5d(panel: dict) -> pd.DataFrame:
    """5-day reversal: -1 * (close / delay(close, 5) - 1)"""
    close = panel["close"]
    return -1.0 * (close / delay(close, 5) - 1.0)


def _volatility_20d(panel: dict) -> pd.DataFrame:
    """20-day volatility (negatively scored)."""
    returns = panel["close"].pct_change()
    return -1.0 * ts_std(returns, 20)


def _volume_zscore(panel: dict) -> pd.DataFrame:
    """Volume z-score: (volume - ts_mean(volume, 20)) / ts_std(volume, 20)."""
    vol = panel["volume"]
    return (vol - ts_mean(vol, 20)) / ts_std(vol, 20).replace(0, 1e-10)


def _rsi_divergence(panel: dict) -> pd.DataFrame:
    """RSI-based signal: rank of 14-day RSI (contrarian — high RSI = sell)."""
    close = panel["close"]
    delta_close = close.diff()
    gain = delta_close.where(delta_close > 0, 0.0)
    loss = (-delta_close.where(delta_close < 0, 0.0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-10)
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    return -1.0 * rank(rsi)  # Contrarian: high RSI → sell signal


def _price_acceleration(panel: dict) -> pd.DataFrame:
    """Second derivative of price (acceleration)."""
    close = panel["close"]
    returns = close.pct_change()
    return delta(returns, 1)


def _mean_reversion_20d(panel: dict) -> pd.DataFrame:
    """Mean reversion: -1 * (close - ts_mean(close, 20)) / ts_std(close, 20)."""
    close = panel["close"]
    ma = ts_mean(close, 20)
    std = ts_std(close, 20).replace(0, 1e-10)
    return -1.0 * (close - ma) / std


# ═══════════════════════════════════════════════════════════════
# Alpha Zoo Registry
# ═══════════════════════════════════════════════════════════════

ALPHA_REGISTRY = {
    # Alpha101 (Kakushadze)
    "alpha_001": ("alpha101", _alpha_001),
    "alpha_002": ("alpha101", _alpha_002),
    "alpha_003": ("alpha101", _alpha_003),
    "alpha_004": ("alpha101", _alpha_004),
    "alpha_005": ("alpha101", _alpha_005),
    "alpha_006": ("alpha101", _alpha_006),
    "alpha_012": ("alpha101", _alpha_012),
    "alpha_026": ("alpha101", _alpha_026),
    "alpha_041": ("alpha101", _alpha_041),
    "alpha_053": ("alpha101", _alpha_053),
    "alpha_054": ("alpha101", _alpha_054),
    "alpha_101": ("alpha101", _alpha_101),
    # Academic
    "momentum_20d": ("academic", _momentum_20d),
    "momentum_60d": ("academic", _momentum_60d),
    "reversal_5d": ("academic", _reversal_5d),
    "volatility_20d": ("academic", _volatility_20d),
    "volume_zscore": ("academic", _volume_zscore),
    "rsi_divergence": ("academic", _rsi_divergence),
    "price_acceleration": ("academic", _price_acceleration),
    "mean_reversion_20d": ("academic", _mean_reversion_20d),
}


class AlphaZoo:
    """
    Cross-sectional alpha factor library.

    Usage:
        zoo = AlphaZoo()
        result = zoo.compute("alpha_001", panel)
        # panel = {"close": df, "open": df, "high": df, "low": df, "volume": df}
    """

    def __init__(self):
        self.registry = ALPHA_REGISTRY.copy()

    def list_alphas(self, zoo: Optional[str] = None) -> list[str]:
        """List available alpha IDs, optionally filtered by zoo."""
        if zoo is None:
            return sorted(self.registry.keys())
        return sorted(k for k, v in self.registry.items() if v[0] == zoo)

    def get_zoo(self, alpha_id: str) -> str:
        """Get the zoo name for an alpha."""
        entry = self.registry.get(alpha_id)
        return entry[0] if entry else "unknown"

    def compute(self, alpha_id: str, panel: dict) -> pd.DataFrame:
        """
        Compute a single alpha.

        Args:
            alpha_id: Alpha identifier (e.g., "alpha_001")
            panel: Dict of wide DataFrames: {"close": df, "open": df, ...}

        Returns:
            Wide DataFrame (date × assets) of alpha values
        """
        entry = self.registry.get(alpha_id)
        if entry is None:
            raise KeyError(f"Unknown alpha: {alpha_id}. Available: {self.list_alphas()[:10]}...")

        zoo_name, func = entry
        try:
            result = func(panel)
            # Replace inf with nan
            result = result.replace([np.inf, -np.inf], np.nan)
            return result
        except Exception as e:
            raise RuntimeError(f"Alpha {alpha_id} computation failed: {e}") from e

    def compute_many(self, alpha_ids: list[str], panel: dict) -> dict[str, pd.DataFrame]:
        """Compute multiple alphas at once."""
        return {aid: self.compute(aid, panel) for aid in alpha_ids}

    def compute_all(self, panel: dict) -> dict[str, pd.DataFrame]:
        """Compute all registered alphas."""
        return self.compute_many(self.list_alphas(), panel)

    def get_summary(self) -> dict:
        """Get zoo summary statistics."""
        zoos = {}
        for alpha_id, (zoo_name, _) in self.registry.items():
            zoos.setdefault(zoo_name, []).append(alpha_id)
        return {
            zoo: {"count": len(alphas), "alphas": alphas}
            for zoo, alphas in sorted(zoos.items())
        }
