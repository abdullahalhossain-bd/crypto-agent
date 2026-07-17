"""utils/indicators/statistics.py
=====================================================================
Statistical Indicators (Improvement #8)
=====================================================================
Z-Score, Rolling Mean, Rolling Std, Skewness, Kurtosis, Entropy,
Variance, Autocorrelation, Hurst Exponent, Stationarity Score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.indicators.caching import cached


@cached()
def zscore(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-Score — (price - mean) / std."""
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / std.replace(0, np.nan)


@cached()
def rolling_mean(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Mean."""
    return close.rolling(period).mean()


@cached()
def rolling_std(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Standard Deviation."""
    return close.rolling(period).std()


@cached()
def rolling_variance(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Variance."""
    return close.rolling(period).var()


@cached()
def skewness(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Skewness — asymmetry of distribution."""
    return close.rolling(period).skew()


@cached()
def kurtosis(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Kurtosis — tail heaviness."""
    return close.rolling(period).kurt()


@cached()
def entropy(close: pd.Series, period: int = 20,
            n_bins: int = 10) -> pd.Series:
    """Shannon Entropy of returns distribution.

    Higher entropy = more random/uncertain
    Lower entropy = more predictable
    """
    returns = close.pct_change()
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(period, len(close)):
        window = returns.iloc[i - period:i].dropna()
        if len(window) < 10:
            continue
        hist, _ = np.histogram(window, bins=n_bins, density=True)
        hist = hist[hist > 0]
        result.iloc[i] = -np.sum(hist * np.log2(hist / hist.sum())) if len(hist) > 0 else 0
    return result


@cached()
def autocorrelation(close: pd.Series, period: int = 20,
                    lag: int = 1) -> pd.Series:
    """Rolling autocorrelation at given lag."""
    returns = close.pct_change()
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(period + lag, len(close)):
        window = returns.iloc[i - period:i]
        if window.std() == 0:
            continue
        result.iloc[i] = window.autocorr(lag=lag)
    return result


@cached()
def hurst_exponent(close: pd.Series, max_lag: int = 20) -> pd.Series:
    """Hurst Exponent — long-memory of time series.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    result = pd.Series(np.nan, index=close.index, dtype=float)
    window = 100  # need enough data points
    for i in range(window, len(close)):
        ts = close.iloc[i - window:i].values
        if np.std(ts) == 0:
            continue
        lags = range(2, min(max_lag, len(ts) // 2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 3:
            continue
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        result.iloc[i] = poly[0]
    return result


@cached()
def stationarity_score(close: pd.Series, period: int = 50) -> pd.Series:
    """Stationarity score via ADF test p-value.

    Returns rolling p-value from Augmented Dickey-Fuller test.
    p < 0.05: stationary (mean-reverting)
    p >= 0.05: non-stationary (trending)
    """
    from statsmodels.tsa.stattools import adfuller
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(period, len(close)):
        window = close.iloc[i - period:i].dropna()
        if len(window) < 30 or window.std() == 0:
            continue
        try:
            p_value = adfuller(window, autolag="AIC")[1]
            result.iloc[i] = p_value
        except Exception:
            continue
    return result


@cached()
def correlation(a: pd.Series, b: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return a.rolling(period).corr(b)


@cached()
def beta(asset: pd.Series, benchmark: pd.Series,
         period: int = 60) -> pd.Series:
    """Rolling beta of asset vs benchmark."""
    cov = asset.rolling(period).cov(benchmark)
    var = benchmark.rolling(period).var()
    return cov / var.replace(0, np.nan)
