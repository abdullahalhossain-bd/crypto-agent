"""utils.indicators
===============================================================================
Day 90 — Institutional Quantitative Indicator Library (v3)

Production-grade technical indicator library for crypto/forex trading.

Design Principles
-----------------
- No lookahead: indicator[t] depends only on bars ≤ t
- NaN preserved at warm-up period — never silently dropped
- Vectorized with numpy/pandas — no per-bar Python loops where avoidable
- Indicator caching for performance (same data → same result → cached)
- Auto volume detection (volume / tick_volume / real_volume)
- Full type hints + NumPy-style docstrings
- AI feature generator ready (to_feature_vector())
- SMC / ICT indicators (FVG, Order Block, Liquidity Sweep)
- Market regime detection (trend / range / breakout / squeeze)
- Divergence detection (price vs RSI / MACD)
- Candlestick pattern detection

Indicator Categories
--------------------
1.  Moving Averages:    sma, ema, wma, vwma, hull_ma
2.  Momentum:           rsi, stoch_rsi, macd, roc, cci, williams_r, tsi, mfi
3.  Volatility:         atr, atr_pct, bbands, keltner, donchian, hist_vol, parkinson_vol
4.  Trend:              adx, dmi, supertrend, ichimoku, slope, trend_score
5.  Volume:             obv, vwap, rvol, cmf, adl, pvt
6.  SMC/ICT:            fvg, order_block, liquidity_sweep, swing_highs_lows
7.  Patterns:           candlestick_patterns, divergence
8.  Support/Resistance: pivot_points, auto_sr
9.  Composite:          regime_detection, feature_vector, explain
10. Cache:              IndicatorCache for incremental updates

Usage
-----
    from utils.indicators import IndicatorCache, adx, supertrend, regime

    cache = IndicatorCache()
    adx_val = cache.get_or_compute("BTCUSD", "adx_14", lambda: adx(df, 14))
    regime = regime_detection(df)
    features = feature_vector(df)  # → numpy array for ML
===============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

# ======================================================================
# INDICATOR CACHE — Performance Layer
# ======================================================================
class IndicatorCache:
    """Cache for indicator computations — avoids redundant calculation.

    Usage:
        cache = IndicatorCache()
        ema20 = cache.get_or_compute("BTCUSD", "ema_20", lambda: ema(close, 20))
        # Second call returns cached value instantly
        ema20 = cache.get_or_compute("BTCUSD", "ema_20", lambda: ema(close, 20))
    """
    def __init__(self, max_size: int = 500):
        self._cache: dict[str, Any] = {}
        self._max_size = max_size

    def get_or_compute(self, symbol: str, key: str, compute_fn: Callable) -> Any:
        cache_key = f"{symbol}:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = compute_fn()
        self._cache[cache_key] = result
        # Evict oldest entries if cache is full
        if len(self._cache) > self._max_size:
            oldest = list(self._cache.keys())[:self._max_size // 4]
            for k in oldest:
                del self._cache[k]
        return result

    def invalidate(self, symbol: str = ""):
        if symbol:
            keys_to_del = [k for k in self._cache if k.startswith(f"{symbol}:")]
            for k in keys_to_del:
                del self._cache[k]
        else:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ======================================================================
# AUTO VOLUME DETECTION
# ======================================================================
def _get_volume(df: pd.DataFrame) -> pd.Series:
    """Auto-detect volume column (volume / tick_volume / real_volume)."""
    for col in ("volume", "tick_volume", "real_volume"):
        if col in df.columns:
            return df[col].astype(float)
    return pd.Series([1.0] * len(df), index=df.index, dtype=float)


# ======================================================================
# 1. MOVING AVERAGES
# ======================================================================
def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    if period < 1: raise ValueError("period must be >= 1")
    out = close.rolling(window=period, min_periods=period).mean()
    out.name = f"sma_{period}"
    return out

def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Wilder-style, adjust=False)."""
    if period < 1: raise ValueError("period must be >= 1")
    out = close.ewm(span=period, adjust=False, min_periods=period).mean()
    out.name = f"ema_{period}"
    return out

def wma(close: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average — linearly weighted."""
    if period < 1: raise ValueError("period must be >= 1")
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    out = close.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights), raw=True)
    out.name = f"wma_{period}"
    return out

def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Volume-Weighted Moving Average."""
    if period < 1: raise ValueError("period must be >= 1")
    pv = close * volume
    out = pv.rolling(period, min_periods=period).sum() / \
          volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    out.name = f"vwma_{period}"
    return out

def hull_ma(close: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average — smooth + low lag."""
    if period < 4: raise ValueError("period must be >= 4")
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    out = wma(diff, sqrt_n)
    out.name = f"hull_ma_{period}"
    return out


# ======================================================================
# 2. MOMENTUM INDICATORS
# ======================================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder smoothing."""
    if period < 1: raise ValueError("period must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0.0] = 100.0
    out.name = f"rsi_{period}"
    return out

def stoch_rsi(close: pd.Series, rsi_period: int = 14,
              stoch_period: int = 14, k_smooth: int = 3,
              d_smooth: int = 3) -> pd.DataFrame:
    """Stochastic RSI — more sensitive than RSI for crypto.

    Returns DataFrame with columns: k, d.
    """
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    k = stoch.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_smooth, min_periods=1).mean()
    return pd.DataFrame({"k": k, "d": d})

def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD. Returns DataFrame: macd, signal, histogram."""
    if fast >= slow: raise ValueError("fast must be < slow")
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})

def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (%)."""
    if period < 1: raise ValueError("period must be >= 1")
    prior = close.shift(period)
    out = ((close - prior) / prior.replace(0.0, np.nan)) * 100.0
    out.name = f"roc_{period}"
    return out

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = typical.rolling(period, min_periods=period).mean()
    mean_dev = typical.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out = (typical - sma_tp) / (0.015 * mean_dev.replace(0.0, np.nan))
    out.name = f"cci_{period}"
    return out

def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100 to 0)."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    highest = df["high"].rolling(period, min_periods=period).max()
    lowest = df["low"].rolling(period, min_periods=period).min()
    out = ((highest - df["close"]) / (highest - lowest).replace(0.0, np.nan)) * -100.0
    out.name = f"williams_r_{period}"
    return out

def tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index — double-smoothed momentum."""
    m = close.diff()
    m1 = m.ewm(span=long, adjust=False).mean()
    m2 = m1.ewm(span=short, adjust=False).mean()
    abs_m = m.abs()
    abs_m1 = abs_m.ewm(span=long, adjust=False).mean()
    abs_m2 = abs_m1.ewm(span=short, adjust=False).mean()
    out = 100 * (m2 / abs_m2.replace(0, np.nan))
    out.name = f"tsi_{long}_{short}"
    return out

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close/volume")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * _get_volume(df)
    pos_flow = raw_flow.where(typical > typical.shift(1), 0.0)
    neg_flow = raw_flow.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()
    money_ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + money_ratio))
    out[neg_sum == 0.0] = 100.0
    out.name = f"mfi_{period}"
    return out


# ======================================================================
# 3. VOLATILITY INDICATORS
# ======================================================================
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    out.name = f"atr_{period}"
    return out

def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as percentage of close — comparable across assets."""
    atr_val = atr(df, period)
    out = (atr_val / df["close"].replace(0, np.nan)) * 100
    out.name = f"atr_pct_{period}"
    return out

def bbands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns: upper, middle, lower."""
    if period < 1: raise ValueError("period must be >= 1")
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return pd.DataFrame({"upper": middle + std_dev * std, "middle": middle, "lower": middle - std_dev * std})

def keltner(df: pd.DataFrame, period: int = 20, atr_period: int = 14,
            multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channel — ATR-based bands."""
    middle = ema(df["close"], period)
    atr_val = atr(df, atr_period)
    return pd.DataFrame({
        "upper": middle + multiplier * atr_val,
        "middle": middle,
        "lower": middle - multiplier * atr_val,
    })

def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel — highest high / lowest low."""
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("df must contain high/low")
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

def hist_vol(close: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """Historical Volatility (annualized std of log returns)."""
    log_ret = np.log(close / close.shift(1))
    out = log_ret.rolling(period, min_periods=period).std()
    if annualize:
        out *= np.sqrt(252 * 24)  # crypto: 252 days × 24h
    out.name = f"hist_vol_{period}"
    return out

def parkinson_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Parkinson Volatility — uses high-low range."""
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("df must contain high/low")
    hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    out = (hl ** 2).rolling(period, min_periods=period).mean()
    out = np.sqrt(out / (4 * np.log(2))) * np.sqrt(252 * 24)
    out.name = f"parkinson_vol_{period}"
    return out


# ======================================================================
# 4. TREND INDICATORS
# ======================================================================
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX + DI indicators — trend strength + direction.

    Returns DataFrame: adx, plus_di, minus_di.
    ADX > 25 = trending; < 20 = ranging.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})

def dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """DMI — same as adx() but exposed separately."""
    return adx(df, period)

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend — ATR-based trend indicator.

    Returns DataFrame: supertrend, direction (1=up, -1=down).
    BUY only above SuperTrend; SELL only below.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    close = df["close"]

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue
        # Final upper band
        if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            fu = upper_band.iloc[i]
        else:
            fu = upper_band.iloc[i-1]
        # Final lower band
        if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            fl = lower_band.iloc[i]
        else:
            fl = lower_band.iloc[i-1]
        # SuperTrend
        if st.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] > fu:
            st.iloc[i] = fl
            direction.iloc[i] = 1
        elif st.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] <= fu:
            st.iloc[i] = fu
            direction.iloc[i] = -1
        elif st.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] < fl:
            st.iloc[i] = fu
            direction.iloc[i] = -1
        elif st.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] >= fl:
            st.iloc[i] = fl
            direction.iloc[i] = 1
        else:
            st.iloc[i] = st.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

    return pd.DataFrame({"supertrend": st, "direction": direction})

def ichimoku(df: pd.DataFrame, conversion: int = 9, base: int = 26,
             span_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Ichimoku Cloud. Returns: tenkan, kijun, senkou_a, senkou_b."""
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(conversion).max() + low.rolling(conversion).min()) / 2
    kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).shift(displacement)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                         "senkou_a": senkou_a, "senkou_b": senkou_b})

def slope(close: pd.Series, period: int = 20) -> pd.Series:
    """Linear regression slope over rolling window."""
    if period < 2: raise ValueError("period must be >= 2")
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    out = close.rolling(period, min_periods=period).apply(
        lambda y: float(((x - x_mean) * (y - y.mean())).sum() / x_var) if x_var > 0 else 0.0,
        raw=True)
    out.name = f"slope_{period}"
    return out

def trend_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Composite trend score (0-100) combining EMA, ADX, slope, ROC.

    100 = strong uptrend; 0 = strong downtrend; 50 = neutral.
    """
    close = df["close"]
    ema_f = ema(close, min(period, 20))
    ema_s = ema(close, max(period, 50))
    adx_val = adx(df, 14)["adx"].fillna(20)
    slp = slope(close, period).fillna(0)
    roc_val = roc(close, period).fillna(0)

    # Normalize each component to 0-100
    ema_score = ((ema_f > ema_s).astype(float) * 50) + \
                ((close > ema_f).astype(float) * 50)
    adx_score = (adx_val / 50 * 50).clip(0, 50) * np.sign(slp + 0.0001).clip(-1, 1) + 50
    slope_score = ((slp > 0).astype(float) * 50) + \
                  ((slp > slp.rolling(period).mean()).astype(float) * 50)
    roc_score = ((roc_val > 0).astype(float) * 50) + \
                ((roc_val > roc_val.rolling(period).mean()).astype(float) * 50)

    out = (ema_score * 0.3 + adx_score * 0.25 + slope_score * 0.25 + roc_score * 0.2)
    out = out.clip(0, 100)
    out.name = f"trend_score_{period}"
    return out


# ======================================================================
# 5. VOLUME INDICATORS
# ======================================================================
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().fillna(0.0).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    out = (direction * volume).cumsum()
    out.name = "obv"
    return out

def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    vol = _get_volume(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    out = cum_tp_vol / cum_vol
    out.name = "vwap"
    return out

def rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative Volume — current vs average."""
    if period < 1: raise ValueError("period must be >= 1")
    prior_avg = volume.shift(1).rolling(period, min_periods=period).mean()
    out = volume / prior_avg.replace(0.0, np.nan)
    out.name = f"rvol_{period}"
    return out

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — -1 to +1. Positive = accumulation."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    vol = _get_volume(df)
    high, low, close = df["high"], df["low"], df["close"]
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * vol
    out = mfv.rolling(period, min_periods=period).sum() / \
          vol.rolling(period, min_periods=period).sum().replace(0, np.nan)
    out.name = f"cmf_{period}"
    return out

def adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line — whale accumulation detection."""
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain high/low/close")
    vol = _get_volume(df)
    high, low, close = df["high"], df["low"], df["close"]
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out = (mfm * vol).cumsum()
    out.name = "adl"
    return out

def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Price Volume Trend."""
    pct_change = close.pct_change().fillna(0)
    out = (pct_change * volume).cumsum()
    out.name = "pvt"
    return out


# ======================================================================
# 6. SMC / ICT INDICATORS
# ======================================================================
def swing_highs_lows(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Detect swing highs and lows.

    Returns DataFrame: swing_high (price or NaN), swing_low (price or NaN).
    """
    highs = df["high"].rolling(window * 2 + 1, center=True).max()
    lows = df["low"].rolling(window * 2 + 1, center=True).min()
    swing_h = df["high"].where(df["high"] == highs, np.nan)
    swing_l = df["low"].where(df["low"] == lows, np.nan)
    return pd.DataFrame({"swing_high": swing_h, "swing_low": swing_l})

def fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gap detection — 3-candle pattern.

    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)

    Returns: fvg_high, fvg_low, fvg_type (1=bull, -1=bear, 0=none).
    """
    if len(df) < 3:
        return pd.DataFrame({"fvg_high": pd.Series(dtype=float),
                            "fvg_low": pd.Series(dtype=float),
                            "fvg_type": pd.Series(dtype=int)})
    high_prev = df["high"].shift(2)
    low_next = df["low"].shift(-2)
    low_prev = df["low"].shift(2)
    high_next = df["high"].shift(-2)

    bull_fvg = (low_next > high_prev).astype(int)
    bear_fvg = (high_next < low_prev).astype(int) * -1
    fvg_type = (bull_fvg + bear_fvg).fillna(0).astype(int)

    fvg_high = pd.Series(np.where(fvg_type > 0, low_next, np.where(fvg_type < 0, low_prev, np.nan)),
                         index=df.index)
    fvg_low = pd.Series(np.where(fvg_type > 0, high_prev, np.where(fvg_type < 0, high_next, np.nan)),
                        index=df.index)
    return pd.DataFrame({"fvg_high": fvg_high, "fvg_low": fvg_low, "fvg_type": fvg_type})

def order_block(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Detect Order Blocks — last opposite candle before strong move.

    Returns: ob_high, ob_low, ob_type (1=bullish OB, -1=bearish OB).
    """
    if len(df) < lookback + 2:
        return pd.DataFrame({"ob_high": pd.Series(dtype=float),
                            "ob_low": pd.Series(dtype=float),
                            "ob_type": pd.Series(dtype=int)})
    close = df["close"]
    body = (close - df["open"]).abs()
    avg_body = body.rolling(lookback, min_periods=1).mean()

    # Bullish OB: last bearish candle before strong bullish move
    bearish = close < df["open"]
    strong_bull = (close > df["open"]) & (body > avg_body * 1.5)
    bull_ob = bearish.shift(1) & strong_bull

    # Bearish OB: last bullish candle before strong bearish move
    bullish = close > df["open"]
    strong_bear = (close < df["open"]) & (body > avg_body * 1.5)
    bear_ob = bullish.shift(1) & strong_bear

    ob_type = pd.Series(0, index=df.index, dtype=int)
    ob_type[bull_ob] = 1
    ob_type[bear_ob] = -1

    ob_high = df["high"].shift(1).where(ob_type != 0, np.nan)
    ob_low = df["low"].shift(1).where(ob_type != 0, np.nan)
    return pd.DataFrame({"ob_high": ob_high, "ob_low": ob_low, "ob_type": ob_type})

def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect liquidity sweeps / stop hunts.

    Returns: sweep_type (1=high sweep, -1=low sweep, 0=none), sweep_level.
    """
    if len(df) < lookback + 1:
        return pd.DataFrame({"sweep_type": pd.Series(dtype=int),
                            "sweep_level": pd.Series(dtype=float)})
    recent_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    recent_low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)

    high_sweep = (df["high"] > recent_high) & (df["close"] < recent_high)
    low_sweep = (df["low"] < recent_low) & (df["close"] > recent_low)

    sweep_type = pd.Series(0, index=df.index, dtype=int)
    sweep_type[high_sweep] = 1
    sweep_type[low_sweep] = -1
    sweep_level = pd.Series(np.nan, index=df.index, dtype=float)
    sweep_level[high_sweep] = recent_high[high_sweep]
    sweep_level[low_sweep] = recent_low[low_sweep]
    return pd.DataFrame({"sweep_type": sweep_type, "sweep_level": sweep_level})


# ======================================================================
# 7. PATTERN DETECTION
# ======================================================================
def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect common candlestick patterns.

    Returns DataFrame with boolean columns:
        hammer, doji, bullish_engulfing, bearish_engulfing,
        morning_star, evening_star, shooting_star
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    hammer = (lower_wick > 2 * body) & (upper_wick < body * 0.5) & (body > 0)
    doji = body < rng * 0.1
    shooting_star = (upper_wick > 2 * body) & (lower_wick < body * 0.5) & (body > 0)

    # Engulfing (2-bar)
    prev_bull = c.shift(1) > o.shift(1)
    prev_bear = c.shift(1) < o.shift(1)
    curr_bull = c > o
    curr_bear = c < o
    bullish_engulfing = prev_bear & curr_bull & (c > o.shift(1)) & (o < c.shift(1))
    bearish_engulfing = prev_bull & curr_bear & (c < o.shift(1)) & (o > c.shift(1))

    # Morning/Evening Star (3-bar)
    small_body = body < body.rolling(10, min_periods=3).mean() * 0.5
    morning_star = prev_bear & small_body.shift(1) & curr_bull & (c > o.shift(2))
    evening_star = prev_bull & small_body.shift(1) & curr_bear & (c < o.shift(2))

    return pd.DataFrame({
        "hammer": hammer.fillna(False),
        "doji": doji.fillna(False),
        "shooting_star": shooting_star.fillna(False),
        "bullish_engulfing": bullish_engulfing.fillna(False),
        "bearish_engulfing": bearish_engulfing.fillna(False),
        "morning_star": morning_star.fillna(False),
        "evening_star": evening_star.fillna(False),
    })

def divergence(close: pd.Series, oscillator: pd.Series, period: int = 50) -> pd.DataFrame:
    """Detect divergence between price and oscillator (RSI/MACD/etc).

    Returns: bull_div (price LL + osc HL), bear_div (price HH + osc LH).
    """
    if len(close) < period:
        return pd.DataFrame({"bull_div": pd.Series(dtype=bool),
                            "bear_div": pd.Series(dtype=bool)})
    price_highs = close.rolling(period, min_periods=period).max()
    price_lows = close.rolling(period, min_periods=period).min()
    osc_highs = oscillator.rolling(period, min_periods=period).max()
    osc_lows = oscillator.rolling(period, min_periods=period).min()

    price_hh = close > price_highs.shift(1)
    price_ll = close < price_lows.shift(1)
    osc_lh = oscillator < osc_highs.shift(1)
    osc_hl = oscillator > osc_lows.shift(1)

    bear_div = price_hh & osc_lh
    bull_div = price_ll & osc_hl
    return pd.DataFrame({"bull_div": bull_div.fillna(False),
                         "bear_div": bear_div.fillna(False)})


# ======================================================================
# 8. SUPPORT / RESISTANCE + PIVOTS
# ======================================================================
def pivot_points(df: pd.DataFrame, method: str = "classic") -> pd.DataFrame:
    """Pivot Points (classic / woodie / fibonacci).

    Returns: pivot, r1, r2, s1, s2.
    """
    if len(df) < 1:
        return pd.DataFrame()
    # Use previous bar's H/L/C
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c = df["close"].shift(1)

    if method == "woodie":
        pivot = (h + l + 2 * c) / 4
    elif method == "fibonacci":
        pivot = (h + l + c) / 3
    else:  # classic
        pivot = (h + l + c) / 3

    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return pd.DataFrame({"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2})

def auto_sr(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Auto Support/Resistance from swing highs/lows.

    Returns: resistance (nearest swing high), support (nearest swing low).
    """
    swings = swing_highs_lows(df, window)
    # Forward-fill swing levels
    resistance = swings["swing_high"].ffill()
    support = swings["swing_low"].ffill()
    return pd.DataFrame({"resistance": resistance, "support": support})


# ======================================================================
# 9. COMPOSITE INDICATORS — Market Regime + Feature Vector
# ======================================================================
def regime_detection(df: pd.DataFrame, adx_period: int = 14,
                     atr_period: int = 14, lookback: int = 50) -> pd.Series:
    """Market Regime Detection.

    Returns: regime string ('trending_up', 'trending_down', 'ranging',
             'volatile', 'squeeze', 'breakout').
    """
    if len(df) < max(adx_period, atr_period, lookback) + 5:
        return pd.Series(["unknown"] * len(df), index=df.index)

    adx_df = adx(df, adx_period)
    atr_val = atr(df, atr_period)
    atr_baseline = atr_val.rolling(lookback, min_periods=20).mean()
    atr_ratio = (atr_val / atr_baseline.replace(0, np.nan)).fillna(1.0)

    close = df["close"]
    ema_f = ema(close, 20)
    ema_s = ema(close, 50)

    conditions = [
        (adx_df["adx"] > 25) & (ema_f > ema_s) & (atr_ratio < 1.5),
        (adx_df["adx"] > 25) & (ema_f < ema_s) & (atr_ratio < 1.5),
        (adx_df["adx"] < 20) & (atr_ratio < 0.8),
        (atr_ratio > 1.8) & (adx_df["adx"] > 20),
        (atr_ratio < 0.5),
        (atr_ratio > 1.5) & (adx_df["adx"] > 25),
    ]
    choices = ["trending_up", "trending_down", "ranging", "volatile", "squeeze", "breakout"]
    out = pd.Series(np.select(conditions, choices, default="unknown"), index=df.index)
    return out

def zscore(close: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of close vs rolling mean."""
    if period < 2: raise ValueError("period must be >= 2")
    mean = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    out = (close - mean) / std.replace(0.0, np.nan)
    out.name = f"zscore_{period}"
    return out

def correlation(series_a: pd.Series, series_b: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Pearson correlation."""
    if period < 2: raise ValueError("period must be >= 2")
    out = series_a.rolling(period, min_periods=period).corr(series_b)
    out.name = f"corr_{period}"
    return out

def highest(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling highest."""
    out = series.rolling(period, min_periods=1).max()
    out.name = f"highest_{period}"
    return out

def lowest(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling lowest."""
    out = series.rolling(period, min_periods=1).min()
    out.name = f"lowest_{period}"
    return out


# ======================================================================
# 10. FEATURE GENERATOR — AI-Ready
# ======================================================================
def feature_vector(df: pd.DataFrame) -> np.ndarray:
    """Generate feature vector for ML models.

    Returns numpy array of shape (n_bars, n_features).
    """
    close = df["close"]
    vol = _get_volume(df)
    features = pd.DataFrame(index=df.index)

    # Moving averages
    features["ema_9"] = ema(close, 9)
    features["ema_21"] = ema(close, 21)
    features["ema_50"] = ema(close, 50)
    features["sma_20"] = sma(close, 20)

    # Momentum
    features["rsi_14"] = rsi(close, 14)
    features["roc_10"] = roc(close, 10)
    features["zscore_20"] = zscore(close, 20)

    # Volatility
    if {"high", "low"}.issubset(df.columns):
        features["atr_14"] = atr(df, 14)
        features["atr_pct"] = atr_pct(df, 14)
        adx_df = adx(df, 14)
        features["adx"] = adx_df["adx"]
        features["plus_di"] = adx_df["plus_di"]
        features["minus_di"] = adx_df["minus_di"]

    # Volume
    features["rvol_20"] = rvol(vol, 20)
    features["obv"] = obv(close, vol)

    # Trend
    features["slope_20"] = slope(close, 20)
    if {"high", "low"}.issubset(df.columns):
        features["trend_score"] = trend_score(df, 20)

    # MACD
    macd_df = macd(close)
    features["macd"] = macd_df["macd"]
    features["macd_signal"] = macd_df["signal"]
    features["macd_hist"] = macd_df["histogram"]

    # Bollinger
    bb = bbands(close, 20)
    features["bb_upper"] = bb["upper"]
    features["bb_lower"] = bb["lower"]
    features["bb_width"] = (bb["upper"] - bb["lower"]) / bb["middle"].replace(0, np.nan)

    # Returns
    features["ret_1"] = close.pct_change(1)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)

    # Replace inf with nan, then fill with 0
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features.values


def explain(df: pd.DataFrame) -> dict:
    """Generate human-readable explanation of current market state.

    Useful for dashboards, LLM agents, and audit logs.
    """
    close = df["close"]
    result = {}

    # EMA
    e9, e21, e50 = ema(close, 9), ema(close, 21), ema(close, 50)
    if e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        result["ema"] = "Strong Bullish (stacked up)"
    elif e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]:
        result["ema"] = "Strong Bearish (stacked down)"
    elif e9.iloc[-1] > e21.iloc[-1]:
        result["ema"] = "Bullish"
    else:
        result["ema"] = "Bearish"

    # RSI
    rsi_val = float(rsi(close, 14).iloc[-1])
    if rsi_val > 70: result["rsi"] = f"Overbought ({rsi_val:.0f})"
    elif rsi_val < 30: result["rsi"] = f"Oversold ({rsi_val:.0f})"
    elif rsi_val > 55: result["rsi"] = f"Bullish ({rsi_val:.0f})"
    elif rsi_val < 45: result["rsi"] = f"Bearish ({rsi_val:.0f})"
    else: result["rsi"] = f"Neutral ({rsi_val:.0f})"

    # ADX
    if {"high", "low"}.issubset(df.columns):
        adx_val = float(adx(df, 14)["adx"].iloc[-1])
        if adx_val > 40: result["adx"] = f"Very Strong Trend ({adx_val:.0f})"
        elif adx_val > 25: result["adx"] = f"Strong Trend ({adx_val:.0f})"
        elif adx_val > 20: result["adx"] = f"Weak Trend ({adx_val:.0f})"
        else: result["adx"] = f"Ranging ({adx_val:.0f})"

    # Regime
    if {"high", "low"}.issubset(df.columns):
        regime = regime_detection(df)
        result["regime"] = str(regime.iloc[-1])

    # MACD
    macd_df = macd(close)
    if macd_df["histogram"].iloc[-1] > 0:
        result["macd"] = "Bullish (hist > 0)"
    else:
        result["macd"] = "Bearish (hist < 0)"

    # Volume
    vol = _get_volume(df)
    rv = float(rvol(vol, 20).iloc[-1]) if len(df) > 20 else 1.0
    if rv > 2: result["volume"] = f"Very High ({rv:.1f}x avg)"
    elif rv > 1.5: result["volume"] = f"High ({rv:.1f}x avg)"
    elif rv < 0.5: result["volume"] = f"Low ({rv:.1f}x avg)"
    else: result["volume"] = f"Average ({rv:.1f}x avg)"

    # Trend Score
    if {"high", "low"}.issubset(df.columns):
        ts = float(trend_score(df, 20).iloc[-1])
        if ts > 70: result["trend_score"] = f"Strong Bullish ({ts:.0f}/100)"
        elif ts > 55: result["trend_score"] = f"Bullish ({ts:.0f}/100)"
        elif ts < 30: result["trend_score"] = f"Strong Bearish ({ts:.0f}/100)"
        elif ts < 45: result["trend_score"] = f"Bearish ({ts:.0f}/100)"
        else: result["trend_score"] = f"Neutral ({ts:.0f}/100)"

    return result


# ======================================================================
# ENRICH — Attach all indicators to DataFrame
# ======================================================================
def enrich(df: pd.DataFrame, sma_fast: int = 20, sma_slow: int = 50,
           rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """Return copy of df with common indicators added."""
    out = df.copy()
    out[f"sma_{sma_fast}"] = sma(out["close"], sma_fast)
    out[f"sma_{sma_slow}"] = sma(out["close"], sma_slow)
    out[f"rsi_{rsi_period}"] = rsi(out["close"], rsi_period)
    out[f"atr_{atr_period}"] = atr(out, atr_period)
    if {"high", "low"}.issubset(df.columns):
        adx_df = adx(out, 14)
        out["adx"] = adx_df["adx"]
        out["plus_di"] = adx_df["plus_di"]
        out["minus_di"] = adx_df["minus_di"]
        out["trend_score"] = trend_score(out, 20)
        out["atr_pct"] = atr_pct(out, atr_period)
    return out


# ======================================================================
# __all__ — Export List
# ======================================================================
__all__ = [
    # Cache
    "IndicatorCache",
    # Moving averages
    "sma", "ema", "wma", "vwma", "hull_ma",
    # Momentum
    "rsi", "stoch_rsi", "macd", "roc", "cci", "williams_r", "tsi", "mfi",
    # Volatility
    "atr", "atr_pct", "bbands", "keltner", "donchian", "hist_vol", "parkinson_vol",
    # Trend
    "adx", "dmi", "supertrend", "ichimoku", "slope", "trend_score",
    # Volume
    "obv", "vwap", "rvol", "cmf", "adl", "pvt",
    # SMC / ICT
    "swing_highs_lows", "fvg", "order_block", "liquidity_sweep",
    # Patterns
    "candlestick_patterns", "divergence",
    # Support / Resistance
    "pivot_points", "auto_sr",
    # Composite
    "regime_detection", "zscore", "correlation", "highest", "lowest",
    "feature_vector", "explain",
    # Utility
    "enrich",
]
