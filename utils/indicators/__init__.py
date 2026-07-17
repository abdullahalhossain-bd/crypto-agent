"""utils/indicators/__init__.py
=====================================================================
Modular Indicator Library — Industrial Grade
=====================================================================
Replaces the monolithic utils/indicators.py with a modular package:

    indicators/
    ├── trend.py          (13 indicators: EMA, SMA, WMA, HMA, VWMA, DEMA, TEMA,
    │                          ZLEMA, KAMA, ALMA, T3, SuperTrend, Ichimoku)
    ├── momentum.py       (13 indicators: RSI, StochRSI, Stochastic, MACD, PPO,
    │                          ROC, Momentum, TRIX, TSI, CCI, Williams %R,
    │                          Ultimate Osc, Awesome Osc)
    ├── volatility.py     (11 indicators: ATR, NATR, BB, BB Width, BB %B,
    │                          Keltner, Donchian, Chaikin Vol, StdDev, HV, Parkinson)
    ├── volume.py         (12 indicators: OBV, VWAP, CMF, MFI, ADL, EOM,
    │                          Vol Osc, Force Index, NVI, PVI, PVT, RVol)
    ├── structure.py      (7 indicators: Swing H/L, HH/HL/LH/LL, BoS, ChoCH,
    │                          MSS, Liquidity Sweep)
    ├── smc.py            (8 indicators: FVG, Order Block, Breaker, Mitigation,
    │                          Premium/Discount, Equal H/L, Liquidity Pool, Imbalance)
    ├── candles.py        (11 patterns: Doji, Hammer, Hanging Man, Shooting Star,
    │                          Bullish/Bearish Engulfing, Morning/Evening Star,
    │                          Harami, 3 White Soldiers, 3 Black Crows)
    ├── statistics.py     (10 indicators: Z-Score, Rolling Mean/Std, Skewness,
    │                          Kurtosis, Entropy, Variance, Autocorrelation,
    │                          Hurst, Stationarity, Correlation, Beta)
    ├── features.py       (14 AI features + confidence scores)
    ├── regime.py         (Volatility + Trend regime detection)
    ├── validation.py     (Data quality checks: NaN, Inf, duplicates, outliers)
    ├── caching.py        (LRU cache + incremental updates)
    ├── diagnostics.py    (Performance + quality metrics)
    ├── registry.py       (IndicatorEngine + IndicatorResult + batch calculate_all)
    └── __init__.py       (this file — public API)

Total: 110+ indicators across 14 modules

Backward Compatibility:
    All function names from the old utils/indicators.py are re-exported here.
    Existing code that does `from utils.indicators import rsi, macd, ema`
    will continue to work without modification.
=====================================================================
"""
from __future__ import annotations

# === Trend (13 + 6 utility = 19 functions) ===
from utils.indicators.trend import (
    sma, ema, wma, vwma, hull_ma, dema, tema, zlema, kama, alma, t3_ma,
    supertrend, ichimoku,
    # Utility functions (kept for backward compat)
    adx, dmi, slope, highest, lowest, trend_score,
)

# === Momentum (13 indicators) ===
from utils.indicators.momentum import (
    rsi, stoch_rsi, stochastic, macd, ppo, roc, momentum, trix, tsi,
    cci, williams_r, ultimate_oscillator, awesome_oscillator,
)

# === Volatility (11 indicators) ===
from utils.indicators.volatility import (
    atr, atr_pct, natr, bollinger_bands, bollinger_width, bollinger_pct_b,
    keltner_channel, donchian_channel, chaikin_volatility, stddev,
    historical_volatility, parkinson_vol,
)

# === Volume (12 indicators) ===
from utils.indicators.volume import (
    obv, vwap, cmf, mfi, adl, ease_of_movement, volume_oscillator,
    force_index, negative_volume_index, positive_volume_index, pvt, rvol,
)

# === Market Structure (7 indicators) ===
from utils.indicators.structure import (
    swing_highs_lows, market_structure, break_of_structure,
    change_of_character, market_structure_shift, liquidity_sweep,
    swing_points,
)

# === Smart Money Concepts (8 indicators) ===
from utils.indicators.smc import (
    detect_fvg, detect_order_block, detect_breaker_block,
    detect_mitigation_block, premium_discount_zone,
    detect_equal_highs_lows, detect_liquidity_pool, detect_imbalance,
)

# === Candlestick Patterns (11 + helper) ===
from utils.indicators.candles import (
    detect_doji, detect_hammer, detect_hanging_man, detect_shooting_star,
    detect_bullish_engulfing, detect_bearish_engulfing,
    detect_morning_star, detect_evening_star, detect_harami,
    detect_three_white_soldiers, detect_three_black_crows,
    detect_all_patterns,
)

# === Statistics (10 indicators) ===
from utils.indicators.statistics import (
    zscore, rolling_mean, rolling_std, rolling_variance, skewness,
    kurtosis, entropy, autocorrelation, hurst_exponent,
    stationarity_score, correlation, beta,
)

# === AI Features (14 + confidence + feature_vector) ===
from utils.indicators.features import (
    ema_distance, price_position, rsi_normalized, atr_percentage,
    bb_width, volume_ratio, momentum_score, trend_score as feature_trend_score,
    volatility_score, candle_body_pct, upper_wick_pct, lower_wick_pct,
    daily_range_pct, gap_pct, feature_vector, confidence_scores,
)

# === Regime Detection ===
from utils.indicators.regime import (
    volatility_regime, trend_regime, regime_detection,
    VolatilityRegime, TrendRegime,
)

# === Validation ===
from utils.indicators.validation import (
    validate_ohlcv, clean_ohlcv, assert_valid,
    ValidationReport,
)

# === Caching ===
from utils.indicators.caching import (
    IndicatorCache, cached, get_global_cache, IncrementalIndicator,
)

# === Diagnostics ===
from utils.indicators.diagnostics import (
    Diagnostics, IndicatorMetric, get_diagnostics,
)

# === Registry + Engine ===
from utils.indicators.registry import (
    IndicatorRegistry, IndicatorEngine, IndicatorResult,
    FeatureMetadata, RiskFeatures,
)


# ----------------------------------------------------------------------
# Backward-compatibility shims (old API names → new locations)
# ----------------------------------------------------------------------
# Old name was bbands; new name is bollinger_bands
def bbands(close, period=20, std_dev=2.0):
    """Backward compat: alias for bollinger_bands."""
    return bollinger_bands(close, period, std_dev)


# Old name was hist_vol; new is historical_volatility
def hist_vol(close, period=20, annualize=True):
    """Backward compat: alias for historical_volatility."""
    return historical_volatility(close, period, annualize)


# Old name was parkinson_vol; already exposed
def parkinson_volatility(df, period=20):
    """Backward compat: alias for parkinson_vol."""
    return parkinson_vol(df, period)


# Old name was donchian; new is donchian_channel
def donchian(df, period=20):
    """Backward compat: alias for donchian_channel."""
    return donchian_channel(df, period)


# Old name was keltner; new is keltner_channel
def keltner(df, period=20, multiplier=2.0):
    """Backward compat: alias for keltner_channel."""
    return keltner_channel(df, period, multiplier)


# Old name was fvg; new is detect_fvg
def fvg(df):
    """Backward compat: alias for detect_fvg."""
    return detect_fvg(df)


# Old name was order_block; new is detect_order_block
def order_block(df, lookback=10):
    """Backward compat: alias for detect_order_block."""
    return detect_order_block(df, lookback)


# Old name was liquidity_sweep; already exposed
# Old name was swing_highs_lows; already exposed

# Old name was candlestick_patterns; new is detect_all_patterns
def candlestick_patterns(df):
    """Backward compat: alias for detect_all_patterns."""
    return detect_all_patterns(df)


# Old name was divergence; provide a simple implementation
def divergence(close: pd.Series, indicator: pd.Series, period: int = 20):
    """Detect divergence between price and an indicator.

    Bullish divergence: price makes lower low, indicator makes higher low
    Bearish divergence: price makes higher high, indicator makes lower high
    """
    import pandas as pd
    import numpy as np
    price_low = close.rolling(period).min()
    price_high = close.rolling(period).max()
    ind_low = indicator.rolling(period).min()
    ind_high = indicator.rolling(period).max()
    bullish = (close < price_low.shift(period)) & (indicator > ind_low.shift(period))
    bearish = (close > price_high.shift(period)) & (indicator < ind_high.shift(period))
    return pd.DataFrame({"bullish_div": bullish, "bearish_div": bearish})


# Old name was enrich; provide wrapper around feature_vector
def enrich(df: pd.DataFrame, period: int = 20):
    """Backward compat: enrich a DataFrame with AI features (returns dict)."""
    return feature_vector(df, period)


# Old name was explain; provide a simple explainer
def explain(df: pd.DataFrame, period: int = 20):
    """Backward compat: return a human-readable explanation of the market state."""
    try:
        fv = feature_vector(df, period)
        lines = [
            f"Market State Explanation for {len(df)} bars:",
            f"  Trend Score: {fv['trend_score']:.3f} ({'bullish' if fv['trend_score'] > 0 else 'bearish'})",
            f"  Momentum: {fv['momentum_score']:.3f}",
            f"  Volatility: {fv['atr_pct']:.3f}% ({'high' if fv['atr_pct'] > 3 else 'low'})",
            f"  RSI: {fv['rsi_normalized']:.3f} ({'overbought' if fv['rsi_normalized'] > 0.4 else 'oversold' if fv['rsi_normalized'] < -0.4 else 'neutral'})",
            f"  Volume: {fv['volume_ratio']:.2f}x avg",
            f"  Price Position: {fv['price_position']:.2f}",
            f"  Candle Body: {fv['candle_body_pct']:.2f}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"explain failed: {e}"


# Old name was pivot_points
def pivot_points(df: pd.DataFrame):
    """Compute classic pivot points (P, R1, R2, S1, S2)."""
    import pandas as pd
    if df is None or df.empty:
        return pd.DataFrame()
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    close = df["close"].shift(1)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pd.DataFrame({"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2})


# Old name was auto_sr (auto support/resistance)
def auto_sr(df: pd.DataFrame, lookback: int = 50, n_levels: int = 5):
    """Auto-detect support and resistance levels."""
    swings = swing_highs_lows(df, lookback=5)
    sh_prices = swings["swing_high_price"].dropna().tail(n_levels).tolist()
    sl_prices = swings["swing_low_price"].dropna().tail(n_levels).tolist()
    return {"resistance": sh_prices, "support": sl_prices}


# Old name was IndicatorCache (already exposed)
# Re-export the legacy alias for the cache class


# ----------------------------------------------------------------------
# Convenience: build a default IndicatorEngine
# ----------------------------------------------------------------------
def get_engine() -> IndicatorEngine:
    """Get a singleton IndicatorEngine with all 110+ indicators registered."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = IndicatorEngine()
    return _ENGINE


_ENGINE: IndicatorEngine | None = None


# ----------------------------------------------------------------------
# Module-level imports for pandas (used in shims above)
# ----------------------------------------------------------------------
import pandas as pd  # noqa: E402
import numpy as np   # noqa: E402


# ----------------------------------------------------------------------
# Public API summary
# ----------------------------------------------------------------------
__all__ = [
    # Trend
    "sma", "ema", "wma", "vwma", "hull_ma", "dema", "tema", "zlema",
    "kama", "alma", "t3_ma", "supertrend", "ichimoku",
    "adx", "dmi", "slope", "highest", "lowest", "trend_score",
    # Momentum
    "rsi", "stoch_rsi", "stochastic", "macd", "ppo", "roc", "momentum",
    "trix", "tsi", "cci", "williams_r", "ultimate_oscillator",
    "awesome_oscillator",
    # Volatility
    "atr", "atr_pct", "natr", "bollinger_bands", "bollinger_width",
    "bollinger_pct_b", "keltner_channel", "donchian_channel",
    "chaikin_volatility", "stddev", "historical_volatility", "parkinson_vol",
    # Volume
    "obv", "vwap", "cmf", "mfi", "adl", "ease_of_movement",
    "volume_oscillator", "force_index", "negative_volume_index",
    "positive_volume_index", "pvt", "rvol",
    # Structure
    "swing_highs_lows", "market_structure", "break_of_structure",
    "change_of_character", "market_structure_shift", "liquidity_sweep",
    "swing_points",
    # SMC
    "detect_fvg", "detect_order_block", "detect_breaker_block",
    "detect_mitigation_block", "premium_discount_zone",
    "detect_equal_highs_lows", "detect_liquidity_pool", "detect_imbalance",
    # Candles
    "detect_doji", "detect_hammer", "detect_hanging_man",
    "detect_shooting_star", "detect_bullish_engulfing",
    "detect_bearish_engulfing", "detect_morning_star", "detect_evening_star",
    "detect_harami", "detect_three_white_soldiers",
    "detect_three_black_crows", "detect_all_patterns",
    # Statistics
    "zscore", "rolling_mean", "rolling_std", "rolling_variance",
    "skewness", "kurtosis", "entropy", "autocorrelation",
    "hurst_exponent", "stationarity_score", "correlation", "beta",
    # AI Features
    "ema_distance", "price_position", "rsi_normalized", "atr_percentage",
    "bb_width", "volume_ratio", "momentum_score", "volatility_score",
    "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
    "daily_range_pct", "gap_pct", "feature_vector", "confidence_scores",
    # Regime
    "volatility_regime", "trend_regime", "regime_detection",
    "VolatilityRegime", "TrendRegime",
    # Validation
    "validate_ohlcv", "clean_ohlcv", "assert_valid", "ValidationReport",
    # Caching
    "IndicatorCache", "cached", "get_global_cache", "IncrementalIndicator",
    # Diagnostics
    "Diagnostics", "IndicatorMetric", "get_diagnostics",
    # Registry + Engine
    "IndicatorRegistry", "IndicatorEngine", "IndicatorResult",
    "FeatureMetadata", "RiskFeatures", "get_engine",
    # Backward compat shims
    "bbands", "hist_vol", "parkinson_volatility", "donchian", "keltner",
    "fvg", "order_block", "candlestick_patterns", "divergence",
    "enrich", "explain", "pivot_points", "auto_sr",
]
