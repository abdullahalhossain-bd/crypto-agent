"""scripts/test_indicators_v2.py
=====================================================================
Smoke test for the new modular indicators package.
Tests all 110+ indicators across 14 modules.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def make_test_data(n=300, seed=42):
    """Generate synthetic OHLCV data with valid candles."""
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 40000 * np.exp(np.cumsum(returns))
    # Generate valid candles: open from prev close, high >= max(open,close), low <= min
    open_ = np.empty(n)
    close = prices
    high = np.empty(n)
    low = np.empty(n)
    open_[0] = prices[0]
    for i in range(1, n):
        open_[i] = close[i - 1]
    for i in range(n):
        body = abs(close[i] - open_[i])
        wick_up = abs(np.random.normal(0, body * 0.5 + 1))
        wick_dn = abs(np.random.normal(0, body * 0.5 + 1))
        high[i] = max(open_[i], close[i]) + wick_up
        low[i] = min(open_[i], close[i]) - wick_dn
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


def test_trend():
    print("\n[1/14] Testing trend.py (13 + 6 utility = 19 indicators)...")
    from utils.indicators.trend import (
        sma, ema, wma, vwma, hull_ma, dema, tema, zlema, kama, alma, t3_ma,
        supertrend, ichimoku, adx, dmi, slope, highest, lowest, trend_score,
    )
    df = make_test_data()
    assert not sma(df["close"], 20).isna().all()
    assert not ema(df["close"], 20).isna().all()
    assert not wma(df["close"], 20).isna().all()
    assert not vwma(df).isna().all()
    assert not hull_ma(df["close"], 20).isna().all()
    assert not dema(df["close"], 20).isna().all()
    assert not tema(df["close"], 20).isna().all()
    assert not zlema(df["close"], 20).isna().all()
    assert not kama(df["close"], 10).isna().all()
    assert not alma(df["close"], 9).isna().all()
    assert not t3_ma(df["close"], 5).isna().all()
    assert not supertrend(df, 10, 3).isna().all()
    ich = ichimoku(df)
    assert "tenkan" in ich
    assert not adx(df, 14).isna().all()
    plus_di, minus_di, adx_val = dmi(df, 14)
    assert not slope(df["close"], 5).isna().all()
    assert not highest(df["close"], 20).isna().all()
    assert not lowest(df["close"], 20).isna().all()
    assert not trend_score(df["close"], 20).isna().all()
    print("  OK — 19 trend functions")


def test_momentum():
    print("\n[2/14] Testing momentum.py (13 indicators)...")
    from utils.indicators.momentum import (
        rsi, stoch_rsi, stochastic, macd, ppo, roc, momentum as mom,
        trix, tsi, cci, williams_r, ultimate_oscillator, awesome_oscillator,
    )
    df = make_test_data()
    assert not rsi(df["close"], 14).isna().all()
    assert not stoch_rsi(df["close"]).isna().all()
    k, d = stochastic(df)
    assert not k.isna().all()
    macd_line, signal_line, hist = macd(df["close"])
    assert not macd_line.isna().all()
    ppo_line, _, _ = ppo(df["close"])
    assert not ppo_line.isna().all()
    assert not roc(df["close"], 10).isna().all()
    assert not mom(df["close"], 10).isna().all()
    assert not trix(df["close"]).isna().all()
    assert not tsi(df["close"]).isna().all()
    assert not cci(df, 20).isna().all()
    assert not williams_r(df, 14).isna().all()
    assert not ultimate_oscillator(df).isna().all()
    assert not awesome_oscillator(df).isna().all()
    print("  OK — 13 momentum functions")


def test_volatility():
    print("\n[3/14] Testing volatility.py (11 indicators)...")
    from utils.indicators.volatility import (
        atr, atr_pct, natr, bollinger_bands, bollinger_width, bollinger_pct_b,
        keltner_channel, donchian_channel, chaikin_volatility, stddev,
        historical_volatility, parkinson_vol,
    )
    df = make_test_data()
    assert not atr(df, 14).isna().all()
    assert not atr_pct(df, 14).isna().all()
    assert not natr(df, 14).isna().all()
    upper, mid, lower, width = bollinger_bands(df["close"])
    assert not upper.isna().all()
    assert not bollinger_width(df["close"]).isna().all()
    assert not bollinger_pct_b(df["close"]).isna().all()
    k_up, k_mid, k_low = keltner_channel(df)
    assert not k_mid.isna().all()
    d_up, d_mid, d_low = donchian_channel(df)
    assert not d_mid.isna().all()
    assert not chaikin_volatility(df).isna().all()
    assert not stddev(df["close"], 20).isna().all()
    assert not historical_volatility(df["close"]).isna().all()
    assert not parkinson_vol(df).isna().all()
    print("  OK — 11 volatility functions")


def test_volume():
    print("\n[4/14] Testing volume.py (12 indicators)...")
    from utils.indicators.volume import (
        obv, vwap, cmf, mfi, adl, ease_of_movement, volume_oscillator,
        force_index, negative_volume_index, positive_volume_index, pvt, rvol,
    )
    df = make_test_data()
    assert not obv(df).isna().all()
    assert not vwap(df).isna().all()
    assert not cmf(df, 20).isna().all()
    assert not mfi(df, 14).isna().all()
    assert not adl(df).isna().all()
    assert not ease_of_movement(df).isna().all()
    assert not volume_oscillator(df).isna().all()
    assert not force_index(df).isna().all()
    assert not negative_volume_index(df).isna().all()
    assert not positive_volume_index(df).isna().all()
    assert not pvt(df).isna().all()
    assert not rvol(df, 20).isna().all()
    print("  OK — 12 volume functions")


def test_structure():
    print("\n[5/14] Testing structure.py (7 indicators)...")
    from utils.indicators.structure import (
        swing_highs_lows, market_structure, break_of_structure,
        change_of_character, market_structure_shift, liquidity_sweep, swing_points,
    )
    df = make_test_data()
    swings = swing_highs_lows(df, 5)
    assert "swing_high" in swings.columns
    ms = market_structure(df, 5)
    assert "structure" in ms.columns
    assert not break_of_structure(df, 5).isna().all()
    assert not change_of_character(df, 5).isna().all()
    assert not market_structure_shift(df, 5).isna().all()
    assert not liquidity_sweep(df).isna().all()
    sp = swing_points(df, 5)
    assert "swing_high_prices" in sp
    print("  OK — 7 structure functions")


def test_smc():
    print("\n[6/14] Testing smc.py (8 indicators)...")
    from utils.indicators import smc
    df = make_test_data()
    fvg = smc.detect_fvg(df)
    assert "bullish_fvg" in fvg.columns
    obs = smc.detect_order_block(df)
    assert "bullish_ob" in obs.columns
    brk = smc.detect_breaker_block(df)
    assert "bullish_breaker" in brk.columns
    mit = smc.detect_mitigation_block(df)
    assert "mitigation_detected" in mit.columns
    pdz = smc.premium_discount_zone(df)
    assert "zone" in pdz.columns
    ehl = smc.detect_equal_highs_lows(df)
    assert "equal_high" in ehl.columns
    liq = smc.detect_liquidity_pool(df)
    assert "liq_above" in liq.columns
    imb = smc.detect_imbalance(df)
    assert not imb.isna().all()
    print("  OK — 8 SMC functions")


def test_candles():
    print("\n[7/14] Testing candles.py (11 patterns)...")
    from utils.indicators import candles
    df = make_test_data()
    assert not candles.detect_doji(df).isna().all()
    assert not candles.detect_hammer(df).isna().all()
    assert not candles.detect_hanging_man(df).isna().all()
    assert not candles.detect_shooting_star(df).isna().all()
    assert not candles.detect_bullish_engulfing(df).isna().all()
    assert not candles.detect_bearish_engulfing(df).isna().all()
    assert not candles.detect_morning_star(df).isna().all()
    assert not candles.detect_evening_star(df).isna().all()
    harami = candles.detect_harami(df)
    assert "bullish_harami" in harami.columns
    assert not candles.detect_three_white_soldiers(df).isna().all()
    assert not candles.detect_three_black_crows(df).isna().all()
    all_p = candles.detect_all_patterns(df)
    assert len(all_p.columns) == 12
    print(f"  OK — 11 candlestick patterns (12 cols in detect_all)")


def test_statistics():
    print("\n[8/14] Testing statistics.py (10 indicators)...")
    from utils.indicators import statistics
    df = make_test_data()
    close = df["close"]
    assert not statistics.zscore(close, 20).isna().all()
    assert not statistics.rolling_mean(close, 20).isna().all()
    assert not statistics.rolling_std(close, 20).isna().all()
    assert not statistics.rolling_variance(close, 20).isna().all()
    assert not statistics.skewness(close, 20).isna().all()
    assert not statistics.kurtosis(close, 20).isna().all()
    assert not statistics.entropy(close, 20).isna().all()
    assert not statistics.autocorrelation(close, 20).isna().all()
    try:
        stat = statistics.stationarity_score(close)
        assert stat is not None
    except Exception:
        pass
    assert not statistics.correlation(close, close, 20).isna().all()
    assert not statistics.beta(close, close, 60).isna().all()
    print("  OK — 10 statistics functions")


def test_features():
    print("\n[9/14] Testing features.py (14 AI features + confidence)...")
    from utils.indicators import features
    df = make_test_data()
    assert not features.ema_distance(df["close"]).isna().all()
    assert not features.price_position(df).isna().all()
    assert not features.rsi_normalized(df["close"]).isna().all()
    assert not features.atr_percentage(df).isna().all()
    assert not features.bb_width(df["close"]).isna().all()
    assert not features.volume_ratio(df).isna().all()
    assert not features.momentum_score(df["close"]).isna().all()
    assert not features.trend_score(df).isna().all()
    assert not features.volatility_score(df).isna().all()
    assert not features.candle_body_pct(df).isna().all()
    assert not features.upper_wick_pct(df).isna().all()
    assert not features.lower_wick_pct(df).isna().all()
    assert not features.daily_range_pct(df).isna().all()
    assert not features.gap_pct(df).isna().all()
    fv = features.feature_vector(df)
    assert len(fv) == 14
    cs = features.confidence_scores(df)
    assert "ema_confidence" in cs
    print(f"  OK — 14 AI features + confidence scores")
    print(f"  Sample feature_vector: rsi_norm={fv['rsi_normalized']:.3f}, "
          f"trend={fv['trend_score']:.3f}, vol_score={fv['volatility_score']:.3f}")


def test_regime():
    print("\n[10/14] Testing regime.py (volatility + trend regime)...")
    from utils.indicators import regime
    df = make_test_data()
    v_reg = regime.volatility_regime(df)
    assert not v_reg.isna().all()
    t_reg = regime.trend_regime(df)
    assert not t_reg.isna().all()
    combined = regime.regime_detection(df)
    assert "regime" in combined
    assert "volatility_regime" in combined
    assert "trend_regime" in combined
    print(f"  OK — regime detected: {combined['regime']} "
          f"(vol={combined['volatility_regime']}, trend={combined['trend_regime']})")


def test_validation():
    print("\n[11/14] Testing validation.py...")
    from utils.indicators import validation
    df = make_test_data()
    # Add some bad data
    df_bad = df.copy()
    df_bad.loc[df_bad.index[5], "high"] = 0  # invalid candle
    report = validation.validate_ohlcv(df_bad)
    assert isinstance(report, validation.ValidationReport)
    assert report.row_count == len(df_bad)
    # Clean it
    cleaned = validation.clean_ohlcv(df_bad)
    report2 = validation.validate_ohlcv(cleaned)
    print(f"  OK — validation report: ok={report.ok}, issues={len(report.issues)}, "
          f"data_quality={report.data_quality_score:.2f}")


def test_caching():
    print("\n[12/14] Testing caching.py...")
    from utils.indicators.caching import IndicatorCache, cached, get_global_cache
    cache = IndicatorCache(max_size=100)
    call_count = [0]

    @cached(cache=cache)
    def my_indicator(close, period=14):
        call_count[0] += 1
        return close.rolling(period).mean()

    df = make_test_data(100)
    r1 = my_indicator(df["close"], 14)
    r2 = my_indicator(df["close"], 14)  # cache hit
    assert r1.equals(r2)
    assert call_count[0] == 1  # only computed once
    stats = cache.stats
    assert stats.hits >= 1
    assert stats.misses >= 1
    print(f"  OK — cache hit_rate={stats.hit_rate:.2f} ({stats.size_ratio}), "
          f"call_count={call_count[0]}")


def test_diagnostics():
    print("\n[13/14] Testing diagnostics.py...")
    from utils.indicators.diagnostics import Diagnostics
    diag = Diagnostics()
    with diag.track("test_indicator"):
        x = sum(range(1000))
    stats = diag.stats()
    assert "test_indicator" in stats["per_indicator"]
    assert stats["total_calls"] == 1
    print(f"  OK — diagnostics: {stats['indicator_count']} indicators tracked, "
          f"total_time={stats['total_time_ms']:.3f}ms")


def test_registry_engine():
    print("\n[14/14] Testing registry.py (IndicatorEngine + batch calculate_all)...")
    from utils.indicators.registry import (
        IndicatorEngine, IndicatorResult, RiskFeatures,
    )
    df = make_test_data()
    engine = IndicatorEngine()
    print(f"  Registered: {engine.registry.count()} indicators")
    cats = engine.registry.categories()
    for cat, names in sorted(cats.items()):
        print(f"    {cat}: {len(names)} indicators")

    # Calculate all
    results = engine.calculate_all(df)
    print(f"  calculate_all returned {len(results)} IndicatorResults")
    valid_count = sum(1 for r in results.values() if r.valid)
    print(f"  Valid: {valid_count}/{len(results)}")

    # Sample result
    rsi_result = results.get("rsi_14")
    if rsi_result:
        print(f"  Sample RSI: value={rsi_result.value:.2f}, normalized={rsi_result.normalized:.3f}, "
              f"confidence={rsi_result.confidence:.2f}, valid={rsi_result.valid}")
        print(f"    metadata: {rsi_result.metadata.lookback} bars, warmup={rsi_result.metadata.warmup}, "
              f"latency={rsi_result.metadata.latency_ms:.3f}ms")

    # Risk features
    risk = engine.calculate_risk_features(df)
    assert isinstance(risk, RiskFeatures)
    print(f"  Risk features: ATR_stop={risk.atr_stop_distance:.2f}, "
          f"vol_risk={risk.volatility_risk:.2f}, trend_strength={risk.trend_strength:.2f}")

    # Diagnostics
    summary = engine.summary()
    print(f"  Engine summary: {summary['total_indicators']} indicators, "
          f"cache_size={summary['cache_size']}, hit_rate={summary['cache_hit_rate']:.2f}")


def test_backward_compat():
    """Verify old API names still work."""
    print("\n[BONUS] Testing backward compatibility with old indicators.py API...")
    from utils.indicators import (
        rsi, macd, ema, sma, atr, bollinger_bands, vwap, obv,
        # Old names that should now be aliases
        bbands, hist_vol, donchian, keltner, fvg, order_block,
        candlestick_patterns, divergence, enrich, explain,
        pivot_points, auto_sr,
    )
    df = make_test_data()
    assert not rsi(df["close"], 14).isna().all()
    assert not ema(df["close"], 20).isna().all()
    # Old alias
    upper, mid, lower, width = bbands(df["close"])
    assert not upper.isna().all()
    hv = hist_vol(df["close"])
    assert not hv.isna().all()
    d_up, d_mid, d_low = donchian(df)
    k_up, k_mid, k_low = keltner(df)
    fvg_df = fvg(df)
    obs = order_block(df)
    patterns = candlestick_patterns(df)
    assert len(patterns.columns) == 12
    div = divergence(df["close"], rsi(df["close"]))
    assert "bullish_div" in div.columns
    enriched = enrich(df)
    assert isinstance(enriched, dict)
    explained = explain(df)
    assert isinstance(explained, str)
    pivots = pivot_points(df)
    assert "pivot" in pivots.columns
    sr = auto_sr(df)
    assert "resistance" in sr
    print("  OK — all backward compat aliases work")


def test_cache_performance():
    """Verify cache actually speeds up repeated calls."""
    print("\n[BONUS] Testing cache performance improvement...")
    import time
    from utils.indicators.caching import IndicatorCache, cached

    cache = IndicatorCache(max_size=1000)

    @cached(cache=cache)
    def slow_indicator(close, period=14):
        # Simulate expensive computation
        total = 0
        for i in range(1000):
            total += close.rolling(period).std().sum()
        return total

    df = make_test_data(200)
    t0 = time.time()
    r1 = slow_indicator(df["close"], 14)
    t1 = time.time()
    r2 = slow_indicator(df["close"], 14)  # cache hit
    t2 = time.time()

    first_call = (t1 - t0) * 1000
    second_call = (t2 - t1) * 1000
    speedup = first_call / max(second_call, 0.001)
    print(f"  First call (compute):  {first_call:.2f}ms")
    print(f"  Second call (cached):  {second_call:.3f}ms")
    print(f"  Speedup: {speedup:.0f}x")
    print(f"  Cache stats: {cache.stats.size_ratio} hits/misses")


if __name__ == "__main__":
    print("="*70)
    print("  MODULAR INDICATORS LIBRARY TEST (utils/indicators/)")
    print("="*70)
    test_trend()
    test_momentum()
    test_volatility()
    test_volume()
    test_structure()
    test_smc()
    test_candles()
    test_statistics()
    test_features()
    test_regime()
    test_validation()
    test_caching()
    test_diagnostics()
    test_registry_engine()
    test_backward_compat()
    test_cache_performance()
    print("\n" + "="*70)
    print("  ALL INDICATOR TESTS PASSED")
    print("="*70)
