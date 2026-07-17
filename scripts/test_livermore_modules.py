"""scripts/test_livermore_modules.py
=====================================================================
Smoke test for the 7 new Livermore operational modules.
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
    open_ = np.empty(n); close = prices
    high = np.empty(n); low = np.empty(n)
    open_[0] = prices[0]
    for i in range(1, n): open_[i] = close[i - 1]
    for i in range(n):
        body = abs(close[i] - open_[i])
        wick_up = abs(np.random.normal(0, body * 0.5 + 1))
        wick_dn = abs(np.random.normal(0, body * 0.5 + 1))
        high[i] = max(open_[i], close[i]) + wick_up
        low[i] = min(open_[i], close[i]) - wick_dn
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


def test_market_phase_detector():
    print("\n[1/7] Testing MarketPhaseDetector...")
    from trading_modules.market_phase_detector import (
        MarketPhaseDetector, MarketPhase, detect_market_phase,
    )
    df = make_test_data()
    detector = MarketPhaseDetector()
    result = detector.detect(df)
    assert result.phase in MarketPhase
    assert 0 <= result.confidence <= 1
    d = result.to_dict()
    assert "phase" in d
    assert "confidence" in d
    print(f"  OK — phase={result.phase.value}, confidence={result.confidence:.2f}")
    print(f"  Range: support={result.range_support:.0f}, resistance={result.range_resistance:.0f}")
    print(f"  Description: {result.description}")


def test_setup_scoring_engine():
    print("\n[2/7] Testing SetupScoringEngine...")
    from trading_modules.setup_scoring_engine import SetupScoringEngine
    df = make_test_data()
    engine = SetupScoringEngine(min_score=80)
    score = engine.score(
        df, signal_action="BUY",
        spread_bps=2.5, slippage_estimate_bps=1.5,
        orderbook_depth_usd=2_500_000,
        session="london", has_pullback=True,
        news_minutes=180, high_impact_news=False,
        confidence=0.75,
    )
    assert 0 <= score.total <= 100
    d = score.to_dict()
    assert "dimensions" in d
    assert "trend" in d["dimensions"]
    print(f"  Score: {score.total:.1f}/100")
    print(f"  Trend: {score.trend.score:.0f}/{score.trend.max_points:.0f} — {score.trend.detail}")
    print(f"  Volume: {score.volume.score:.0f}/{score.volume.max_points:.0f} — {score.volume.detail}")
    print(f"  Liquidity: {score.liquidity.score:.0f}/{score.liquidity.max_points:.0f}")
    print(f"  Volatility: {score.volatility.score:.0f}/{score.volatility.max_points:.0f}")
    print(f"  Timing: {score.timing.score:.0f}/{score.timing.max_points:.0f}")
    print(f"  Passed: {score.passed}, Multiplier: {score.position_multiplier}x")
    print(f"  Recommendation: {score.recommendation}")


def test_expected_value_calculator():
    print("\n[3/7] Testing ExpectedValueCalculator...")
    from trading_modules.expected_value_calculator import ExpectedValueCalculator
    calc = ExpectedValueCalculator(min_sample_size=30)

    # Good setup: 62% win rate, 1.8R avg win, 50 trades
    result = calc.calculate(
        win_rate=0.62, avg_win_r=1.8, avg_loss_r=1.0,
        sample_size=50, account_equity=10000, risk_per_trade_pct=2.0,
    )
    assert result.ev_per_trade_r > 0  # positive EV
    assert result.kelly_fraction > 0
    print(f"  Good setup (62% WR, 1.8R win, 50 trades):")
    print(f"    EV/trade: {result.ev_per_trade_r:.3f}R = ${result.ev_per_trade_usd:.2f}")
    print(f"    Kelly: {result.kelly_fraction:.1%}, Half-Kelly: {result.kelly_half:.1%}")
    print(f"    Risk of Ruin: {result.risk_of_ruin:.2%}")
    print(f"    Recommendation: {result.recommendation}")

    # Bad setup: 35% win rate
    bad = calc.calculate(win_rate=0.35, avg_win_r=1.5, sample_size=40)
    assert bad.ev_per_trade_r < 0
    print(f"\n  Bad setup (35% WR):")
    print(f"    EV/trade: {bad.ev_per_trade_r:.3f}R — {bad.recommendation}")

    # Monte Carlo
    mc = calc.monte_carlo(0.62, 1.8, 1.0, n_trades=200, n_simulations=1000)
    print(f"\n  Monte Carlo (200 trades, 1000 sims):")
    print(f"    Median final: ${mc['median_final_equity']:.0f}")
    print(f"    5th pctile: ${mc['p05_final_equity']:.0f}")
    print(f"    Ruin prob: {mc['ruin_probability']:.1%}")


def test_portfolio_exposure_analyzer():
    print("\n[4/7] Testing PortfolioExposureAnalyzer...")
    from trading_modules.portfolio_exposure_analyzer import (
        PortfolioExposureAnalyzer, classify_symbol, extract_currencies,
    )

    # Test symbol classification
    assert classify_symbol("EURUSD") == "forex"
    assert classify_symbol("BTCUSD") == "crypto"
    assert classify_symbol("XAUUSD") == "metal"
    assert classify_symbol("Volatility 75 Index") == "synthetic"
    assert extract_currencies("EURUSD") == ("EUR", "USD")
    print("  Symbol classification OK")

    analyzer = PortfolioExposureAnalyzer()
    # 3 correlated USD pairs
    analyzer.add_position("EURUSD", "BUY", 1.0, 1.0850)
    analyzer.add_position("GBPUSD", "BUY", 1.0, 1.2750)
    analyzer.add_position("AUDUSD", "BUY", 1.0, 0.6550)

    report = analyzer.analyze()
    d = report.to_dict()
    print(f"  Currency exposure: {d['currency_exposure']}")
    print(f"  Asset class: {d['asset_class_exposure']}")
    print(f"  Gross: ${d['gross_exposure_usd']:,.0f}, Net: ${d['net_exposure_usd']:,.0f}")
    print(f"  Correlation risk: {d['correlation_risk']}")
    print(f"  Effective positions: {d['effective_positions']:.1f} (actual: 3)")
    print(f"  Directional bias: {d['directional_bias']}")
    print(f"  Warnings: {d['warnings']}")
    print(f"  Recommendation: {d['recommendation']}")

    # Can we add another?
    allowed, reason = analyzer.can_add_position("NZDUSD", "BUY", 1.0, 0.6200)
    print(f"  Can add NZDUSD? {allowed} — {reason}")


def test_emotion_volatility_filter():
    print("\n[5/7] Testing EmotionVolatilityFilter...")
    from trading_modules.emotion_volatility_filter import (
        EmotionVolatilityFilter, Emotion, TradingMode,
    )

    filt = EmotionVolatilityFilter()

    # Normal market
    df = make_test_data()
    state = filt.detect(df, spread_bps=3.0, orderbook_depth_usd=2_000_000)
    d = state.to_dict()
    print(f"  Normal market: emotion={d['emotion']}, mode={d['mode']}")
    print(f"    vol_pctile={d['volatility_percentile']:.2f}, rvol={d['volume_ratio']:.2f}")
    print(f"    Description: {d['description']}")

    # Simulate panic: add a huge drop bar
    df_panic = df.copy()
    last_idx = df_panic.index[-1]
    df_panic.loc[last_idx, "close"] = df_panic.loc[last_idx, "close"] * 0.94  # -6% drop
    df_panic.loc[last_idx, "volume"] = df_panic.loc[last_idx, "volume"] * 4   # 4x volume

    panic_state = filt.detect(df_panic, spread_bps=12.0)
    print(f"\n  Panic market: emotion={panic_state.emotion.value}, mode={panic_state.mode.value}")
    print(f"    Price change: {panic_state.price_change_pct:.1f}%")
    print(f"    Volume ratio: {panic_state.volume_ratio:.1f}x")
    print(f"    Actions: {panic_state.actions[:2]}")

    # Test all emotions can be detected
    assert state.emotion in Emotion
    assert state.mode in TradingMode
    print("  OK — emotion + mode detection works")


def test_system_health_monitor():
    print("\n[6/7] Testing SystemHealthMonitor...")
    from trading_modules.system_health_monitor import (
        SystemHealthMonitor, HealthStatus,
    )

    monitor = SystemHealthMonitor()

    # Check disk
    disk = monitor.check_disk_space("/")
    print(f"  Disk: {disk.value} — {disk.message}")

    # Check memory
    mem = monitor.check_memory()
    print(f"  Memory: {mem.value} — {mem.message}")

    # Check CPU
    cpu = monitor.check_cpu()
    print(f"  CPU: {cpu.value} — {cpu.message}")

    # Check database (should work)
    import os
    os.makedirs("data", exist_ok=True)
    db = monitor.check_database("data/test_health.db")
    print(f"  Database: {db.value} — {db.message}")

    # Check network
    net = monitor.check_network()
    print(f"  Network: {net.value} — {net.message}")

    # Check clock
    clk = monitor.check_clock_drift()
    print(f"  Clock: {clk.value} — {clk.message}")

    # Overall summary
    summary = monitor.health_summary()
    print(f"\n  Overall: {summary['status'].upper()}")
    print(f"  Can trade: {summary['can_trade']}")
    print(f"  Emergency: {summary['emergency']}")
    print(f"  Failed: {summary['failed_components']}")

    # Cleanup
    try:
        os.remove("data/test_health.db")
    except Exception:
        pass


def test_adaptive_learning_engine():
    print("\n[7/7] Testing AdaptiveLearningEngine...")
    from trading_modules.adaptive_learning_engine import AdaptiveLearningEngine

    engine = AdaptiveLearningEngine(min_trades_for_weight=5)

    # Record 20 trades: momentum in london + trend_up = good (12W, 8L, avg_win 1.8R)
    np.random.seed(42)
    for i in range(20):
        win = np.random.random() < 0.60  # 60% win rate
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.8 if win else -1.0
        engine.record_outcome(
            strategy="momentum", session="london", regime="trend_up",
            setup="pullback", symbol="BTCUSD",
            pnl=pnl, r_multiple=r, confidence=0.65,
            hold_time_s=3600,
        )

    # Record 10 trades: mean_reversion in range = bad (3W, 7L)
    for i in range(10):
        win = np.random.random() < 0.30  # 30% win rate
        pnl = np.random.uniform(20, 50) if win else -np.random.uniform(25, 45)
        r = 1.5 if win else -1.0
        engine.record_outcome(
            strategy="mean_reversion", session="asia", regime="range",
            setup="reversal", symbol="ETHUSD",
            pnl=pnl, r_multiple=r, confidence=0.60,
            hold_time_s=1800,
        )

    # Get weights for a momentum pullback in london trend_up
    weights = engine.get_weights(
        strategy="momentum", session="london",
        regime="trend_up", setup="pullback", symbol="BTCUSD",
    )
    print(f"  Weights for momentum/london/trend_up/pullback/BTCUSD:")
    for k, v in weights.items():
        print(f"    {k:30s}: {v:.2f}")

    # Get weights for mean_reversion
    weights2 = engine.get_weights(
        strategy="mean_reversion", session="asia",
        regime="range", setup="reversal", symbol="ETHUSD",
    )
    print(f"\n  Weights for mean_reversion/asia/range/reversal/ETHUSD:")
    for k, v in weights2.items():
        print(f"    {k:30s}: {v:.2f}")

    # Rankings
    print(f"\n  Strategy ranking:")
    for name, s in engine.rank_strategies():
        print(f"    {name:20s} trades={s.trades} WR={s.win_rate:.0%} EV={s.ev_per_trade:+.3f}R weight={s.weight:.2f}x")

    # Avoid list
    avoid = engine.avoid_list(min_trades=5)
    print(f"\n  Avoid list: {avoid}")

    # Full stats table
    print()
    print(engine.stats_table())


if __name__ == "__main__":
    print("=" * 70)
    print("  7 NEW LIVERMORE OPERATIONAL MODULES TEST")
    print("  (Principles 101-120: Market Timing, Capital Preservation,")
    print("   Continuous Adaptation, Complete Trading Intelligence)")
    print("=" * 70)
    test_market_phase_detector()
    test_setup_scoring_engine()
    test_expected_value_calculator()
    test_portfolio_exposure_analyzer()
    test_emotion_volatility_filter()
    test_system_health_monitor()
    test_adaptive_learning_engine()
    print("\n" + "=" * 70)
    print("  ALL 7 MODULE TESTS PASSED")
    print("=" * 70)
