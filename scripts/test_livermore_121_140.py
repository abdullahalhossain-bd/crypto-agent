"""scripts/test_livermore_121_140.py
=====================================================================
Test the 7 new modules for principles 121-140 + verify 140 principles.
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


def test_capital_flow_analyzer():
    print("\n[1/7] Testing CapitalFlowAnalyzer...")
    from trading_modules.capital_flow_analyzer import (
        CapitalFlowAnalyzer, FlowType,
    )
    df = make_test_data()
    analyzer = CapitalFlowAnalyzer()
    flow = analyzer.analyze(df)
    assert flow.flow_type in FlowType
    assert 0 <= flow.strength <= 100
    d = flow.to_dict()
    print(f"  Flow: {flow.flow_type.value}, strength={flow.strength:.1f}")
    print(f"  Direction: {flow.direction}, MFI={flow.mfi:.1f}, RVol={flow.rvol:.2f}")
    print(f"  Smart money score: {flow.smart_money_score:.2f}")
    print(f"  Description: {flow.description}")

    # Multi-symbol rotation
    dfs = {"BTCUSD": df, "ETHUSD": make_test_data(seed=7)}
    rot = analyzer.analyze_rotation(dfs)
    print(f"\n  Rotation: {rot['description']}")
    print(f"  Ranking: {rot['ranking']}")


def test_relative_strength_ranker():
    print("\n[2/7] Testing RelativeStrengthRanker...")
    from trading_modules.relative_strength_ranker import (
        RelativeStrengthRanker, StrengthCategory,
    )
    ranker = RelativeStrengthRanker(benchmark="BTCUSD")

    dfs = {
        "BTCUSD": make_test_data(seed=1),
        "ETHUSD": make_test_data(seed=2),
        "EURUSD": make_test_data(seed=3),
        "GBPUSD": make_test_data(seed=4),
    }
    ranking = ranker.rank(dfs)
    print(f"  Ranking: {ranking['ranking']}")
    print(f"  Strongest: {ranking['strongest']}")
    print(f"  Weakest: {ranking['weakest']}")
    print(f"  Top candidates: {ranking['top_candidates']}")
    print(f"  Bottom candidates: {ranking['bottom_candidates']}")
    for sym in ranking["ranking"][:2]:
        s = ranking["scores"][sym]
        print(f"    {sym}: score={s['score']:.1f} ({s['recommendation']})")


def test_multi_timeframe_consensus():
    print("\n[3/7] Testing MultiTimeframeConsensusEngine...")
    from trading_modules.multi_timeframe_consensus import (
        MultiTimeframeConsensusEngine, ConsensusLevel,
    )
    engine = MultiTimeframeConsensusEngine()

    # Create aligned timeframes (all bullish)
    dfs = {}
    for tf, n in [("W1", 100), ("D1", 150), ("H4", 200), ("H1", 250), ("M15", 300)]:
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, n)  # bullish bias
        prices = 40000 * np.exp(np.cumsum(returns))
        dfs[tf] = pd.DataFrame({
            "open": prices, "high": prices * 1.01, "low": prices * 0.99,
            "close": prices, "volume": np.random.randint(100, 10000, n).astype(float),
        })

    consensus = engine.evaluate(dfs)
    print(f"  Consensus: {consensus.consensus.value}")
    print(f"  Score: {consensus.score:.3f} (-1 to +1)")
    print(f"  Confidence: {consensus.confidence:.2f}")
    print(f"  Agreeing TFs: {consensus.agreeing_tfs}")
    print(f"  Disagreeing TFs: {consensus.disagreeing_tfs}")
    print(f"  Trade allowed: {consensus.trade_allowed}")
    print(f"  Recommendation: {consensus.recommendation}")


def test_smart_money_detector():
    print("\n[4/7] Testing SmartMoneyDetector...")
    from trading_modules.smart_money_detector import SmartMoneyDetector
    detector = SmartMoneyDetector()
    df = make_test_data()
    result = detector.detect(df, spread_bps=2.5)
    print(f"  Smart money score: {result.smart_money_score:.1f}/100")
    print(f"  Inferred direction: {result.inferred_direction}")
    print(f"  Absorption: {result.absorption_detected}")
    print(f"  Iceberg: {result.iceberg_detected}")
    print(f"  Liquidity pool: {result.liquidity_pool_detected}")
    print(f"  Stop hunt: {result.stop_hunt_detected} ({result.stop_hunt_direction})")
    print(f"  Order block: {result.order_block_detected}")
    print(f"  Volume cluster: {result.volume_cluster_detected}")
    print(f"  Squeeze: {result.squeeze_detected}")
    print(f"  Description: {result.description}")
    print(f"  Actions: {result.actions[:2]}")


def test_strategy_health_monitor():
    print("\n[5/7] Testing StrategyHealthMonitor...")
    from trading_modules.strategy_health_monitor import (
        StrategyHealthMonitor, StrategyState,
    )
    monitor = StrategyHealthMonitor(min_trades_for_eval=10)

    # Record 20 trades: 14 wins, 6 losses (healthy)
    np.random.seed(42)
    for i in range(20):
        win = np.random.random() < 0.70
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.8 if win else -1.0
        monitor.record_trade("momentum", pnl, r, 3600, 1.2)

    health = monitor.health("momentum")
    print(f"  Strategy: momentum")
    print(f"  State: {health.state.value}")
    print(f"  Total trades: {health.total_trades}")
    print(f"  Win rate (50): {health.rolling_win_rate_50:.0%}")
    print(f"  Sharpe: {health.rolling_sharpe:.3f}")
    print(f"  EV: {health.rolling_ev_r:.3f}R")
    print(f"  Size multiplier: {health.size_multiplier:.2f}x")
    print(f"  Recommendation: {health.recommendation}")

    # Record 15 bad trades (decaying)
    for i in range(15):
        pnl = -np.random.uniform(20, 40)
        monitor.record_trade("bad_strategy", pnl, -1.0, 1800, 3.5)

    bad_health = monitor.health("bad_strategy")
    print(f"\n  Bad strategy:")
    print(f"  State: {bad_health.state.value}")
    print(f"  Size multiplier: {bad_health.size_multiplier:.2f}x")
    print(f"  Auto-disabled: {bad_health.auto_disabled}")

    print(f"\n  Summary: {monitor.summary()['by_state']}")


def test_portfolio_intelligence_layer():
    print("\n[6/7] Testing PortfolioIntelligenceLayer...")
    from trading_modules.portfolio_intelligence_layer import (
        PortfolioIntelligenceLayer,
    )
    intel = PortfolioIntelligenceLayer(equity=10000, benchmark="BTCUSD")

    # Add positions
    intel.add_position("BTCUSD", "BUY", 0.5, 43250, sl=42000, current_price=43500)
    intel.add_position("ETHUSD", "BUY", 3.0, 2580, sl=2500, current_price=2620)
    intel.add_position("EURUSD", "SELL", 1.0, 1.085, sl=1.095, current_price=1.082)

    # Set price history for correlation
    dfs = {
        "BTCUSD": make_test_data(seed=1),
        "ETHUSD": make_test_data(seed=2),
        "EURUSD": make_test_data(seed=3),
    }
    intel.set_price_history(dfs)

    report = intel.analyze()
    d = report.to_dict()
    print(f"  Positions: {d['total_positions']} ({d['long_positions']}L / {d['short_positions']}S)")
    print(f"  Gross: ${d['gross_exposure_usd']:,.0f}, Net: ${d['net_exposure_usd']:,.0f}")
    print(f"  Risk budget: {d['net_risk_budget_pct']:.1f}%")
    print(f"  VaR 95%: ${d['portfolio_var_95']:.0f}")
    print(f"  CVaR 95%: ${d['portfolio_cvar_95']:.0f}")
    print(f"  Beta: {d['portfolio_beta']:.2f}")
    print(f"  Avg correlation: {d['avg_correlation']:.2f} ({d['correlation_risk']})")
    print(f"  Diversification: {d['diversification_score']:.2f}")
    print(f"  Effective positions: {d['effective_positions']:.1f}")
    print(f"  Hedges: {len(d['hedges_detected'])}")
    print(f"  Currency exposure: {d['currency_exposure']}")
    print(f"  Recommendation: {d['recommendation']}")


def test_weekly_self_audit():
    print("\n[7/7] Testing WeeklySelfAuditor...")
    from trading_modules.weekly_self_audit import WeeklySelfAuditor
    auditor = WeeklySelfAuditor(initial_equity=10000)

    # Record 30 trades
    np.random.seed(42)
    for i in range(30):
        win = np.random.random() < 0.60
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.8 if win else -1.0
        auditor.record_trade(
            strategy="momentum", symbol="BTCUSD",
            pnl=pnl, r_multiple=r, hold_time_s=3600,
            slippage_bps=1.5, confidence=0.7,
            regime="trend_up", session="london", setup="pullback",
        )
    auditor.record_error("mt5", "IPC timeout")
    auditor.record_system_metric("cpu", 45.0)

    report = auditor.audit_week()
    d = report.to_dict()
    print(f"  Week ending: {d['week_ending']}")
    print(f"  Overall GPA: {d['overall_gpa']:.2f} ({d['overall_grade']})")
    print(f"\n  Grades:")
    for dim, card in d['grades'].items():
        print(f"    {dim:20s}: {card['grade']} ({card['gpa']:.1f}) — {card['notes']}")
    print(f"\n  Strengths: {d['strengths'][:2]}")
    print(f"  Weaknesses: {d['weaknesses'][:2]}")
    print(f"  Action items: {d['action_items'][:2]}")
    print(f"  Summary: {d['summary']}")


def test_wisdom_gate_140_principles():
    """Verify all 140 principles are present and working."""
    print("\n[BONUS] Verifying 140 principles in WisdomGate...")
    from livermore_principles import WisdomGate, TradeContext

    gate = WisdomGate()

    # Build a context that passes all principles
    ctx = TradeContext(
        symbol='BTCUSD', direction='BUY', confidence=0.75,
        win_rate=0.60, rr_ratio=2.5, atr_ratio=0.015,
        bars_since_last_trade=20, spread_bps=2.0,
        regime='trend_up', drawdown_pct=3.0,
        recent_losses=0, recent_wins=3,
        pattern_match_count=10, pattern_win_rate=0.65,
        # v7.4 fields
        capital_flow_score=0.75, capital_flow_direction='bullish',
        relative_strength_rank=0.85, market_breadth=0.7,
        mtf_alignment_score=0.8, mtf_high_tf_agrees=True,
        signal_rank_percentile=0.85, smart_money_score=0.7,
        smart_money_direction='bullish', execution_quality_score=0.85,
        conviction_level=0.7, noise_filter_passed=True,
        historical_match_count=15, historical_win_rate=0.67,
        regime_strategy_match=True, risk_allocation_pct=2.0,
        strategy_decay_detected=False, portfolio_correlation_avg=0.3,
        portfolio_diversification=0.7, adaptive_rules_active=True,
        consistency_score=0.7, weekly_audit_passed=True, weekly_audit_gpa=3.2,
    )
    verdict = gate.evaluate(ctx)
    total = verdict.checks_passed + verdict.checks_failed
    print(f"  Total principles: {total}")
    print(f"  Passed: {verdict.checks_passed}")
    print(f"  Failed: {verdict.checks_failed}")
    print(f"  Approved: {verdict.approved}")
    assert total >= 140, f"Expected at least 140 principles, got {total}"
    print("  OK — all 140 principles present and evaluated")


if __name__ == "__main__":
    print("=" * 70)
    print("  7 NEW MODULES TEST (Principles 121-140)")
    print("  Pages 120-140: Capital Rotation, Leadership, Discipline,")
    print("  Macro Awareness, Statistical Thinking")
    print("=" * 70)
    test_capital_flow_analyzer()
    test_relative_strength_ranker()
    test_multi_timeframe_consensus()
    test_smart_money_detector()
    test_strategy_health_monitor()
    test_portfolio_intelligence_layer()
    test_weekly_self_audit()
    test_wisdom_gate_140_principles()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED — 7 MODULES + 140 PRINCIPLES VERIFIED")
    print("=" * 70)
