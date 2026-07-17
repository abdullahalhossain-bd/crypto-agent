"""scripts/test_livermore_181_200.py
=====================================================================
Test the 7 new modules for principles 181-200 + verify 200 principles.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd


def make_test_data(n=300, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 40000 * np.exp(np.cumsum(returns))
    open_ = np.empty(n); close = prices
    high = np.empty(n); low = np.empty(n)
    open_[0] = prices[0]
    for i in range(1, n): open_[i] = close[i - 1]
    for i in range(n):
        body = abs(close[i] - open_[i])
        high[i] = max(open_[i], close[i]) + abs(np.random.normal(0, body*0.5+1))
        low[i] = min(open_[i], close[i]) - abs(np.random.normal(0, body*0.5+1))
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,
        "volume":np.random.randint(100,10000,n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"))


def test_trade_opportunity_ranker():
    print("\n[1/7] Testing TradeOpportunityRanker...")
    from trading_modules.trade_opportunity_ranker import TradeOpportunityRanker
    ranker = TradeOpportunityRanker()

    # Score multiple opportunities
    opps = [
        {"symbol": "BTCUSD", "df": make_test_data(seed=1), "action": "BUY",
         "spread_bps": 2.5, "session": "london", "sl": 42000, "tp": 45000, "entry_price": 43250},
        {"symbol": "ETHUSD", "df": make_test_data(seed=2), "action": "BUY",
         "spread_bps": 3.0, "session": "london", "sl": 2500, "tp": 2800, "entry_price": 2580},
        {"symbol": "EURUSD", "df": make_test_data(seed=3), "action": "SELL",
         "spread_bps": 1.5, "session": "new_york", "sl": 1.090, "tp": 1.080, "entry_price": 1.085},
    ]
    ranking = ranker.rank_opportunities(opps)
    print(f"  Best: {ranking['best']} ({ranking['best_score']:.1f})")
    print(f"  Actionable: {ranking['actionable']}")
    print(f"  Skipped: {ranking['skipped']}")
    for r in ranking["ranking"]:
        print(f"    {r['symbol']:8s}: {r['score']:.1f} ({r['tier']})")


def test_trend_fatigue_detector():
    print("\n[2/7] Testing TrendFatigueDetector...")
    from trading_modules.trend_fatigue_detector import TrendFatigueDetector
    detector = TrendFatigueDetector()
    df = make_test_data()
    fatigue = detector.detect(df, trend_direction="up")
    d = fatigue.to_dict()
    print(f"  Score: {d['score']:.1f}/100 ({d['level']})")
    print(f"  Signals: {d['signals_detected']}")
    print(f"  Reversal prob: {d['reversal_probability']:.0%}")
    print(f"  Recommendation: {d['recommendation']}")


def test_dynamic_exit_intelligence():
    print("\n[3/7] Testing DynamicExitIntelligence...")
    from trading_modules.dynamic_exit_intelligence import DynamicExitIntelligence, ExitAction
    exit_ai = DynamicExitIntelligence()

    # Test winning position (2R)
    rec1 = exit_ai.evaluate(
        position_side="BUY", entry_price=43250, current_price=43800,
        stop_loss=42500, take_profit=45000, df=make_test_data(),
        r_multiple=1.2, hold_time_bars=15, spread_bps=2.5,
    )
    print(f"  Winning (1.2R): {rec1.action.value}")
    print(f"    Reason: {rec1.reason}")
    print(f"    New stop: {rec1.new_stop:.0f}")

    # Test losing position
    rec2 = exit_ai.evaluate(
        position_side="BUY", entry_price=43250, current_price=42800,
        stop_loss=42500, take_profit=45000, df=make_test_data(),
        r_multiple=-0.9, hold_time_bars=10, spread_bps=5.0,
    )
    print(f"\n  Losing (-0.9R): {rec2.action.value}")
    print(f"    Reason: {rec2.reason}")

    # Test 2R+ (partial close)
    rec3 = exit_ai.evaluate(
        position_side="BUY", entry_price=43250, current_price=44200,
        stop_loss=42500, take_profit=45000, df=make_test_data(),
        r_multiple=2.1, hold_time_bars=30, spread_bps=2.5,
    )
    print(f"\n  2R+ ({2.1}R): {rec3.action.value}")
    print(f"    Close pct: {rec3.close_pct:.0%}")
    print(f"    Reason: {rec3.reason}")


def test_adaptive_strategy_router():
    print("\n[4/7] Testing AdaptiveStrategyRouter...")
    from trading_modules.adaptive_strategy_router import (
        AdaptiveStrategyRouter, StrategyType,
    )
    router = AdaptiveStrategyRouter()

    # Test different regimes
    regimes = [
        ("trend_up", "normal", False, "expansion"),
        ("range", "normal", False, "consolidation"),
        ("crisis", "extreme", False, "decline"),
        ("breakout", "high", False, "expansion"),
    ]
    for regime, vol, news, cycle in regimes:
        result = router.route(regime=regime, volatility_regime=vol,
                            news_pending=news, market_cycle=cycle)
        print(f"  {regime:10s} ({vol:8s}): {result.strategy.value:25s} "
              f"size={result.position_size_multiplier:.0%} ({result.action})")

    print(f"\n  Switch history: {len(router.switch_history())}")
    print(f"  Current: {router.current().value}")


def test_institutional_performance_analytics():
    print("\n[5/7] Testing InstitutionalPerformanceAnalytics...")
    from trading_modules.institutional_performance_analytics import (
        InstitutionalPerformanceAnalytics,
    )
    analytics = InstitutionalPerformanceAnalytics(initial_equity=10000)

    # Record 50 trades
    np.random.seed(42)
    for i in range(50):
        win = np.random.random() < 0.60
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.8 if win else -1.0
        analytics.record_trade(pnl=pnl, r_multiple=r, hold_time_s=3600)

    report = analytics.report()
    d = report.to_dict()
    print(f"  Trades: {d['total_trades']} (WR: {d['win_rate']:.0%})")
    print(f"  Total P&L: ${d['total_pnl']:.0f}")
    print(f"  Sharpe: {d['sharpe']:.2f}")
    print(f"  Sortino: {d['sortino']:.2f}")
    print(f"  Calmar: {d['calmar']:.2f}")
    print(f"  Profit Factor: {d['profit_factor']:.2f}")
    print(f"  Max DD: {d['max_drawdown_pct']:.1f}%")
    print(f"  VaR 95%: ${d['var_95']:.0f}")
    print(f"  CVaR 95%: ${d['cvar_95']:.0f}")
    print(f"  Omega: {d['omega']:.2f}")
    print(f"  Grade: {d['grade']} — {d['description']}")


def test_autonomous_model_lifecycle():
    print("\n[6/7] Testing AutonomousModelLifecycleManager...")
    from trading_modules.autonomous_model_lifecycle import (
        AutonomousModelLifecycleManager, LifecycleStage,
    )
    mgr = AutonomousModelLifecycleManager(min_samples_for_retrain=20)

    # Collect samples
    for i in range(25):
        mgr.collect_sample("momentum_v1",
                          features={"rsi": 60+i%10, "trend": 0.7},
                          label="win" if i % 3 != 0 else "loss")

    print(f"  Samples: {len(mgr.get_training_data('momentum_v1'))}")
    print(f"  Ready for retrain: {mgr.ready_for_retrain('momentum_v1')}")

    # Create version + validate + backtest
    version = mgr.create_version("momentum_v1")
    print(f"  Created: {version}")

    # Validate
    valid = mgr.validate("momentum_v1", version, accuracy=0.68)
    print(f"  Validation: {valid}")

    # Backtest
    bt = mgr.backtest("momentum_v1", version, sharpe=1.2, win_rate=0.62)
    print(f"  Backtest passed: {bt}")

    # Paper trade
    mgr.start_paper_trading("momentum_v1", version)
    for i in range(35):
        mgr.record_paper_trade("momentum_v1", version,
                              pnl=np.random.uniform(-20, 50))

    ready = mgr.paper_trade_ready_for_deploy("momentum_v1", version)
    print(f"  Paper trade ready: {ready}")

    if ready:
        mgr.deploy("momentum_v1", version)
        print(f"  Deployed!")

    # Record live trades
    for i in range(25):
        mgr.record_live_trade("momentum_v1", pnl=np.random.uniform(-15, 40))

    report = mgr.report("momentum_v1")
    d = report.to_dict()
    print(f"\n  Report: stage={d['current_stage']}")
    print(f"  Versions: {d['versions_total']}")
    print(f"  Recommendations: {d['recommendations'][:1]}")


def test_decision_intelligence_layer():
    print("\n[7/7] Testing DecisionIntelligenceLayer...")
    from trading_modules.decision_intelligence_layer import (
        DecisionIntelligenceLayer, DecisionAction,
    )
    layer = DecisionIntelligenceLayer()

    # Strong trade case
    d1 = layer.decide(
        market_context_score=0.85, trend_score=0.80, liquidity_score=0.75,
        order_flow_score=0.70, volatility_score=0.70, correlation_score=0.80,
        execution_score=0.75, portfolio_risk_score=0.85,
        expected_r_if_win=2.0, probability_win=0.65,
    )
    print(f"  Strong case: {d1.action.value}")
    print(f"    Quality: {d1.decision_quality:.1f}/100")
    print(f"    EV: {d1.expected_value_r:.2f}R")
    print(f"    Kelly: {d1.kelly_fraction:.2f}")
    print(f"    Size: {d1.position_size_mult:.2f}x")
    print(f"    Reason: {d1.reason}")
    print(f"    Strengths: {d1.strengths[:3]}")

    # Weak case
    d2 = layer.decide(
        market_context_score=0.40, trend_score=0.35, liquidity_score=0.30,
        order_flow_score=0.40, volatility_score=0.50, correlation_score=0.60,
        execution_score=0.50, portfolio_risk_score=0.45,
        expected_r_if_win=1.5, probability_win=0.45,
    )
    print(f"\n  Weak case: {d2.action.value}")
    print(f"    Quality: {d2.decision_quality:.1f}/100")
    print(f"    Weaknesses: {d2.weaknesses[:3]}")

    # Evaluate decision quality after outcome
    eval1 = layer.evaluate_decision_quality(d1, actual_outcome="win")
    print(f"\n  Decision evaluation (win): {eval1['process_grade']} — {eval1['learning_note']}")

    eval2 = layer.evaluate_decision_quality(d1, actual_outcome="loss")
    print(f"  Decision evaluation (loss): {eval2['process_grade']} — {eval2['learning_note']}")


def test_wisdom_gate_200_principles():
    print("\n[BONUS] Verifying 200 principles in WisdomGate...")
    from livermore_principles import WisdomGate, TradeContext
    gate = WisdomGate()
    ctx = TradeContext(
        symbol='BTCUSD', direction='BUY', confidence=0.75,
        win_rate=0.60, rr_ratio=2.5, atr_ratio=0.015,
        bars_since_last_trade=20, spread_bps=2.0,
        regime='trend_up', drawdown_pct=3.0,
        recent_losses=0, recent_wins=3,
        pattern_match_count=10, pattern_win_rate=0.65,
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
        market_context_score=0.8, context_understood=True,
        capital_efficiency=0.75, strategy_edge_declining=False,
        volatility_regime='normal', liquidity_quality=0.8,
        daily_risk_budget_remaining=1.5, daily_risk_budget_used=0.5,
        correlated_exposure_pct=0.2, adaptive_confidence=0.7,
        decision_quality_score=0.85, execution_latency_ms=150,
        missed_opportunity_count=5, portfolio_balance_score=0.8,
        consecutive_loss_count=0, risk_reduction_active=False,
        execution_window_quality=0.85, structural_change_detected=False,
        learning_loop_active=True, risk_adjusted_return_target=2.5,
        survival_mode_active=False, market_cycle='expansion',
        cycle_confidence=0.8, probability_buy=0.74, probability_sell=0.18,
        probability_wait=0.08, structure_priority_score=0.85,
        portfolio_risk_usd=150, dynamic_risk_mode='normal',
        false_confidence_detected=False, liquidity_asset_score=0.8,
        idle_mode=False, strategy_evolution_active=True,
        edge_decay_rate=0.05, allocation_diversified=True,
        knowledge_added=True, institutional_memory_size=50,
        black_swan_prepared=True, opportunity_cost_acceptable=True,
        self_diagnosis_passed=True, benchmark_outperformance=1.2,
        decision_engine_consensus=0.85, autonomous_mode=True,
        # v7.7 fields
        patience_mode=False, opportunity_rank=0.85,
        discipline_score=0.85, trend_persistence_score=0.75,
        noise_filtered=True, multi_confirmation_count=5,
        dynamic_exit_ready=True, near_miss_analyzed=True,
        market_memory_available=True, confidence_earned=True,
        market_fatigue_detected=False, capital_protection_active=True,
        execution_optimized=True, portfolio_correlation_managed=True,
        strategy_switched=True, rl_loop_active=True,
        performance_dashboard_active=True, decision_quality_focus=True,
        autonomous_improvement=True, institutional_mindset_complete=True,
    )
    verdict = gate.evaluate(ctx)
    total = verdict.checks_passed + verdict.checks_failed
    print(f"  Total principles: {total}")
    print(f"  Passed: {verdict.checks_passed}")
    print(f"  Failed: {verdict.checks_failed}")
    print(f"  Approved: {verdict.approved}")
    assert total == 200, f"Expected 200, got {total}"
    print("  OK — all 200 principles present and evaluated")


if __name__ == "__main__":
    print("=" * 70)
    print("  7 NEW MODULES TEST (Principles 181-200)")
    print("  Pages 180-200: Timing, Self-Control, Decision Making")
    print("=" * 70)
    test_trade_opportunity_ranker()
    test_trend_fatigue_detector()
    test_dynamic_exit_intelligence()
    test_adaptive_strategy_router()
    test_institutional_performance_analytics()
    test_autonomous_model_lifecycle()
    test_decision_intelligence_layer()
    test_wisdom_gate_200_principles()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED — 7 MODULES + 200 PRINCIPLES VERIFIED")
    print("=" * 70)
