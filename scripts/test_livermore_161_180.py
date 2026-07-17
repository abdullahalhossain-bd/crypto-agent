"""scripts/test_livermore_161_180.py
=====================================================================
Test the 7 new modules for principles 161-180 + verify 180 principles.
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


def test_market_cycle_engine():
    print("\n[1/7] Testing MarketCycleEngine...")
    from trading_modules.market_cycle_engine import MarketCycleEngine, CyclePhase
    engine = MarketCycleEngine()
    df = make_test_data()
    cycle = engine.detect(df)
    assert cycle.phase in CyclePhase
    d = cycle.to_dict()
    print(f"  Phase: {d['phase']}, confidence={d['confidence']:.2f}")
    print(f"  Price trend: {d['price_trend']:+.2f}, vol: {d['volume_trend']}")
    print(f"  Range position: {d['range_position']:.0%}")
    print(f"  Strategy: {d['strategy_recommendation']}")
    print(f"  Description: {d['description'][:80]}")


def test_institutional_memory_database():
    print("\n[2/7] Testing InstitutionalMemoryDatabase...")
    from trading_modules.institutional_memory_database import InstitutionalMemoryDatabase
    db = InstitutionalMemoryDatabase(db_path="data/test_memory.db")

    # Store memories
    for i in range(10):
        db.store_pattern(
            symbol="BTCUSD", pattern="bull_flag",
            features={"rsi": 60+i, "atr_pct": 1.5, "volume": 1.3},
            outcome="win" if i < 7 else "loss",
            pnl=42 if i < 7 else -30,
            r_multiple=1.8 if i < 7 else -1.0,
            regime="trend_up", session="london",
        )
    for i in range(5):
        db.store_pattern(
            symbol="ETHUSD", pattern="bear_flag",
            features={"rsi": 35+i, "atr_pct": 2.0, "volume": 1.5},
            outcome="win" if i < 3 else "loss",
            pnl=35 if i < 3 else -25,
            r_multiple=1.5 if i < 3 else -1.0,
            regime="trend_down", session="new_york",
        )

    # Query similar
    similar = db.query_similar(symbol="BTCUSD",
        features={"rsi": 62, "atr_pct": 1.4, "volume": 1.2}, top_k=3)
    print(f"  Similar memories found: {len(similar)}")

    # Pattern stats
    stats = db.query_by_pattern("bull_flag")
    print(f"  Pattern 'bull_flag': {stats['count']} cases, WR={stats['win_rate']:.0%}, avg_r={stats['avg_r']:.2f}")

    # Regime stats
    regime = db.query_by_regime("trend_up")
    print(f"  Regime 'trend_up': {regime['count']} cases, WR={regime['win_rate']:.0%}")

    # Overall stats
    overall = db.stats()
    print(f"  Total memories: {overall['total_memories']}")
    print(f"  Patterns: {overall['patterns_in_memory']}")

    # Cleanup
    try: os.remove("data/test_memory.db")
    except Exception: pass


def test_opportunity_cost_analyzer():
    print("\n[3/7] Testing OpportunityCostAnalyzer...")
    from trading_modules.opportunity_cost_analyzer import OpportunityCostAnalyzer, OpportunityDecision
    analyzer = OpportunityCostAnalyzer()

    # Excellent setup
    d1 = analyzer.evaluate(current_score=90, expected_better_setup_minutes=60,
                           capital_idle_cost_pct=0.01, setup_frequency_per_day=8,
                           historical_avg_score=65, historical_avg_r=0.3)
    print(f"  Score 90: {d1.decision.value} — {d1.reason[:60]}")

    # Poor setup
    d2 = analyzer.evaluate(current_score=40, expected_better_setup_minutes=30)
    print(f"  Score 40: {d2.decision.value} — {d2.reason[:60]}")

    # Marginal setup
    d3 = analyzer.evaluate(current_score=72, expected_better_setup_minutes=15,
                           setup_frequency_per_day=12, historical_avg_r=0.4)
    print(f"  Score 72: {d3.decision.value} — {d3.reason[:60]}")
    print(f"    EV of waiting: {d3.expected_value_of_waiting:.3f}R")
    print(f"    Risk of missing: {d3.risk_of_missing:.0%}")


def test_strategy_evolution_manager():
    print("\n[4/7] Testing StrategyEvolutionManager...")
    from trading_modules.strategy_evolution_manager import StrategyEvolutionManager, EvolutionStage
    mgr = StrategyEvolutionManager(min_samples_for_retrain=10)

    # Collect trades
    np.random.seed(42)
    for i in range(15):
        win = np.random.random() < 0.40  # declining performance
        r = 1.5 if win else -1.0
        mgr.collect_trade("momentum", {"rsi": 62, "trend": 0.7},
                         "win" if win else "loss", 50 if win else -40, r)

    # Check decay
    print(f"  Decay: {mgr.get_decay('momentum'):.0%}")
    print(f"  Should retrain: {mgr.should_retrain('momentum')}")

    # Trigger retrain
    if mgr.should_retrain("momentum"):
        version = mgr.trigger_retrain("momentum")
        print(f"  Retrained: {version}")

        # Validate
        passed = mgr.validate("momentum", version,
                              validation_ev_r=0.4, validation_win_rate=0.60,
                              validation_sharpe=1.5)
        print(f"  Validation passed: {passed}")

        # Shadow test
        mgr.start_shadow_test("momentum", version)
        for i in range(25):
            r = 0.6 if np.random.random() < 0.65 else -0.5
            mgr.record_shadow_trade("momentum", version, r)

        # Check promotion
        good = mgr.shadow_performance_good("momentum", version)
        print(f"  Shadow outperforming: {good}")

        if good:
            mgr.promote("momentum", version)
            print(f"  Promoted to production")

    report = mgr.report("momentum")
    print(f"  Stage: {report.current_stage.value}")
    print(f"  Versions: {report.versions_total}")
    print(f"  Recommendations: {report.recommendations[:1]}")


def test_portfolio_allocation_optimizer():
    print("\n[5/7] Testing PortfolioAllocationOptimizer...")
    from trading_modules.portfolio_allocation_optimizer import PortfolioAllocationOptimizer
    opt = PortfolioAllocationOptimizer(equity=10000)

    # Set performance
    opt.set_strategy_performance("trend", win_rate=0.62, avg_r=0.5, sharpe=1.5, sample_size=50)
    opt.set_strategy_performance("breakout", win_rate=0.45, avg_r=0.3, sharpe=1.0, sample_size=30)
    opt.set_strategy_performance("mean_reversion", win_rate=0.55, avg_r=0.2, sharpe=0.8, sample_size=20)

    # Optimize for expansion
    alloc1 = opt.optimize(market_cycle="expansion", risk_budget_remaining=0.8)
    print(f"  EXPANSION allocation:")
    for s, pct in alloc1.allocation.items():
        print(f"    {s:15s}: {pct:.0%} (${alloc1.allocation_usd[s]:,.0f})")
    print(f"  Diversification: {alloc1.diversification_score:.2f}")
    print(f"  Reason: {alloc1.reason[:70]}")

    # Optimize for decline
    alloc2 = opt.optimize(market_cycle="decline", risk_budget_remaining=0.3)
    print(f"\n  DECLINE allocation:")
    for s, pct in alloc2.allocation.items():
        print(f"    {s:15s}: {pct:.0%}")
    print(f"  Cash reserve: {alloc2.cash_reserve:.0%}")


def test_ai_self_diagnosis():
    print("\n[6/7] Testing AISelfDiagnosis...")
    from trading_modules.ai_self_diagnosis import AISelfDiagnosis
    diag = AISelfDiagnosis(min_trades_for_diagnosis=10)

    # Record trades with mixed performance
    np.random.seed(42)
    for i in range(30):
        win = np.random.random() < 0.55
        # Make Asia session perform poorly
        session = "asia" if i % 5 == 0 else "london"
        regime = "range" if i % 4 == 0 else "trend_up"
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.5 if win else -1.0
        diag.record_trade(
            strategy="momentum", session=session, regime=regime,
            pnl=pnl, r_multiple=r, confidence=0.7,
            slippage_bps=1.5, followed_rules=i % 10 != 0,
        )

    report = diag.diagnose()
    d = report.to_dict()
    print(f"  Overall health: {d['overall_health']:.1f}/10")
    print(f"  Improvement trajectory: {d['improvement_trajectory']:+.2f}")
    print(f"  Dimensions:")
    for dim, score in d['dimensions'].items():
        print(f"    {dim:15s}: {score:.1f}/10")
    print(f"  Top weaknesses: {len(d['top_weaknesses'])}")
    for w in d['top_weaknesses'][:2]:
        print(f"    [{w['dimension']}] {w['symptom']}")
        print(f"      Fix: {w['fix']}")
    print(f"  Self-awareness: {d['self_awareness_notes'][:70]}")


def test_institutional_decision_engine():
    print("\n[7/7] Testing InstitutionalDecisionEngine...")
    from trading_modules.institutional_decision_engine import InstitutionalDecisionEngine, Decision
    engine = InstitutionalDecisionEngine()

    # Strong approve case
    d1 = engine.evaluate(
        structure_score=0.85, liquidity_score=0.80, order_flow_score=0.70,
        volatility_score=0.75, correlation_score=0.80, macro_score=0.75,
        risk_budget_score=0.90, execution_score=0.80,
    )
    print(f"  Strong case: {d1.decision.value} (consensus={d1.consensus:.2f})")
    print(f"    Size multiplier: {d1.position_size_multiplier:.2f}x")
    print(f"    Reason: {d1.reason}")

    # Veto case (no liquidity)
    d2 = engine.evaluate(
        structure_score=0.80, liquidity_score=0.10,  # VETO
        order_flow_score=0.70, volatility_score=0.70,
        correlation_score=0.80, macro_score=0.70,
        risk_budget_score=0.80, execution_score=0.70,
    )
    print(f"\n  Veto case: {d2.decision.value}")
    print(f"    Vetos: {d2.vetos}")
    print(f"    Reason: {d2.reason}")

    # Marginal case
    d3 = engine.evaluate(
        structure_score=0.55, liquidity_score=0.60, order_flow_score=0.40,
        volatility_score=0.50, correlation_score=0.60, macro_score=0.45,
        risk_budget_score=0.50, execution_score=0.55,
    )
    print(f"\n  Marginal case: {d3.decision.value} (consensus={d3.consensus:.2f})")
    approves = sum(1 for v in d3.dimensions.values() if v.vote.value == "APPROVE")
    rejects = sum(1 for v in d3.dimensions.values() if v.vote.value == "REJECT")
    print(f"    {approves} approve, {rejects} reject")


def test_wisdom_gate_180_principles():
    print("\n[BONUS] Verifying 180 principles in WisdomGate...")
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
        # v7.6 fields
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
    )
    verdict = gate.evaluate(ctx)
    total = verdict.checks_passed + verdict.checks_failed
    print(f"  Total principles: {total}")
    print(f"  Passed: {verdict.checks_passed}")
    print(f"  Failed: {verdict.checks_failed}")
    print(f"  Approved: {verdict.approved}")
    assert total >= 180, f"Expected at least 180, got {total}"
    print("  OK — all 180 principles present and evaluated")


if __name__ == "__main__":
    print("=" * 70)
    print("  7 NEW MODULES TEST (Principles 161-180)")
    print("  Pages 160-180: Cycles, Survival, Adaptive Intelligence")
    print("=" * 70)
    test_market_cycle_engine()
    test_institutional_memory_database()
    test_opportunity_cost_analyzer()
    test_strategy_evolution_manager()
    test_portfolio_allocation_optimizer()
    test_ai_self_diagnosis()
    test_institutional_decision_engine()
    test_wisdom_gate_180_principles()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED — 7 MODULES + 180 PRINCIPLES VERIFIED")
    print("=" * 70)
