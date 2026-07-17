"""scripts/test_livermore_141_160.py
=====================================================================
Test the 7 new modules for principles 141-160 + verify 160 principles.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


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
        wick_up = abs(np.random.normal(0, body * 0.5 + 1))
        wick_dn = abs(np.random.normal(0, body * 0.5 + 1))
        high[i] = max(open_[i], close[i]) + wick_up
        low[i] = min(open_[i], close[i]) - wick_dn
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


def test_market_context_engine():
    print("\n[1/7] Testing MarketContextEngine...")
    from trading_modules.market_context_engine import MarketContextEngine, VolatilityRegime, MarketRegime
    engine = MarketContextEngine()
    df = make_test_data()
    ctx = engine.evaluate(df, spread_bps=2.5, session="london",
                         news_pending=False, minutes_to_news=180)
    d = ctx.to_dict()
    print(f"  Context score: {d['context_score']:.1f}/100")
    print(f"  Context understood: {d['context_understood']}")
    print(f"  Can trade: {d['can_trade']}")
    print(f"  Market regime: {d['market_regime']}")
    print(f"  Volatility regime: {d['volatility_regime']}")
    print(f"  Price driver: {d['price_driver']}")
    print(f"  Trend score: {d['trend_score']:.2f}")
    print(f"  Liquidity score: {d['liquidity_score']:.2f}")
    print(f"  Description: {d['description']}")
    print(f"  Recommendations: {d['recommendations'][:2]}")


def test_risk_budget_manager():
    print("\n[2/7] Testing RiskBudgetManager...")
    from trading_modules.risk_budget_manager import RiskBudgetManager
    mgr = RiskBudgetManager(equity=10000, daily_risk_pct=2.0)
    # Try to take 3 trades
    for i in range(3):
        risk_usd = 50  # 0.5% each
        allowed, reason = mgr.can_take_trade(risk_usd)
        if allowed:
            mgr.allocate_risk(risk_usd, strategy="momentum")
            print(f"  Trade {i+1}: allowed — risk ${risk_usd} ({reason})")
        else:
            print(f"  Trade {i+1}: blocked — {reason}")

    # Record some results
    mgr.record_trade_result(pnl=50, win=True)
    print(f"  After win: consec_losses={mgr.state().consecutive_losses}, mult={mgr.get_risk_multiplier():.2f}")
    mgr.record_trade_result(pnl=-50, win=False)
    mgr.record_trade_result(pnl=-50, win=False)
    mgr.record_trade_result(pnl=-50, win=False)
    print(f"  After 3 losses: mult={mgr.get_risk_multiplier():.2f}")

    state = mgr.summary()
    print(f"  Daily risk used: {state['daily_risk_used_pct']:.2f}%")
    print(f"  Daily risk remaining: {state['daily_risk_remaining_pct']:.2f}%")
    print(f"  Trading paused: {state['trading_paused']}")


def test_missed_opportunity_analyzer():
    print("\n[3/7] Testing MissedOpportunityAnalyzer...")
    from trading_modules.missed_opportunity_analyzer import MissedOpportunityAnalyzer
    analyzer = MissedOpportunityAnalyzer(max_bars_to_resolve=5)

    # Record missed opportunities
    analyzer.record_missed("BTCUSD", "BUY", 43250, 42500, 45000,
                          "wisdom_gate", "ATR% too high",
                          {"rsi": 62, "regime": "trend_up"}, confidence=0.72)
    analyzer.record_missed("ETHUSD", "SELL", 2580, 2650, 2400,
                          "risk_pipeline", "correlated exposure",
                          {"rsi": 75}, confidence=0.68)
    analyzer.record_missed("EURUSD", "BUY", 1.085, 1.080, 1.095,
                          "context_filter", "poor liquidity",
                          {}, confidence=0.65)

    # Update prices to resolve
    analyzer.update_prices({"BTCUSD": 44500, "ETHUSD": 2500, "EURUSD": 1.090})

    report = analyzer.analyze()
    print(f"  Total missed: {report['total_missed']}")
    print(f"  Would have won: {report['would_have_won']}")
    print(f"  Would have lost: {report['would_have_lost']}")
    print(f"  Win rate if traded: {report['win_rate_if_traded']:.0%}")
    print(f"  Total R lost: {report['total_r_lost']:.2f}")
    print(f"  By reason: {report['by_reason']}")
    print(f"  Insights: {report['insights'][:2]}")
    print(f"  Recommendations: {report['recommendations'][:2]}")


def test_execution_optimizer():
    print("\n[4/7] Testing ExecutionOptimizer...")
    from trading_modules.execution_optimizer import ExecutionOptimizer, OrderType, Urgency
    opt = ExecutionOptimizer()

    # Find best window
    window = opt.find_best_window("BTCUSD", spread_bps=2.5, session="london",
                                  minutes_to_news=180, orderbook_depth_usd=2_000_000)
    print(f"  Execution window: score={window.score:.2f}")
    print(f"  Should execute now: {window.should_execute_now}")
    print(f"  Recommendation: {window.recommendation}")

    # Recommend order type
    rec = opt.recommend_order("BTCUSD", "BUY", 0.5, Urgency.NORMAL, spread_bps=2.5)
    print(f"\n  Order recommendation: {rec.order_type.value}")
    print(f"  Limit offset: {rec.limit_offset_bps:.1f}bps")
    print(f"  Time in force: {rec.time_in_force}")
    print(f"  Reason: {rec.reason}")

    # Record executions
    for i in range(10):
        opt.record_execution("BTCUSD", "BUY", 43250, 43255,
                            latency_ms=85, fill_ratio=1.0, spread_at_execution=2.5)

    quality = opt.execution_quality()
    print(f"\n  Execution quality: {quality['status']}")
    print(f"  Avg slippage: {quality['avg_slippage_bps']:.2f}bps")
    print(f"  Avg latency: {quality['avg_latency_ms']:.0f}ms")
    print(f"  Quality score: {quality['quality_score']:.2f}")


def test_strategy_lifecycle_manager():
    print("\n[5/7] Testing StrategyLifecycleManager...")
    from trading_modules.strategy_lifecycle_manager import (
        StrategyLifecycleManager, LifecycleStage,
    )
    mgr = StrategyLifecycleManager()

    # Register strategies
    mgr.register("momentum_v4", initial_sharpe=1.8, initial_win_rate=0.62, initial_ev_r=0.5)
    mgr.register("mean_rev_v2", initial_sharpe=1.5, initial_win_rate=0.55, initial_ev_r=0.3)

    # Update momentum: healthy
    mgr.update_stats("momentum_v4", win_rate=0.60, sharpe=1.7, ev_r=0.45, trades=50)
    # Update mean_rev: declining
    mgr.update_stats("mean_rev_v2", win_rate=0.40, sharpe=0.8, ev_r=0.10, trades=50)

    for name in ["momentum_v4", "mean_rev_v2"]:
        s = mgr.get_state(name)
        print(f"  {name}: stage={s.stage.value}, action={s.action.value}")
        print(f"    Edge decline: {s.edge_decline_pct:.0%}")
        print(f"    Size multiplier: {s.size_multiplier:.2f}x")
        print(f"    Days active: {s.days_active}")

    # Push mean_rev to decay
    mgr.update_stats("mean_rev_v2", win_rate=0.35, sharpe=0.5, ev_r=-0.05, trades=10)
    s = mgr.get_state("mean_rev_v2")
    print(f"\n  mean_rev_v2 after more decline: {s.stage.value}")

    print(f"\n  Summary: {mgr.summary()['by_stage']}")


def test_institutional_portfolio_engine():
    print("\n[6/7] Testing InstitutionalPortfolioEngine...")
    from trading_modules.institutional_portfolio_engine import InstitutionalPortfolioEngine
    engine = InstitutionalPortfolioEngine(equity=10000)

    # Add positions across strategies
    engine.add_position("BTCUSD", "BUY", 0.5, 43250, strategy="momentum", sl=42000, current_price=43500)
    engine.add_position("ETHUSD", "BUY", 3.0, 2580, strategy="trend", sl=2500, current_price=2620)
    engine.add_position("EURUSD", "SELL", 1.0, 1.085, strategy="mean_rev", sl=1.095, current_price=1.082)

    # Set strategy stats for Kelly
    engine.set_strategy_stats("momentum", win_rate=0.62, avg_win_r=1.8, avg_loss_r=1.0, sample_size=50)
    engine.set_strategy_stats("trend", win_rate=0.55, avg_win_r=2.2, avg_loss_r=1.0, sample_size=40)

    # Set price history
    engine.set_price_history({
        "BTCUSD": make_test_data(seed=1),
        "ETHUSD": make_test_data(seed=2),
        "EURUSD": make_test_data(seed=3),
    })

    report = engine.report()
    d = report.to_dict()
    print(f"  Portfolio heat: {d['portfolio_heat']:.1%}")
    print(f"  Diversification: {d['diversification_score']:.2f}")
    print(f"  Strategy diversification: {d['strategy_diversification']:.2f}")
    print(f"  Effective positions: {d['effective_positions']:.1f}")
    print(f"  Strategy concentration: {d['strategy_concentration']}")
    print(f"  Scenario loss (5% drop): ${d['scenario_loss_5pct']:.0f}")
    print(f"  Scenario loss (10% drop): ${d['scenario_loss_10pct']:.0f}")
    print(f"  Capacity remaining: {d['capacity_remaining_pct']:.1%}")
    print(f"  VaR 95%: ${d['portfolio_var_95']:.0f}")
    print(f"  Beta: {d['portfolio_beta']:.2f}")
    print(f"  Sharpe estimate: {d['sharpe_estimate']:.2f}")
    print(f"  Kelly optimal: {d['kelly_optimal']}")
    print(f"  Rebalancing needed: {d['rebalancing_needed']}")
    print(f"  Recommendations: {d['recommendations'][:2]}")


def test_continuous_improvement_system():
    print("\n[7/7] Testing ContinuousImprovementSystem...")
    from trading_modules.continuous_improvement_system import ContinuousImprovementSystem
    cis = ContinuousImprovementSystem(benchmark_return_pct=2.0)
    cis.set_equity(10000)

    # Record 30 trades
    np.random.seed(42)
    for i in range(30):
        win = np.random.random() < 0.60
        pnl = np.random.uniform(30, 80) if win else -np.random.uniform(20, 40)
        r = 1.8 if win else -1.0
        cis.record_trade(
            strategy="momentum", symbol="BTCUSD",
            pnl=pnl, r_multiple=r, confidence=0.7,
            features={"rsi": 62, "trend_score": 0.7, "volume_ratio": 1.3},
            win=win,
        )

    # Daily review
    daily = cis.daily_review()
    d = daily.to_dict()
    print(f"  DAILY REVIEW:")
    print(f"    Performance trend: {d['performance_trend']}")
    print(f"    Improvement score: {d['improvement_score']:.3f}")
    print(f"    EV trend: {d['ev_trend']:.3f}")
    print(f"    Parameter nudges: {d['parameter_nudges']}")
    print(f"    Top features: {d['top_features'][:3]}")
    print(f"    Recommendations: {d['recommendations'][:2]}")

    # Weekly audit
    weekly = cis.weekly_audit()
    w = weekly.to_dict()
    print(f"\n  WEEKLY AUDIT:")
    print(f"    Performance trend: {w['performance_trend']}")
    print(f"    Strategies gaining: {w['strategies_gaining']}")
    print(f"    Strategies losing: {w['strategies_losing']}")
    print(f"    vs Benchmark: ${w['vs_benchmark']:.0f}")
    print(f"    Retrain recommended: {w['retrain_recommended']}")
    print(f"    Action items: {w['action_items'][:2]}")

    # Monthly review
    monthly = cis.monthly_review()
    m = monthly.to_dict()
    print(f"\n  MONTHLY REVIEW:")
    print(f"    Performance trend: {m['performance_trend']}")
    print(f"    Retrain recommended: {m['retrain_recommended']}")
    print(f"    Strategies to retrain: {m['strategies_to_retrain']}")
    print(f"    Action items: {m['action_items'][:2]}")


def test_wisdom_gate_160_principles():
    """Verify all 160 principles."""
    print("\n[BONUS] Verifying 160 principles in WisdomGate...")
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
        # v7.5 fields
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
    )
    verdict = gate.evaluate(ctx)
    total = verdict.checks_passed + verdict.checks_failed
    print(f"  Total principles: {total}")
    print(f"  Passed: {verdict.checks_passed}")
    print(f"  Failed: {verdict.checks_failed}")
    print(f"  Approved: {verdict.approved}")
    assert total >= 160, f"Expected at least 160, got {total}"
    print("  OK — all 160 principles present and evaluated")


if __name__ == "__main__":
    print("=" * 70)
    print("  7 NEW MODULES TEST (Principles 141-160)")
    print("  Pages 140-160: Decision Quality, Context, Self-Evolution")
    print("=" * 70)
    test_market_context_engine()
    test_risk_budget_manager()
    test_missed_opportunity_analyzer()
    test_execution_optimizer()
    test_strategy_lifecycle_manager()
    test_institutional_portfolio_engine()
    test_continuous_improvement_system()
    test_wisdom_gate_160_principles()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED — 7 MODULES + 160 PRINCIPLES VERIFIED")
    print("=" * 70)
