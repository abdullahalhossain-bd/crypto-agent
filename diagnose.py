"""diagnose.py
=====================================================================
Full Project Diagnostic — Tests EVERY Step of the Trading Pipeline
=====================================================================
This script walks through the ENTIRE trading pipeline step-by-step,
testing each module and showing exactly what works and what fails.

Usage:
    python diagnose.py              # full diagnostic
    python diagnose.py --quick      # quick check (imports only)
    python diagnose.py --verbose    # show all details

Output:
    ✓ PASS — each working step with actual result
    ✗ FAIL — each broken step with error message
    Summary at the end with total pass/fail count
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ----------------------------------------------------------------------
# Test result tracking
# ----------------------------------------------------------------------
class TestResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed = False
        self.error = ""
        self.result: Any = None
        self.duration_ms = 0.0
        self.details: str = ""


_results: List[TestResult] = []


def test_step(category: str, name: str, fn, verbose: bool = False) -> TestResult:
    """Run a single test step and record the result."""
    r = TestResult(name=name, category=category)
    t0 = time.time()
    try:
        r.result = fn()
        r.passed = True
        r.duration_ms = (time.time() - t0) * 1000
        status = f"{GREEN}✓ PASS{RESET}"
        detail = ""
        if r.result is not None:
            if isinstance(r.result, dict):
                detail = str({k: v for k, v in list(r.result.items())[:3]})
            elif isinstance(r.result, (int, float, str, bool)):
                detail = str(r.result)
            elif isinstance(r.result, (list, tuple)):
                detail = f"{len(r.result)} items"
            else:
                detail = type(r.result).__name__
        r.details = detail
        print(f"  {status} [{r.duration_ms:.0f}ms] {category}/{name}"
              + (f" → {detail[:80]}" if detail and verbose else ""))
    except Exception as e:
        r.passed = False
        r.error = str(e)
        r.duration_ms = (time.time() - t0) * 1000
        print(f"  {RED}✗ FAIL{RESET} [{r.duration_ms:.0f}ms] {category}/{name}")
        print(f"         Error: {e}")
        if verbose:
            traceback.print_exc()
    _results.append(r)
    return r


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{RESET}")


def summary() -> None:
    section("DIAGNOSTIC SUMMARY")
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)
    total = len(_results)

    by_category: Dict[str, Tuple[int, int]] = {}
    for r in _results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = (0, 0)
        p, f = by_category[cat]
        if r.passed:
            by_category[cat] = (p + 1, f)
        else:
            by_category[cat] = (p, f + 1)

    print(f"\n  {BOLD}By Category:{RESET}")
    for cat, (p, f) in sorted(by_category.items()):
        status = GREEN if f == 0 else RED if p == 0 else YELLOW
        print(f"    {status}{cat:40s}{RESET} {p:3d} pass / {f:3d} fail")

    print(f"\n  {BOLD}Total:{RESET} {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED — System is healthy!{RESET}")
    else:
        print(f"\n  {RED}{BOLD}{failed} tests failed — see details above{RESET}")
        print(f"\n  {BOLD}Failed steps:{RESET}")
        for r in _results:
            if not r.passed:
                print(f"    {RED}✗{RESET} {r.category}/{r.name}: {r.error[:80]}")
    print()


# ----------------------------------------------------------------------
# Generate synthetic test data
# ----------------------------------------------------------------------
def make_test_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate valid synthetic OHLCV data."""
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 40000 * np.exp(np.cumsum(returns))
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
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))


# ======================================================================
# DIAGNOSTIC TESTS
# ======================================================================

def test_imports(verbose: bool) -> None:
    """Test 1: All module imports."""
    section("STEP 1: Module Imports")

    modules = [
        ("Foundation", "architecture.event_bus", "EventBus"),
        ("Foundation", "architecture.state_machine", "StateMachine"),
        ("Foundation", "architecture.recovery_engine", "SnapshotEngine"),
        ("Perception", "architecture.feature_pipeline", "FeaturePipeline"),
        ("Perception", "architecture.regime_orchestrator", "RegimeOrchestrator"),
        ("Perception", "trading_modules.market_context_engine", "MarketContextEngine"),
        ("Perception", "trading_modules.market_cycle_engine", "MarketCycleEngine"),
        ("Perception", "trading_modules.market_phase_detector", "MarketPhaseDetector"),
        ("Perception", "trading_modules.trend_fatigue_detector", "TrendFatigueDetector"),
        ("Brain", "architecture.multi_agent", "MultiAgentCoordinator"),
        ("Brain", "architecture.memory_system", "MemorySystem"),
        ("Brain", "architecture.online_learning", "OnlineLearner"),
        ("Brain", "trading_modules.decision_intelligence_layer", "DecisionIntelligenceLayer"),
        ("Brain", "trading_modules.institutional_memory_database", "InstitutionalMemoryDatabase"),
        ("Brain", "trading_modules.adaptive_learning_engine", "AdaptiveLearningEngine"),
        ("Brain", "trading_modules.strategy_evolution_manager", "StrategyEvolutionManager"),
        ("Brain", "trading_modules.autonomous_model_lifecycle", "AutonomousModelLifecycleManager"),
        ("Defense", "architecture.risk_pipeline", "RiskPipeline"),
        ("Defense", "architecture.portfolio_manager_v2", "PortfolioManager"),
        ("Defense", "architecture.self_healing", "SelfHealingSystem"),
        ("Defense", "livermore_principles", "WisdomGate"),
        ("Defense", "trading_modules.risk_budget_manager", "RiskBudgetManager"),
        ("Defense", "trading_modules.system_health_monitor", "SystemHealthMonitor"),
        ("Defense", "trading_modules.strategy_health_monitor", "StrategyHealthMonitor"),
        ("Defense", "trading_modules.ai_self_diagnosis", "AISelfDiagnosis"),
        ("Execution", "trading_modules.execution_optimizer", "ExecutionOptimizer"),
        ("Execution", "trading_modules.dynamic_exit_intelligence", "DynamicExitIntelligence"),
        ("Execution", "trading_modules.trade_opportunity_ranker", "TradeOpportunityRanker"),
        ("Execution", "trading_modules.opportunity_cost_analyzer", "OpportunityCostAnalyzer"),
        ("Execution", "trading_modules.adaptive_strategy_router", "AdaptiveStrategyRouter"),
        ("Execution", "trading_modules.setup_scoring_engine", "SetupScoringEngine"),
        ("Execution", "trading_modules.expected_value_calculator", "ExpectedValueCalculator"),
        ("Execution", "trading_modules.portfolio_allocation_optimizer", "PortfolioAllocationOptimizer"),
        ("Execution", "trading_modules.institutional_decision_engine", "InstitutionalDecisionEngine"),
        ("Execution", "trading_modules.emotion_volatility_filter", "EmotionVolatilityFilter"),
        ("Execution", "trading_modules.capital_flow_analyzer", "CapitalFlowAnalyzer"),
        ("Execution", "trading_modules.relative_strength_ranker", "RelativeStrengthRanker"),
        ("Execution", "trading_modules.multi_timeframe_consensus", "MultiTimeframeConsensusEngine"),
        ("Execution", "trading_modules.smart_money_detector", "SmartMoneyDetector"),
        ("Evolution", "trading_modules.continuous_improvement_system", "ContinuousImprovementSystem"),
        ("Evolution", "trading_modules.strategy_lifecycle_manager", "StrategyLifecycleManager"),
        ("Evolution", "trading_modules.weekly_self_audit", "WeeklySelfAuditor"),
        ("Evolution", "trading_modules.missed_opportunity_analyzer", "MissedOpportunityAnalyzer"),
        ("Evolution", "trading_modules.portfolio_intelligence_layer", "PortfolioIntelligenceLayer"),
        ("Evolution", "trading_modules.institutional_portfolio_engine", "InstitutionalPortfolioEngine"),
        ("Observability", "architecture.institutional_monitoring", "InstitutionalMonitor"),
        ("Observability", "architecture.decision_audit", "DecisionAuditor"),
        ("Observability", "architecture.simulation_layer", "SimulationLayer"),
        ("Observability", "trading_modules.institutional_performance_analytics", "InstitutionalPerformanceAnalytics"),
        # BUG FIX: was pointed at architecture.master_orchestrator, which
        # main.py deliberately quarantines on boot (see main.py's
        # "CRITICAL REGRESSION GUARD" — the v9/MasterOrchestrator pipeline
        # was retired in favor of architecture.integration.TradingBot).
        # That made this check — and the 3 downstream tests that import
        # MasterOrchestrator (boot, 7-layer check, E2E cycle) — permanently
        # fail with "No module named 'architecture.master_orchestrator'"
        # even on a perfectly healthy install. Test the class that's
        # actually live instead.
        ("Integration", "architecture.integration", "TradingBot"),
        ("Signals", "engine.signals_v3", "Signal"),
        ("Indicators", "utils.indicators.registry", "IndicatorEngine"),
    ]

    for cat, mod_name, class_name in modules:
        def _import(m=mod_name, c=class_name):
            mod = __import__(m, fromlist=[c])
            cls = getattr(mod, c)
            return cls.__name__
        test_step(cat, f"import {class_name}", _import, verbose)


def test_indicators(verbose: bool) -> None:
    """Test 2: Indicator library."""
    section("STEP 2: Indicator Library (66 indicators)")

    df = make_test_data()

    def _engine():
        from utils.indicators.registry import IndicatorEngine
        engine = IndicatorEngine()
        return {"count": engine.registry.count(),
                "categories": len(engine.registry.categories())}
    test_step("Indicators", "IndicatorEngine creation", _engine, verbose)

    def _calc_all():
        from utils.indicators.registry import IndicatorEngine
        engine = IndicatorEngine()
        results = engine.calculate_all(df)
        valid = sum(1 for r in results.values() if r.valid)
        return {"total": len(results), "valid": valid}
    test_step("Indicators", "calculate_all (66 indicators)", _calc_all, verbose)

    # Test individual indicator categories
    categories = [
        ("trend", "utils.indicators.trend", ["sma", "ema", "wma", "adx", "supertrend"]),
        ("momentum", "utils.indicators.momentum", ["rsi", "macd", "cci", "williams_r"]),
        ("volatility", "utils.indicators.volatility", ["atr", "bollinger_bands", "keltner_channel"]),
        ("volume", "utils.indicators.volume", ["obv", "vwap", "mfi", "cmf"]),
        ("structure", "utils.indicators.structure", ["swing_highs_lows", "break_of_structure"]),
        ("smc", "utils.indicators.smc", ["detect_fvg", "detect_order_block"]),
        ("candles", "utils.indicators.candles", ["detect_all_patterns"]),
        ("statistics", "utils.indicators.statistics", ["zscore", "skewness", "kurtosis"]),
        ("features", "utils.indicators.features", ["feature_vector", "confidence_scores"]),
        ("regime", "utils.indicators.regime", ["regime_detection"]),
        ("validation", "utils.indicators.validation", ["validate_ohlcv"]),
        ("caching", "utils.indicators.caching", ["IndicatorCache"]),
        ("diagnostics", "utils.indicators.diagnostics", ["Diagnostics"]),
    ]

    for cat_name, mod_name, funcs in categories:
        def _test_cat(m=mod_name, fs=funcs):
            mod = __import__(m, fromlist=fs)
            results = {}
            for f in fs:
                fn = getattr(mod, f)
                results[f] = callable(fn)
            return results
        test_step("Indicators", f"{cat_name} ({len(funcs)} funcs)", _test_cat, verbose)


def test_wisdom_gate(verbose: bool) -> None:
    """Test 3: WisdomGate with 200 principles."""
    section("STEP 3: WisdomGate (200 Livermore Principles)")

    def _init():
        from livermore_principles import WisdomGate
        gate = WisdomGate()
        return {"class": type(gate).__name__}
    test_step("WisdomGate", "initialization", _init, verbose)

    # Test with passing context
    def _passing():
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
        return {"approved": verdict.approved,
                "passed": verdict.checks_passed,
                "failed": verdict.checks_failed,
                "total": verdict.checks_passed + verdict.checks_failed,
                "position_mult": round(verdict.position_multiplier, 4)}
    test_step("WisdomGate", "200 principles (passing context)", _passing, verbose)

    # Test with failing context
    def _failing():
        from livermore_principles import WisdomGate, TradeContext
        gate = WisdomGate()
        ctx = TradeContext(
            symbol='BTCUSD', direction='BUY', confidence=0.20,
            win_rate=0.30, rr_ratio=0.8, atr_ratio=0.08,
            bars_since_last_trade=1, spread_bps=25.0,
            regime='crisis', drawdown_pct=20.0,
            recent_losses=5, recent_wins=0,
            news_pending=True,
            emotional_market=True,
            is_averaging_down=True,
            # Add all required fields with failing values
            capital_flow_score=0.1, relative_strength_rank=0.1,
            market_breadth=0.1, mtf_high_tf_agrees=False,
            signal_rank_percentile=0.1, smart_money_score=0.1,
            execution_quality_score=0.1, conviction_level=0.1,
            noise_filter_passed=False, historical_match_count=15,
            historical_win_rate=0.20, regime_strategy_match=False,
            risk_allocation_pct=8.0, strategy_decay_detected=True,
            portfolio_correlation_avg=0.9, portfolio_diversification=0.1,
            adaptive_rules_active=False, consistency_score=0.1,
            weekly_audit_passed=False, weekly_audit_gpa=1.0,
            market_context_score=0.1, context_understood=False,
            capital_efficiency=0.1, strategy_edge_declining=True,
            volatility_regime='extreme', liquidity_quality=0.1,
            daily_risk_budget_remaining=0.0, daily_risk_budget_used=3.0,
            correlated_exposure_pct=0.8, adaptive_confidence=0.1,
            decision_quality_score=0.1, execution_latency_ms=3000,
            portfolio_balance_score=0.1, consecutive_loss_count=5,
            execution_window_quality=0.1, structural_change_detected=True,
            learning_loop_active=False, risk_adjusted_return_target=0.5,
            survival_mode_active=True, market_cycle='decline',
            cycle_confidence=0.2, probability_buy=0.98, probability_sell=0.0,
            probability_wait=0.02, structure_priority_score=0.1,
            portfolio_risk_usd=5000, dynamic_risk_mode='minimum',
            false_confidence_detected=True, liquidity_asset_score=0.1,
            idle_mode=True, strategy_evolution_active=False,
            edge_decay_rate=0.5, allocation_diversified=False,
            knowledge_added=False, institutional_memory_size=0,
            black_swan_prepared=False, opportunity_cost_acceptable=False,
            self_diagnosis_passed=False, benchmark_outperformance=-2.0,
            decision_engine_consensus=0.1, autonomous_mode=False,
            patience_mode=True, opportunity_rank=0.1,
            discipline_score=0.1, trend_persistence_score=0.1,
            noise_filtered=False, multi_confirmation_count=1,
            dynamic_exit_ready=False, near_miss_analyzed=False,
            market_memory_available=False, confidence_earned=False,
            market_fatigue_detected=True, capital_protection_active=False,
            execution_optimized=False, portfolio_correlation_managed=False,
            strategy_switched=False, rl_loop_active=False,
            performance_dashboard_active=False, decision_quality_focus=False,
            autonomous_improvement=False, institutional_mindset_complete=False,
        )
        verdict = gate.evaluate(ctx)
        return {"approved": verdict.approved,
                "passed": verdict.checks_passed,
                "failed": verdict.checks_failed,
                "total": verdict.checks_passed + verdict.checks_failed}
    test_step("WisdomGate", "200 principles (failing context)", _failing, verbose)


def test_signals(verbose: bool) -> None:
    """Test 4: Signal v3 (immutable decision contract)."""
    section("STEP 4: Signal v3 (188 fields, 25 dataclasses)")

    def _create():
        from engine.signals_v3 import Signal, Action, SignalQuality
        sig = Signal.buy("BTCUSD", "M15", strength=0.85, price=43250,
                        stop_loss=42500, take_profit=45000,
                        strategy_id="Momentum_v4", strategy_version="4.1")
        return {"action": sig.action.value, "strength": sig.strength,
                "symbol": sig.symbol, "signal_id": sig.signal_id[:8]}
    test_step("Signals", "factory buy()", _create, verbose)

    def _builder():
        from engine.signals_v3 import SignalBuilder, Action, SignalQuality
        sig = (SignalBuilder()
               .with_symbol("ETHUSD", "M15")
               .with_action(Action.SELL, strength=0.78, quality=SignalQuality.A_PLUS)
               .with_price(2580.0)
               .with_sl_tp(2650, 2400)
               .with_strategy("Transformer_v9", "9.0.0")
               .with_confidence(overall=0.82, trend=0.85, momentum=0.78)
               .build())
        return {"action": sig.action.value, "quality": sig.quality.value,
                "strategy": sig.identity.strategy_id}
    test_step("Signals", "SignalBuilder pattern", _builder, verbose)

    def _serialize():
        from engine.signals_v3 import Signal
        sig = Signal.buy("BTCUSD", "M15", strength=0.75, price=43250)
        d = sig.to_dict()
        json_str = sig.to_json()
        sig2 = Signal.from_json(json_str)
        return {"dict_keys": len(d), "json_len": len(json_str),
                "roundtrip": sig2.signal_id == sig.signal_id}
    test_step("Signals", "serialization roundtrip", _serialize, verbose)


def _safe_delete_db(db_path: str) -> bool:
    """Safely delete a corrupted DB file, handling Windows file locking.

    Returns True if deleted, False if couldn't (locked or other error).
    """
    import sqlite3
    # First, try to close any open connections by connecting with WAL mode
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass

    # Wait a moment for Windows to release the file handle
    import time
    time.sleep(0.1)

    # Try multiple times to delete (Windows may take a moment to release)
    for attempt in range(5):
        try:
            os.remove(db_path)
            return True
        except PermissionError:
            # File is locked by another process — wait and retry
            time.sleep(0.5)
        except FileNotFoundError:
            return True  # Already deleted
        except Exception:
            return False

    # If still can't delete, try renaming it
    try:
        corrupted_name = db_path + ".corrupted"
        os.rename(db_path, corrupted_name)
        print(f"         {YELLOW}WARNING: Couldn't delete {db_path} (locked), renamed to .corrupted{RESET}")
        return True
    except Exception:
        print(f"         {RED}WARNING: Couldn't delete or rename {db_path} — it may be locked by another process.{RESET}")
        print(f"         {YELLOW}Fix: Close any running bot instances, then run: del data\\*.db{RESET}")
        return False


def _clean_corrupted_dbs() -> None:
    """Clean all corrupted SQLite databases in data/ directory."""
    import glob, sqlite3
    for db in glob.glob("data/*.db"):
        is_corrupted = False
        try:
            conn = sqlite3.connect(db, timeout=1.0)
            conn.execute("SELECT 1").fetchone()
            conn.close()
        except Exception:
            is_corrupted = True

        if is_corrupted:
            print(f"         {YELLOW}Found corrupted DB: {db} — attempting cleanup...{RESET}")
            _safe_delete_db(db)


def test_master_orchestrator(verbose: bool) -> None:
    """Test 5: TradingBot boot + layer check.

    BUG FIX: this used to import architecture.master_orchestrator.
    MasterOrchestrator, which main.py deliberately quarantines on every
    boot (see main.py's "CRITICAL REGRESSION GUARD") because that v9
    pipeline was retired in favor of architecture.integration.TradingBot.
    So this test — on a perfectly healthy, correctly-quarantined install —
    always failed with "No module named 'architecture.master_orchestrator'".
    Now it boots the orchestrator that's actually live.
    """
    section("STEP 5: TradingBot Orchestrator")

    # Clean corrupted DBs first (Windows-safe)
    _clean_corrupted_dbs()

    def _boot():
        from architecture.integration import TradingBot
        cfg = {
            "capital": 10000.0,
            "runtime": {"snapshot_dir": "data/snapshots"},
            "symbols": [{"name": "BTCUSD"}, {"name": "ETHUSD"}, {"name": "EURUSD"}],
            "symbols_auto_load": False,
        }
        bot = TradingBot(cfg, mode="paper")
        ok = bot.boot()
        n_symbols = len(getattr(bot, "_symbols", []))
        bot.shutdown("diagnose.py boot test done")
        return {"booted": ok, "symbols_loaded": n_symbols}
    test_step("Orchestrator", "boot (TradingBot, paper mode)", _boot, verbose)

    def _layers():
        from architecture.integration import TradingBot
        cfg = {
            "capital": 10000.0,
            "symbols": [{"name": "BTCUSD"}],
            "symbols_auto_load": False,
        }
        bot = TradingBot(cfg, mode="paper")
        bot.boot()
        # These are the real attribute names TradingBot wires up in
        # __init__ (verified against architecture/integration.py) —
        # NOT the old MasterOrchestrator._modules dict layout, which no
        # longer exists.
        layers = {
            "foundation": ["state_machine", "event_bus", "snapshot_engine",
                           "recovery_engine", "config_versioning"],
            "persistence": ["db", "idempotency"],
            "perception": ["feature_pipeline", "regime_orchestrator"],
            "brain": ["ai_model_manager", "multi_agent", "memory_system",
                      "online_learner"],
            "hands": ["exchange", "portfolio"],
            "defense": ["risk_pipeline", "breakers", "self_healing",
                        "decision_auditor"],
            "observability": ["monitor", "simulation"],
        }
        results = {}
        for layer, expected in layers.items():
            present = sum(1 for attr in expected if getattr(bot, attr, None) is not None)
            results[layer] = f"{present}/{len(expected)}"
        bot.shutdown("diagnose.py layer test done")
        return results
    test_step("Orchestrator", "layer wiring check", _layers, verbose)


def test_pipeline_steps(verbose: bool) -> None:
    """Test 6: Full trading pipeline step-by-step."""
    section("STEP 6: Full Trading Pipeline (28 steps)")

    df = make_test_data()

    # Step 1: Feature computation
    def _step1():
        from architecture.feature_pipeline import build_default_pipeline
        pipe = build_default_pipeline()
        fv = pipe.compute("BTCUSD", df)
        return {"features": len(fv.features), "warmed_up": fv.is_warmed_up}
    test_step("Pipeline", "1. Feature computation", _step1, verbose)

    # Step 2: Market context
    def _step2():
        from trading_modules.market_context_engine import MarketContextEngine
        engine = MarketContextEngine()
        ctx = engine.evaluate(df, spread_bps=2.5, session="london")
        return {"score": round(ctx.context_score, 1),
                "regime": ctx.market_regime.value,
                "can_trade": ctx.can_trade}
    test_step("Pipeline", "2. Market context evaluation", _step2, verbose)

    # Step 3: Market cycle
    def _step3():
        from trading_modules.market_cycle_engine import MarketCycleEngine
        engine = MarketCycleEngine()
        cycle = engine.detect(df)
        return {"phase": cycle.phase.value, "confidence": round(cycle.confidence, 2)}
    test_step("Pipeline", "3. Market cycle detection", _step3, verbose)

    # Step 4: Market phase (Wyckoff)
    def _step4():
        from trading_modules.market_phase_detector import MarketPhaseDetector
        det = MarketPhaseDetector()
        phase = det.detect(df)
        return {"phase": phase.phase.value, "confidence": round(phase.confidence, 2)}
    test_step("Pipeline", "4. Market phase (Wyckoff)", _step4, verbose)

    # Step 5: Trend fatigue
    def _step5():
        from trading_modules.trend_fatigue_detector import TrendFatigueDetector
        det = TrendFatigueDetector()
        fatigue = det.detect(df, trend_direction="up")
        return {"score": fatigue.score, "level": fatigue.level}
    test_step("Pipeline", "5. Trend fatigue detection", _step5, verbose)

    # Step 6: Capital flow
    def _step6():
        from trading_modules.capital_flow_analyzer import CapitalFlowAnalyzer
        analyzer = CapitalFlowAnalyzer()
        flow = analyzer.analyze(df)
        return {"type": flow.flow_type.value, "strength": round(flow.strength, 1)}
    test_step("Pipeline", "6. Capital flow analysis", _step6, verbose)

    # Step 7: Smart money
    def _step7():
        from trading_modules.smart_money_detector import SmartMoneyDetector
        det = SmartMoneyDetector()
        sm = det.detect(df, spread_bps=2.5)
        return {"score": sm.smart_money_score, "direction": sm.inferred_direction}
    test_step("Pipeline", "7. Smart money detection", _step7, verbose)

    # Step 8: Multi-agent consensus
    def _step8():
        from architecture.multi_agent import build_default_coordinator
        from architecture.feature_pipeline import FeatureVector
        coord = build_default_coordinator()
        fv = FeatureVector(symbol="BTCUSD", timestamp="now", bar_close=43250,
                          features={"ema_9": 43100, "ema_21": 43000,
                                   "ema_50": 42800, "adx_14": 30,
                                   "supertrend": 42900, "macd": 5,
                                   "macd_signal": 3, "roc_10": 1.2,
                                   "rsi_14": 60, "rvol": 1.5,
                                   "stoch_rsi": 0.6, "bb_width": 0.03,
                                   "fvg_present": False, "order_block": False,
                                   "regime": "trend_up", "atr_pct": 0.015})
        consensus = coord.evaluate("BTCUSD", df, fv, {"equity": 10000})
        return {"action": consensus.action, "strength": round(consensus.strength, 2),
                "agreement": round(consensus.agreement_score, 2)}
    test_step("Pipeline", "8. Multi-agent consensus (5 agents)", _step8, verbose)

    # Step 9: Opportunity ranking
    def _step9():
        from trading_modules.trade_opportunity_ranker import TradeOpportunityRanker
        ranker = TradeOpportunityRanker()
        opp = ranker.score_opportunity("BTCUSD", df, "BUY",
                                       spread_bps=2.5, session="london")
        return {"score": opp.score, "tier": opp.tier}
    test_step("Pipeline", "9. Opportunity ranking (0-100)", _step9, verbose)

    # Step 10: Setup scoring
    def _step10():
        from trading_modules.setup_scoring_engine import SetupScoringEngine
        engine = SetupScoringEngine()
        score = engine.score(df, "BUY", spread_bps=2.5, session="london",
                            has_pullback=True, news_minutes=180)
        return {"total": score.total, "passed": score.passed,
                "mult": score.position_multiplier}
    test_step("Pipeline", "10. Setup scoring (5 dimensions)", _step10, verbose)

    # Step 11: Opportunity cost
    def _step11():
        from trading_modules.opportunity_cost_analyzer import OpportunityCostAnalyzer
        analyzer = OpportunityCostAnalyzer()
        oc = analyzer.evaluate(current_score=75, expected_better_setup_minutes=30)
        return {"decision": oc.decision.value, "ev_waiting": round(oc.expected_value_of_waiting, 3)}
    test_step("Pipeline", "11. Opportunity cost analysis", _step11, verbose)

    # Step 12: Emotion filter
    def _step12():
        from trading_modules.emotion_volatility_filter import EmotionVolatilityFilter
        filt = EmotionVolatilityFilter()
        state = filt.detect(df, spread_bps=3.0)
        return {"emotion": state.emotion.value, "mode": state.mode.value}
    test_step("Pipeline", "12. Emotion & volatility filter", _step12, verbose)

    # Step 13: Risk budget
    def _step13():
        from trading_modules.risk_budget_manager import RiskBudgetManager
        mgr = RiskBudgetManager(equity=10000)
        allowed, reason = mgr.can_take_trade(risk_usd=50)
        return {"allowed": allowed, "reason": reason,
                "remaining": mgr.state().daily_risk_remaining}
    test_step("Pipeline", "13. Risk budget check", _step13, verbose)

    # Step 14: Expected value
    def _step14():
        from trading_modules.expected_value_calculator import ExpectedValueCalculator
        calc = ExpectedValueCalculator()
        ev = calc.calculate(win_rate=0.62, avg_win_r=1.8, avg_loss_r=1.0,
                           sample_size=50, account_equity=10000)
        return {"ev_r": round(ev.ev_per_trade_r, 3),
                "kelly": round(ev.kelly_fraction, 3),
                "ror": round(ev.risk_of_ruin, 4)}
    test_step("Pipeline", "14. Expected value calculation", _step14, verbose)

    # Step 15: Institutional decision engine
    def _step15():
        from trading_modules.institutional_decision_engine import InstitutionalDecisionEngine
        engine = InstitutionalDecisionEngine()
        decision = engine.evaluate(
            structure_score=0.8, liquidity_score=0.75, order_flow_score=0.7,
            volatility_score=0.7, correlation_score=0.8, macro_score=0.75,
            risk_budget_score=0.9, execution_score=0.8,
        )
        return {"decision": decision.decision.value,
                "consensus": round(decision.consensus, 2)}
    test_step("Pipeline", "15. Institutional decision engine (8D)", _step15, verbose)

    # Step 16: Decision intelligence
    def _step16():
        from trading_modules.decision_intelligence_layer import DecisionIntelligenceLayer
        layer = DecisionIntelligenceLayer()
        di = layer.decide(
            market_context_score=0.8, trend_score=0.75, liquidity_score=0.7,
            order_flow_score=0.6, volatility_score=0.7, correlation_score=0.8,
            execution_score=0.75, portfolio_risk_score=0.85,
            expected_r_if_win=2.0, probability_win=0.65,
        )
        return {"action": di.action.value, "quality": round(di.decision_quality, 1),
                "ev_r": round(di.expected_value_r, 2)}
    test_step("Pipeline", "16. Decision intelligence layer", _step16, verbose)

    # Step 17: Dynamic exit
    def _step17():
        from trading_modules.dynamic_exit_intelligence import DynamicExitIntelligence
        exit_ai = DynamicExitIntelligence()
        rec = exit_ai.evaluate(
            "BUY", 43250, 43800, 42500, 45000, df,
            r_multiple=1.2, hold_time_bars=15,
        )
        return {"action": rec.action.value, "new_stop": rec.new_stop}
    test_step("Pipeline", "17. Dynamic exit intelligence", _step17, verbose)

    # Step 18: Strategy router
    def _step18():
        from trading_modules.adaptive_strategy_router import AdaptiveStrategyRouter
        router = AdaptiveStrategyRouter()
        route = router.route(regime="trend_up", volatility_regime="normal")
        return {"strategy": route.strategy.value, "action": route.action}
    test_step("Pipeline", "18. Adaptive strategy router", _step18, verbose)

    # Step 19: Portfolio allocation
    def _step19():
        from trading_modules.portfolio_allocation_optimizer import PortfolioAllocationOptimizer
        opt = PortfolioAllocationOptimizer(equity=10000)
        alloc = opt.optimize(market_cycle="expansion", risk_budget_remaining=0.8)
        return {"deployed": round(alloc.total_deployed, 2),
                "diversification": round(alloc.diversification_score, 2)}
    test_step("Pipeline", "19. Portfolio allocation optimizer", _step19, verbose)

    # Step 20: Performance analytics
    def _step20():
        from trading_modules.institutional_performance_analytics import InstitutionalPerformanceAnalytics
        analytics = InstitutionalPerformanceAnalytics(initial_equity=10000)
        np.random.seed(42)
        for _ in range(20):
            pnl = np.random.uniform(-30, 50)
            analytics.record_trade(pnl=pnl, r_multiple=pnl/30)
        report = analytics.report()
        return {"trades": report.total_trades, "grade": report.grade,
                "sharpe": round(report.sharpe, 2)}
    test_step("Pipeline", "20. Performance analytics (Sharpe/Sortino)", _step20, verbose)

    # Step 21: System health
    def _step21():
        from trading_modules.system_health_monitor import SystemHealthMonitor
        mon = SystemHealthMonitor()
        mon.check_disk_space("/")
        mon.check_memory()
        mon.check_cpu()
        h = mon.health_summary()
        return {"status": h["status"], "can_trade": h["can_trade"],
                "failed": len(h["failed_components"])}
    test_step("Pipeline", "21. System health monitor", _step21, verbose)

    # Step 22: Self-diagnosis
    def _step22():
        from trading_modules.ai_self_diagnosis import AISelfDiagnosis
        diag = AISelfDiagnosis(min_trades_for_diagnosis=5)
        np.random.seed(42)
        for _ in range(10):
            diag.record_trade("momentum", "london", "trend_up",
                            np.random.uniform(-20, 50), np.random.uniform(-0.5, 1.5),
                            0.7)
        report = diag.diagnose()
        return {"health": round(report.overall_health, 1),
                "weaknesses": len(report.top_weaknesses)}
    test_step("Pipeline", "22. AI self-diagnosis", _step22, verbose)

    # Step 23: Memory system
    def _step23():
        from architecture.memory_system import MemorySystem
        import os
        mem = MemorySystem(db_path="data/test_diag.db")
        for i in range(5):
            mem.encode_episode("BTCUSD", "M15", "BUY",
                              {"rsi": 60+i}, "trend_up",
                              43250, 43800, 42500, 45000, 0.1,
                              3600, 50.0, "momentum")
        similar = mem.retrieve_similar({"rsi": 62}, symbol="BTCUSD", top_k=3)
        _safe_delete_db("data/test_diag.db")
        return {"episodes": mem.stats()["episodic_count"], "similar_found": len(similar)}
    test_step("Pipeline", "23. Memory system (episodic)", _step23, verbose)

    # Step 24: Decision audit
    def _step24():
        from architecture.decision_audit import DecisionAuditor
        import os
        audit = DecisionAuditor(db_path="data/test_audit.db")
        aid = audit.start_decision("BTCUSD", 1, {"rsi": 62}, 10000, 0, 3.0, 43250)
        audit.finalize_decision(aid, approved=True, lots=0.1, entry_price=43250)
        records = audit.query("BTCUSD")
        _safe_delete_db("data/test_audit.db")
        return {"audit_id": aid[:8], "records": len(records)}
    test_step("Pipeline", "24. Decision audit trail", _step24, verbose)

    # Step 25: Continuous improvement
    def _step25():
        from trading_modules.continuous_improvement_system import ContinuousImprovementSystem
        cis = ContinuousImprovementSystem()
        cis.set_equity(10000)
        np.random.seed(42)
        for _ in range(25):
            cis.record_trade("momentum", "BTCUSD", 50, 1.5, 0.7)
        daily = cis.daily_review()
        return {"trend": daily.performance_trend,
                "improvement": round(daily.improvement_score, 3)}
    test_step("Pipeline", "25. Continuous improvement system", _step25, verbose)

    # Step 26: Snapshot engine
    def _step26():
        from architecture.recovery_engine import SnapshotEngine
        snap = SnapshotEngine(snapshot_dir="data/test_snapshots")
        sid = snap.take_snapshot(cycle=1, equity=10000, notes="test")
        latest = snap.latest_snapshot()
        snap.mark_clean_shutdown()
        import shutil
        try:
            shutil.rmtree("data/test_snapshots", ignore_errors=True)
        except:
            pass
        return {"snapshot_id": sid[:12] if sid else None,
                "restored": latest is not None}
    test_step("Pipeline", "26. Snapshot & recovery engine", _step26, verbose)

    # Step 27: Event bus
    def _step27():
        from architecture.event_bus import EventBus, EventType
        bus = EventBus()
        received = []
        bus.subscribe(EventType.POSITION_OPENED, lambda e: received.append(e))
        bus.emit(EventType.POSITION_OPENED, {"symbol": "BTCUSD"}, "test")
        return {"emitted": bus.metrics()["total_emitted"],
                "received": len(received)}
    test_step("Pipeline", "27. Event bus (pub/sub)", _step27, verbose)

    # Step 28: State machine
    def _step28():
        from architecture.state_machine import StateMachine, BotState
        sm = StateMachine()
        sm.transition(BotState.CONNECTING)
        sm.transition(BotState.SYNCING)
        sm.transition(BotState.WARMUP)
        sm.transition(BotState.LIVE)
        return {"state": sm.current.value, "can_trade": sm.can_trade(),
                "transitions": sm.transition_count}
    test_step("Pipeline", "28. State machine (10 states)", _step28, verbose)


def test_end_to_end(verbose: bool) -> None:
    """Test 7: End-to-end cycle with synthetic data.

    BUG FIX: was importing the retired architecture.master_orchestrator.
    MasterOrchestrator (quarantined by main.py — see test_master_orchestrator
    above for the full explanation). Rewritten against the live TradingBot,
    whose cycle() takes no df_dict argument — it pulls data itself through
    self.exchange, so synthetic data is seeded via PaperAdapter.set_candle_data()
    before calling cycle(), matching how TradingBot actually consumes data.
    """
    section("STEP 7: End-to-End Cycle (TradingBot)")

    # Clean corrupted DBs (Windows-safe)
    _clean_corrupted_dbs()

    def _e2e():
        from architecture.integration import TradingBot
        cfg = {
            "capital": 10000.0,
            "runtime": {"snapshot_dir": "data/snapshots"},
            "symbols": [{"name": "BTCUSD"}, {"name": "ETHUSD"}, {"name": "EURUSD"}],
            "symbols_auto_load": False,
        }
        bot = TradingBot(cfg, mode="paper")
        bot.boot()

        # Seed synthetic data for 3 symbols into the PaperAdapter — this is
        # how TradingBot.cycle() actually sources candles in paper mode
        # (it calls self.exchange internally; there's no df_dict param).
        for sym, seed in (("BTCUSD", 1), ("ETHUSD", 2), ("EURUSD", 3)):
            bot.exchange.set_candle_data(sym, make_test_data(seed=seed))

        result = bot.cycle()
        bot.shutdown("e2e test done")
        return {
            "cycle": result.cycle,
            "state": result.state,
            "regime": result.regime,
            "trades_placed": result.trades_placed,
            "trades_rejected": result.trades_rejected,
            "time_ms": round(result.cycle_time_ms, 0),
        }
    test_step("E2E", "full cycle (3 symbols)", _e2e, verbose)


def test_mt5_connector(verbose: bool) -> None:
    """Test 8: MT5 connector (if available)."""
    section("STEP 8: MT5 Connector (if available)")

    def _import():
        try:
            import MetaTrader5 as mt5
            return {"mt5_available": True, "version": mt5.__version__}
        except ImportError:
            return {"mt5_available": False, "reason": "MetaTrader5 not installed"}
    test_step("MT5", "MetaTrader5 import", _import, verbose)

    def _connector():
        # Audit-fix C3: NO hard-coded credentials. Read MT5 credentials from
        # environment variables (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
        # MT5_TERMINAL_PATH) — fall back to the config.yaml values via
        # config_loader if the env vars are absent. If neither source
        # provides credentials, skip the connector test with a clear note
        # instead of silently using a real account.
        from brokers.mt5_connector import MT5Connector, MT5Unavailable
        import os as _os
        login = _os.environ.get("MT5_LOGIN", "")
        password = _os.environ.get("MT5_PASSWORD", "")
        server = _os.environ.get("MT5_SERVER", "Deriv-Demo")
        terminal_path = _os.environ.get(
            "MT5_TERMINAL_PATH",
            r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe")
        if not (login and password):
            # Try loading from config.yaml as a fallback.
            try:
                from config_loader import load_config, ConfigError
                cfg = load_config(validate=False)
                mt5_cfg = cfg.get("mt5", {})
                login = str(mt5_cfg.get("login", "") or "")
                password = mt5_cfg.get("password", "") or ""
                server = mt5_cfg.get("server", server)
                terminal_path = mt5_cfg.get("terminal_path", terminal_path)
            except Exception:
                pass
        if not (login and password):
            return {
                "status": "skipped — set MT5_LOGIN/MT5_PASSWORD env vars "
                          "or define mt5.* in config.yaml (audit C3: no hard-coded creds)"
            }
        try:
            conn = MT5Connector(
                login=int(login), password=password,
                server=server, terminal_path=terminal_path,
            )
            return {"class": type(conn).__name__, "login": conn.login}
        except MT5Unavailable:
            return {"status": "MT5 not available on this system"}
        except (ValueError, TypeError) as e:
            return {"status": f"invalid MT5 config: {e!r}"}
    test_step("MT5", "connector creation", _connector, verbose)


def test_database(verbose: bool) -> None:
    """Test 9: Database operations."""
    section("STEP 9: Database (SQLite persistence)")

    def _init():
        from database import Database
        import os
        os.makedirs("data", exist_ok=True)
        db = Database("data/test_diag.db")
        return {"tables": "created"}
    test_step("Database", "initialization", _init, verbose)

    def _trade():
        from database import Database
        import os
        db = Database("data/test_diag.db")
        db.save_trade_open(ticket=12345, symbol="BTCUSD", action="BUY",
                          lots=0.1, entry_price=43250, stop_loss=42500,
                          take_profit=45000, magic=100000, mode="demo",
                          gate_score=85, grade="A", strategy_type="momentum",
                          rsi=62, ema_fast=43100, ema_slow=42800, atr=450)
        db.save_trade_close(ticket=12345, exit_price=44000, pnl=75,
                           reason="TP_HIT")
        stats = db.get_stats()
        _safe_delete_db("data/test_diag.db")
        return stats
    test_step("Database", "trade open + close + stats", _trade, verbose)


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Full project diagnostic")
    parser.add_argument("--quick", action="store_true", help="quick check (imports only)")
    parser.add_argument("--verbose", action="store_true", help="show all details")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}")
    print("=" * 70)
    print("  INDUSTRIAL AI TRADING AGENT v9.0 — FULL DIAGNOSTIC")
    print(f"  {datetime.now(tz=timezone.utc).isoformat()}")
    print("=" * 70)
    print(f"{RESET}")

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Clean corrupted DBs BEFORE anything else (Windows-safe)
    print(f"\n{YELLOW}  Pre-check: Cleaning corrupted databases...{RESET}")
    _clean_corrupted_dbs()
    print()

    # Run tests
    test_imports(args.verbose)

    if not args.quick:
        test_indicators(args.verbose)
        test_wisdom_gate(args.verbose)
        test_signals(args.verbose)
        test_master_orchestrator(args.verbose)
        test_pipeline_steps(args.verbose)
        test_end_to_end(args.verbose)
        test_mt5_connector(args.verbose)
        test_database(args.verbose)

    summary()


if __name__ == "__main__":
    main()
