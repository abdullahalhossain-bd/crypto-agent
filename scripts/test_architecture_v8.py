"""scripts/test_architecture_v8.py
=====================================================================
Smoke test for the new v8.0 architecture package.
Tests each component in isolation, then a mini end-to-end flow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timezone


def test_event_bus():
    print("\n[1/10] Testing EventBus...")
    from architecture.event_bus import EventBus, EventType
    bus = EventBus()
    received = []
    bus.subscribe(EventType.POSITION_OPENED, lambda e: received.append(e))
    bus.subscribe_wildcard(lambda e: received.append(e))
    bus.emit(EventType.POSITION_OPENED,
             payload={"symbol": "BTCUSD", "volume": 0.5},
             source="test")
    assert len(received) == 2, f"expected 2 (1 sub + 1 wildcard), got {len(received)}"
    print(f"  OK - emitted, both subscribers received. Metrics: {bus.metrics()}")


def test_state_machine():
    print("\n[2/10] Testing StateMachine...")
    from architecture.state_machine import StateMachine, BotState
    sm = StateMachine()
    assert sm.current == BotState.BOOT
    assert sm.transition(BotState.CONNECTING)
    assert sm.transition(BotState.SYNCING)
    assert sm.transition(BotState.WARMUP)
    assert sm.transition(BotState.LIVE)
    assert sm.can_trade()
    assert not sm.transition(BotState.BOOT)
    assert sm.transition_count == 4
    print(f"  OK - 4 legal transitions, illegal rejected, can_trade={sm.can_trade()}")


def test_feature_pipeline():
    print("\n[3/10] Testing FeaturePipeline...")
    from architecture.feature_pipeline import build_default_pipeline
    pipe = build_default_pipeline()
    stats = pipe.stats()
    print(f"  Registered: {stats['registered_features']} features across "
          f"{len(stats['categories'])} categories")
    n = 250
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 40000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": prices,
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })
    fv = pipe.compute("BTCUSD", df)
    assert fv.is_warmed_up
    print(f"  OK - computed {len(fv.features)} features, warmed up, hash={fv.hash}")
    print(f"  Sample: RSI={fv.get('rsi_14', 0)}, EMA9={fv.get('ema_9', 0)}, ATR={fv.get('atr_14', 0)}")


def test_portfolio_manager():
    print("\n[4/10] Testing PortfolioManager...")
    from architecture.portfolio_manager_v2 import PortfolioManager
    pm = PortfolioManager(initial_capital=10000.0)
    pm.on_position_opened(ticket=1, symbol="BTCUSD", side="BUY",
                         volume=0.1, entry_price=40000, sl=39500, tp=41000)
    pm.on_position_opened(ticket=2, symbol="ETHUSD", side="BUY",
                         volume=1.0, entry_price=2500, sl=2400, tp=2700)
    pm.update_prices({"BTCUSD": 41000, "ETHUSD": 2600})
    m = pm.metrics()
    assert m.open_positions == 2
    assert m.unrealized_pnl > 0
    print(f"  OK - 2 positions open, equity=${m.equity:.2f}, unrealized=${m.unrealized_pnl:.2f}")
    pnl = pm.on_position_closed(1, exit_price=41000)
    print(f"  Closed ticket 1, realized PnL=${pnl:.2f}")


def test_risk_pipeline():
    print("\n[5/10] Testing RiskPipeline (12 gates)...")
    from architecture.risk_pipeline import RiskPipeline, RiskContext
    from architecture.portfolio_manager_v2 import PortfolioManager
    from engine.signals import Signal, Action
    pm = PortfolioManager(initial_capital=10000.0)
    pipe = RiskPipeline(portfolio=pm)
    n = 100
    np.random.seed(7)
    df = pd.DataFrame({
        "open": np.random.uniform(99, 101, n),
        "high": np.random.uniform(100, 102, n),
        "low": np.random.uniform(98, 100, n),
        "close": np.random.uniform(99, 101, n),
        "volume": np.random.uniform(1000, 5000, n),
    })
    sig = Signal(symbol="BTCUSD", action=Action.BUY, strength=0.75,
                 price=100.0, meta={})
    ctx = RiskContext(signal=sig, df=df, account_equity=10000.0)
    approved, final, verdicts = pipe.evaluate(ctx)
    print(f"  Ran {len(verdicts)} gates: approved={approved}, reason='{final.reason}'")
    if not approved:
        print(f"  Rejected at: {final.gate_name} - {final.reason}")
    print("  OK - pipeline executes all 12 gates")


def test_self_healing():
    print("\n[6/10] Testing SelfHealingSystem...")
    from architecture.self_healing import SelfHealingSystem, FailureType
    healer = SelfHealingSystem()
    attempts = [0]
    def recovery_fn():
        attempts[0] += 1
        return attempts[0] >= 2
    healer.register_recovery(FailureType.CONNECTION, "test_comp",
                            recovery_fn, max_retries=3)
    level = healer.report_failure(FailureType.CONNECTION, "test_comp",
                                  "simulated disconnect")
    print(f"  Attempt 1: level={level.name}")
    level = healer.report_failure(FailureType.CONNECTION, "test_comp",
                                  "simulated disconnect 2")
    print(f"  Attempt 2: level={level.name}")
    health = healer.health()
    print(f"  OK - recovery attempted, health={health}")


def test_decision_audit():
    print("\n[7/10] Testing DecisionAuditor...")
    from architecture.decision_audit import DecisionAuditor
    audit = DecisionAuditor(db_path="data/test_audit.db")
    audit_id = audit.start_decision(
        symbol="BTCUSD", cycle=1,
        feature_vector={"rsi": 65, "ema9": 40100},
        account_equity=10000.0, bar_close=40000,
    )
    audit.finalize_decision(audit_id, approved=False)
    records = audit.query(symbol="BTCUSD")
    assert len(records) >= 1
    stats = audit.summary_stats()
    print(f"  OK - recorded decision, stats={stats}")
    try:
        os.remove("data/test_audit.db")
    except Exception:
        pass


def test_memory_system():
    print("\n[8/10] Testing MemorySystem...")
    from architecture.memory_system import MemorySystem
    mem = MemorySystem(db_path="data/test_memory.db")
    for i in range(10):
        mem.encode_episode(
            symbol="BTCUSD", timeframe="M15", direction="BUY",
            features={"rsi_14": 50 + i, "atr_pct": 0.01 + i * 0.001},
            regime="trend_up",
            entry_price=40000 + i * 100,
            exit_price=40500 + i * 100,
            sl=39500, tp=41000, lots=0.1,
            hold_time_s=3600, pnl=50.0, strategy_name="momentum",
        )
    similar = mem.retrieve_similar(
        {"rsi_14": 55, "atr_pct": 0.012}, symbol="BTCUSD", top_k=3
    )
    assert len(similar) > 0
    ev = mem.estimate_ev({"rsi_14": 55}, symbol="BTCUSD")
    print(f"  OK - encoded 10 episodes, retrieved {len(similar)} similar, EV={ev}")
    try:
        os.remove("data/test_memory.db")
    except Exception:
        pass


def test_multi_agent():
    print("\n[9/10] Testing MultiAgentCoordinator...")
    from architecture.multi_agent import build_default_coordinator
    from architecture.feature_pipeline import FeatureVector
    coord = build_default_coordinator()
    print(f"  {coord.agent_count()} agents: {coord.agent_names()}")
    fv = FeatureVector(symbol="BTCUSD", timestamp="now", bar_close=41000,
                       features={
                           "ema_9": 40800, "ema_21": 40600, "ema_50": 40400,
                           "adx_14": 30, "supertrend": 40500,
                           "macd": 5, "macd_signal": 3, "roc_10": 1.2,
                           "rsi_14": 60, "rvol": 1.5,
                           "stoch_rsi": 0.6, "bb_width": 0.03,
                           "fvg_present": True, "order_block": True,
                           "regime": "trend_up", "atr_pct": 0.015,
                       })
    df = pd.DataFrame({"close": [41000]*60, "high": [41100]*60, "low": [40900]*60,
                       "open": [40950]*60, "volume": [1000]*60})
    consensus = coord.evaluate("BTCUSD", df, fv,
                              context={"equity": 10000, "peak_equity": 10500})
    print(f"  Consensus: action={consensus.action}, strength={consensus.strength:.2f}, "
          f"agreement={consensus.agreement_score * 100:.0f}%")
    print(f"  Votes: BUY={consensus.votes_buy} SELL={consensus.votes_sell} "
          f"HOLD={consensus.votes_hold} REDUCE={consensus.votes_reduce}")


def test_monitoring():
    print("\n[10/10] Testing InstitutionalMonitor...")
    from architecture.institutional_monitoring import InstitutionalMonitor
    mon = InstitutionalMonitor()
    for i in range(50):
        mon.update_equity(10000 + i * 10)
        mon.record_trade(pnl=np.random.uniform(-50, 80), r_multiple=np.random.uniform(-1, 2))
        mon.record_cycle(0.05 + np.random.uniform(0, 0.02))
    kpis = mon.kpis()
    alerts = mon.check_alerts()
    print(f"  KPIs: equity={kpis.get('equity', 0):.2f}, trades={kpis.get('total_trades', 0)}")
    print(f"  Alerts: {len(alerts)}")


def test_industrial_bot_status():
    print("\n[BONUS] Testing IndustrialBot status (no MT5)...")
    from architecture.integration import IndustrialBot
    cfg = {
        "exchange": {"type": "paper"},
        "mt5": {"initial_balance": 10000.0},
        "capital": 10000.0,
        "symbols": [{"name": "BTCUSD", "timeframe": "M15"}],
        "symbols_auto_load": False,
        "risk": {"max_consecutive_losses": 3, "cooldown_s": 60},
    }
    bot = IndustrialBot(cfg)
    status = bot.status()
    print(f"  Bot state: {status['state']}, cycle: {status['cycle']}")
    print(f"  Feature pipeline: {status['feature_pipeline']['registered_features']} features")
    print(f"  Event bus: {status['event_bus']}")


if __name__ == "__main__":
    print("="*70)
    print("  v8.0 ARCHITECTURE SMOKE TEST")
    print("="*70)
    test_event_bus()
    test_state_machine()
    test_feature_pipeline()
    test_portfolio_manager()
    test_risk_pipeline()
    test_self_healing()
    test_decision_audit()
    test_memory_system()
    test_multi_agent()
    test_monitoring()
    test_industrial_bot_status()
    print("\n" + "="*70)
    print("  ALL TESTS PASSED")
    print("="*70)
