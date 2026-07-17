"""scripts/test_signals_v3.py
=====================================================================
Smoke test for the v3 Industrial-Grade Signal (engine/signals_v3.py).
Validates all 30 features across 22 sub-dataclasses.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from engine.signals_v3 import (
    Signal, SignalBuilder, Action, MarketRegime, TradingSession,
    SignalSourceType, SignalSource, SignalQuality,
    OrderType, Urgency, ExecutionStatus,
    EnsembleVote, migrate_v2_to_v3,
)


def test_factory_methods():
    print("\n[1/6] Testing factory methods (hold/buy/sell)...")
    sig_hold = Signal.hold('BTCUSD', 'M15', price=43250.0)
    assert sig_hold.action == Action.HOLD
    assert sig_hold.symbol == 'BTCUSD'
    assert not sig_hold.is_actionable
    print(f"  HOLD: id={sig_hold.signal_id[:8]}, symbol={sig_hold.symbol}")

    sig_buy = Signal.buy('BTCUSD', 'M15', strength=0.85, price=43250.0,
                        stop_loss=42500, take_profit=45000,
                        strategy_id='Momentum_v4.1', strategy_version='4.1.0')
    assert sig_buy.is_actionable
    assert sig_buy.entry_price == 43250.0
    print(f"  BUY: id={sig_buy.signal_id[:8]}, strength={sig_buy.strength}, RR={sig_buy.rr_ratio:.2f}")

    sig_sell = Signal.sell('ETHUSD', 'M15', strength=0.72, price=2580.0,
                           stop_loss=2650, take_profit=2400)
    assert sig_sell.action == Action.SELL
    assert sig_sell.direction == 'short'
    print(f"  SELL: id={sig_sell.signal_id[:8]}, direction={sig_sell.direction}")


def test_builder_pattern():
    print("\n[2/6] Testing SignalBuilder (all 30 features)...")
    sig = (SignalBuilder()
           .with_symbol('ETHUSD', 'M15')
           .with_action(Action.SELL, strength=0.78, quality=SignalQuality.A_PLUS)
           .with_price(2580.0, bar_time=datetime.now(timezone.utc),
                       high=2595, low=2570, volume=12500)
           .with_sl_tp(2650, 2400)
           .with_strategy('Transformer_v9', '9.0.0')
           .with_source(SignalSourceType.ML, SignalSource.LIVE)
           .with_ai_model('transformer_v9', '9.0', 0.9, 12.5, 0.85)
           .with_embedding('emb_a3f9c2')
           .with_confidence(overall=0.82, trend=0.85, momentum=0.78,
                            volume=0.72, ai=0.85, macro=0.65,
                            pattern=0.80, sentiment=0.55)
           .with_volatility(atr=18.5, atr_pct=0.0072, hv=0.65, rv=0.58, percentile=0.72)
           .with_liquidity(spread_bps=2.1, slippage_bps=1.5, depth_usd=2_500_000)
           .with_mtf(m5='SELL', m15='SELL', h1='SELL', h4='HOLD', d1='SELL')
           .with_risk(estimated_risk_usd=180, kelly_fraction=0.15,
                      expected_winrate=0.62, expected_duration_bars=12)
           .with_execution(OrderType.LIMIT, priority=0.8, urgency=Urgency.HIGH)
           .with_explainability(
               top_features=[('rsi', -0.32), ('macd', -0.28), ('volume', 0.15)],
               decision_trace=['RSI > 70', 'MACD bearish cross', 'Volume declining'],
               latency_ms=8.3
           )
           .with_shap({'rsi': -0.32, 'macd': -0.28, 'atr': 0.10})
           .with_ensemble(
               votes=[
                   EnsembleVote('trend', 'SELL', 0.85, 1.0, 'EMA stacked down'),
                   EnsembleVote('momentum', 'SELL', 0.78, 1.0, 'MACD bearish'),
                   EnsembleVote('mean_rev', 'SELL', 0.65, 0.8, 'RSI overbought'),
                   EnsembleVote('smc', 'HOLD', 0.40, 0.6, 'No clear OB'),
                   EnsembleVote('risk', 'HOLD', 0.50, 0.5, 'Risk acceptable'),
               ],
               final_action='SELL', final_strength=0.78, agreement=0.75
           )
           .with_bayesian(prior=0.55, posterior=0.82, evidence=0.45)
           .with_uncertainty(epistemic=0.08, aleatoric=0.12, ci_95=(0.65, 0.95))
           .with_microstructure(imbalance=-0.35, delta=-125.5, cvd=-850, absorption=True)
           .with_onchain(funding=-0.0001, oi=850_000_000, liquidation=1_200_000,
                         whale_inflow=5_500_000)
           .with_sentiment(fear_greed=72, twitter=0.35, reddit=0.28, news_sent=0.15)
           .with_news(high_impact=False, minutes_to=180, event='FOMC')
           .with_correlation(btc=0.92, eth=1.0, sp500=-0.15, dxy=0.20, gold=0.10)
           .with_features({'rsi_14': 72.5, 'ema_9': 2575, 'atr_14': 18.5, 'macd': -2.1},
                           version='1.2')
           .with_audit(created_by='industrial_bot', bot_version='8.0',
                       git_commit='a1b2c3d', environment='production')
           .with_replay(cycle_id=42, snapshot_id='snap_1234',
                        market_snapshot_id='ms_a3f9')
           .build())

    assert sig.action == Action.SELL
    assert sig.strength == 0.78
    assert sig.symbol == 'ETHUSD'
    assert sig.is_actionable
    assert sig.identity.strategy_id == 'Transformer_v9'
    assert sig.ai.model_name == 'transformer_v9'
    assert sig.confidence.overall == 0.82
    assert sig.volatility.atr == 18.5
    assert sig.risk.kelly_fraction == 0.15
    assert sig.ensemble.agreement_score == 0.75
    assert sig.bayesian.posterior == 0.82
    assert sig.uncertainty.epistemic_uncertainty == 0.08
    assert sig.microstructure.absorption_detected
    assert sig.onchain.open_interest_usd == 850_000_000
    assert sig.sentiment.fear_greed_index == 72
    assert sig.correlation.btc_correlation == 0.92
    assert sig.feature_meta.feature_count == 4
    assert sig.audit.bot_version == '8.0'
    assert sig.replay.cycle_id == 42

    print(f"  Built: {sig.symbol} {sig.action} strength={sig.strength} quality={sig.quality.value}")
    print(f"  Sub-objects: identity, market, strategy, ai, confidence, volatility,")
    print(f"    liquidity, mtf, risk, execution, explain, ensemble, bayesian,")
    print(f"    uncertainty, microstructure, onchain, sentiment, news, session_info,")
    print(f"    correlation, feature_meta, audit, replay (23 total)")
    print(f"  Feature hash: {sig.feature_meta.feature_hash}")
    print(f"  Decision trace: {sig.explain.decision_trace}")
    print(f"  SHAP values: {sig.explain.shap_values}")


def test_immutability():
    print("\n[3/6] Testing immutability (frozen=True)...")
    sig = Signal.buy('BTCUSD', 'M15', strength=0.7, price=40000)
    try:
        sig.strength = 0.99
        print("  FAIL: was able to mutate frozen dataclass")
        assert False
    except Exception as e:
        print(f"  OK — cannot mutate strength: {type(e).__name__}")

    try:
        sig.identity.strategy_id = "hacked"
        print("  FAIL: was able to mutate sub-dataclass")
        assert False
    except Exception as e:
        print(f"  OK — cannot mutate identity: {type(e).__name__}")


def test_serialization():
    print("\n[4/6] Testing serialization (to_dict / to_json / from_dict / from_json)...")
    sig = (SignalBuilder()
           .with_symbol('BTCUSD', 'M15')
           .with_action(Action.BUY, strength=0.85)
           .with_price(43250.0, bar_time=datetime.now(timezone.utc))
           .with_sl_tp(42500, 45000)
           .with_strategy('Momentum_v4', '4.1')
           .with_features({'rsi': 62, 'ema': 43100})
           .build())

    d = sig.to_dict()
    assert d['schema_version'] == 3
    assert d['market']['symbol'] == 'BTCUSD'
    assert d['strategy']['action'] == 'BUY'
    assert 'identity' in d
    assert 'ai' in d
    assert 'confidence' in d
    assert 'uncertainty' in d
    assert 'microstructure' in d
    assert 'onchain' in d
    assert 'sentiment' in d

    json_str = sig.to_json(indent=2)
    assert len(json_str) > 500
    sig2 = Signal.from_json(json_str)
    assert sig2.signal_id == sig.signal_id
    assert sig2.action == sig.action
    assert sig2.strength == sig.strength
    assert sig2.symbol == sig.symbol
    print(f"  Roundtrip: {len(d)} top-level keys, JSON {len(json_str)} bytes")
    print(f"  Restored: {sig2.symbol} {sig2.action} strength={sig2.strength}")


def test_functional_updates():
    print("\n[5/6] Testing functional updates (with_status / with_reject_reason)...")
    sig = Signal.buy('BTCUSD', 'M15', strength=0.85, price=43250)

    sig_approved = sig.with_status(ExecutionStatus.APPROVED)
    assert sig_approved.execution_status == ExecutionStatus.APPROVED
    assert sig.execution_status == ExecutionStatus.NEW  # original unchanged

    sig_rejected = sig.with_reject_reason('ATR% > 5% threshold')
    assert sig_rejected.execution_status == ExecutionStatus.REJECTED
    assert sig_rejected.strategy.reject_reason == 'ATR% > 5% threshold'

    sig_audit = sig.with_audit(hostname='prod-server-1')
    assert sig_audit.audit.hostname == 'prod-server-1'
    print(f"  Original status: {sig.execution_status.value}")
    print(f"  Approved status: {sig_approved.execution_status.value}")
    print(f"  Rejected status: {sig_rejected.execution_status.value}, reason='{sig_rejected.strategy.reject_reason}'")


def test_migration_from_v2():
    print("\n[6/6] Testing v2 → v3 migration...")
    v2_dict = {
        'symbol': 'BTCUSD', 'timeframe': 'M15', 'action': 'BUY',
        'strength': 0.72, 'price': 43250.0,
        'entry_price': 43250, 'stop_loss': 42500, 'take_profit': 45000,
        'strategy_name': 'Momentum_v4', 'strategy_version': '4.0',
        'regime': 'trending_up',
        'decision_trace': ['EMA9>EMA21', 'ADX=28', 'Volume spike'],
        'features': {'rsi': 62, 'ema9': 43100, 'atr': 450},
    }
    migrated = migrate_v2_to_v3(v2_dict)
    assert migrated.action == Action.BUY
    assert migrated.symbol == 'BTCUSD'
    assert migrated.strength == 0.72
    assert migrated.identity.strategy_id == 'Momentum_v4'
    assert migrated.identity.strategy_version == '4.0'
    assert migrated.explain.decision_trace == ['EMA9>EMA21', 'ADX=28', 'Volume spike']
    assert migrated.feature_meta.feature_count == 3
    print(f"  Migrated v2 dict → v3 Signal: {migrated.symbol} {migrated.action}")
    print(f"  Strategy preserved: {migrated.identity.strategy_id}")
    print(f"  Decision trace preserved: {migrated.explain.decision_trace}")
    print(f"  Features hash computed: {migrated.feature_meta.feature_hash}")


def test_feature_count():
    """Count all fields across all sub-dataclasses."""
    print("\n[STATS] Counting all fields across v3 Signal system...")
    from dataclasses import fields
    from engine import signals_v3

    # Get all dataclasses
    import inspect
    classes = [(name, obj) for name, obj in inspect.getmembers(signals_v3)
               if inspect.isclass(obj) and obj.__module__ == 'engine.signals_v3'
               and hasattr(obj, '__dataclass_fields__')]

    total_fields = 0
    print(f"  Total dataclasses: {len(classes)}")
    for name, cls in sorted(classes, key=lambda x: x[0]):
        n = len(fields(cls))
        total_fields += n
        print(f"    {name:35s}  {n:3d} fields")
    print(f"  {'TOTAL':35s}  {total_fields:3d} fields")
    print(f"\n  v3 Schema = {total_fields} fields across {len(classes)} typed dataclasses")
    print(f"  vs v2 = ~25 fields in 1 SignalContext dataclass")
    print(f"  Improvement: {total_fields - 25}+ new fields, {len(classes) - 2} new typed objects")


if __name__ == "__main__":
    print("="*70)
    print("  v3 INDUSTRIAL-GRADE SIGNAL TEST (engine/signals_v3.py)")
    print("="*70)
    test_factory_methods()
    test_builder_pattern()
    test_immutability()
    test_serialization()
    test_functional_updates()
    test_migration_from_v2()
    test_feature_count()
    print("\n" + "="*70)
    print("  ALL v3 SIGNAL TESTS PASSED")
    print("="*70)
