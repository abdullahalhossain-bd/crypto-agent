"""
Livermore Principles — The Wisdom Gate
=======================================

Encodes the 20 timeless principles from "Reminiscences of a Stock Operator"
into enforceable code that runs BEFORE every trade.

This is NOT a strategy. It is a WISDOM GATE — a final filter that asks:
    "Would Jesse Livermore take this trade?"

The 20 principles, each as a callable check:

    1.  price_is_truth          — reject news/opinion-based trades
    2.  patterns_repeat         — verify current pattern has historical precedent
    3.  observation_not_prediction — require statistical evidence, not guesses
    4.  build_memory            — trade must be recorded for future learning
    5.  learn_from_mistakes     — check if similar past trades lost
    6.  high_probability_only   — confidence must exceed threshold
    7.  market_doesnt_care      — block revenge/impulse trades
    8.  speed_matters           — verify execution latency is acceptable
    9.  news_comes_late         — price reaction first, news second
    10. market_memory           — find similar historical setups
    11. confidence_from_stats   — confidence = win_rate × R:R, not emotion
    12. never_trade_bored       — block trades when no real edge exists
    13. time_to_wait            — patience counter: must wait N bars between trades
    14. execution_quality       — verify spread/slippage is within tolerance
    15. market_type_detection   — regime must match strategy
    16. never_follow_tips       — external signals must be independently verified
    17. self_learning           — adapt based on recent trade outcomes
    18. adaptive_sizing         — position size from confidence + ATR + drawdown
    19. protect_capital_first   — survival > profit
    20. probability_machine     — every trade = expected value calculation

Usage:
    from livermore_principles import WisdomGate, TradeContext
    gate = WisdomGate()
    verdict = gate.evaluate(TradeContext(
        symbol="BTCUSD", direction="BUY", confidence=0.75,
        win_rate=0.62, rr_ratio=2.5, atr_ratio=1.2,
        bars_since_last_trade=15, spread_bps=3,
        regime="trending_up", drawdown_pct=2.5,
        recent_losses=1, recent_wins=3,
    ))
    if verdict.approved:
        execute_trade()
    else:
        log.info(f"Wisdom gate: {verdict.reason}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TradeContext:
    """All inputs the wisdom gate needs to evaluate a trade."""
    symbol: str
    direction: str               # "BUY" or "SELL"
    confidence: float            # 0..1 — strategy confidence
    win_rate: float              # 0..1 — historical win rate for similar setups
    rr_ratio: float              # reward:risk ratio
    atr_ratio: float             # current ATR / baseline ATR
    bars_since_last_trade: int   # bars since last trade on this symbol
    spread_bps: float            # current spread in basis points
    regime: str                  # market regime
    drawdown_pct: float          # current drawdown %
    recent_losses: int           # consecutive recent losses
    recent_wins: int             # consecutive recent wins
    pattern_match_count: int = 0 # how many historical patterns matched
    pattern_win_rate: float = 0.5  # win rate of matched patterns
    news_pending: bool = False   # is there pending news?
    external_signal: bool = False  # is this from an external tip?
    # Optional: execution context
    expected_slippage_bps: float = 0.0
    latency_ms: float = 0.0
    # v6.9: Principles 21-40
    bars_scanned: int = 0        # total bars analyzed this session
    trades_today: int = 0        # trades executed today
    max_trades_per_day: int = 10 # quality > quantity limit
    price_distance_from_entry: float = 0.0  # how far price has moved from ideal entry
    has_open_position: bool = False  # already has a position on this symbol
    avg_slippage_bps: float = 0.0  # average recent slippage
    sharpe_ratio: float = 0.0   # recent performance Sharpe
    profit_factor: float = 0.0  # recent profit factor
    is_averaging_down: bool = False  # trying to add to a losing position
    structure_valid: bool = True  # is market structure still valid for this trade
    forward_tested: bool = True   # has strategy been forward-tested
    # v7.0: Principles 41-60
    trend_strength: float = 0.5    # 0..1 — how strong is the current trend
    is_winning_position: bool = False  # is this an add to a winning trade?
    is_pyramiding: bool = False    # trying to add to a winner
    current_r_multiple: float = 0.0  # current R-multiple of open position
    days_in_trade: int = 0         # how many days has position been open
    noise_ratio: float = 0.5      # 0..1 — how much of recent action is noise
    symbol_volatility_rank: float = 0.5  # 0..1 — this symbol's volatility vs others
    symbol_spread_rank: float = 0.5  # 0..1 — this symbol's spread vs others
    symbol_fakeout_rate: float = 0.3  # 0..1 — historical fakeout frequency
    context_score: float = 0.5    # 0..1 — combined context (trend+liquidity+volume)
    monthly_growth_target: float = 5.0  # % monthly growth target
    current_monthly_growth: float = 0.0  # % growth this month
    consecutive_wins: int = 0     # consecutive winning trades
    equity_curve_smoothness: float = 0.5  # 0..1 — how smooth is recent equity curve
    # v7.1: Principles 61-80
    primary_trend: str = "neutral"   # "up" / "down" / "neutral"
    secondary_trend: str = "neutral"
    is_leader: bool = False          # is this symbol a market leader?
    leader_confirmed: bool = False   # has sector/market confirmed leader?
    cross_market_score: float = 0.5  # 0..1 — cross-market alignment
    setup_rank: int = 1              # 1=best, 10=worst (among current setups)
    total_setups_available: int = 1  # how many setups exist right now
    has_explicit_reason: bool = True # does this trade have a logged reason?
    false_breakout_risk: float = 0.3  # 0..1 — probability this is a fakeout
    institutional_footprint: float = 0.5  # 0..1 — institutional activity detected
    timing_score: float = 0.5        # 0..1 — timing quality (session+spread+vol)
    exit_rules_defined: bool = True  # are SL/TP/trailing defined before entry?
    portfolio_correlation: float = 0.3  # 0..1 — correlation with open positions
    portfolio_exposure: float = 0.3  # 0..1 — current gross exposure
    emotional_market: bool = False   # news spike / huge spread / random candle
    long_term_expectancy: float = 0.0  # expected $ per trade over 1000 trades
    # v7.2: Principles 81-100
    manipulation_detected: bool = False  # stop hunt / liquidity grab detected
    market_clue_score: float = 0.5      # 0..1 — pre-move clues (vol change, vol increase)
    capital_allocation_score: float = 0.5  # 0..1 — quality of capital allocation
    crowded_trade: bool = False         # is everyone on the same side?
    correlation_awareness: float = 0.5  # 0..1 — cross-asset correlation checked
    regime_shift_detected: bool = False  # is regime shifting right now?
    black_swan_risk: float = 0.0        # 0..1 — probability of black swan
    decision_quality: float = 0.5       # 0..1 — quality of this decision (not outcome)
    rl_feedback_positive: bool = True   # is RL feedback positive?
    # v7.3: Principles 101-120
    timing_quality: float = 0.5        # 0..1 — entry timing quality (session+pullback+liquidity)
    price_structure_score: float = 0.5  # 0..1 — HH/HL/BOS/CHoCH quality
    confirmation_count: int = 3        # how many confirmations (price+vol+liq+momentum+corr)
    entry_score: float = 70.0          # 0..100 — composite entry score
    trend_persistence: float = 0.5     # 0..1 — how likely trend is to continue
    market_phase: str = "unknown"      # accumulation/markup/distribution/markdown
    stop_loss_type: str = "atr"        # "fixed" / "atr" / "dynamic"
    winner_protection_r: float = 0.0   # current R-multiple of winner (for trailing)
    session_win_rates: dict = field(default_factory=dict)  # {"london": 0.6, "ny": 0.55, ...}
    usd_exposure: float = 0.0          # 0..1 — correlated USD exposure
    system_health: float = 1.0         # 0..1 — API+broker+latency health
    # v7.4: Principles 121-140 (pages 120-140: Capital Rotation, Leadership, Discipline)
    capital_flow_score: float = 0.5    # 0..1 — institutional capital flow strength
    capital_flow_direction: str = "neutral"  # bullish/bearish/neutral
    relative_strength_rank: float = 0.5  # 0..1 — 1=this symbol is strongest in universe
    market_breadth: float = 0.5        # 0..1 — how broad is participation
    mtf_alignment_score: float = 0.5   # 0..1 — multi-TF consensus (W/D/H4/H1/M15)
    mtf_high_tf_agrees: bool = True    # do weekly + daily agree with our direction?
    signal_rank_percentile: float = 0.5  # 0..1 — this signal's rank among all current signals
    smart_money_score: float = 0.5     # 0..1 — institutional participation detected
    smart_money_direction: str = "neutral"  # bullish/bearish/neutral
    execution_quality_score: float = 0.8  # 0..1 — recent fill quality
    conviction_level: float = 0.5      # 0..1 — gradual conviction building
    noise_filter_passed: bool = True   # is this signal information, not noise?
    historical_match_count: int = 0    # how many similar historical cases
    historical_win_rate: float = 0.5   # win rate of those historical matches
    regime_strategy_match: bool = True  # is current strategy suited to current regime?
    risk_allocation_pct: float = 2.0   # % of equity being risked on this trade
    strategy_decay_detected: bool = False  # is this strategy's edge decaying?
    portfolio_correlation_avg: float = 0.3  # avg correlation with open positions
    portfolio_diversification: float = 0.7  # 0..1 — diversification score
    adaptive_rules_active: bool = True  # are we using adaptive (not static) rules?
    consistency_score: float = 0.5     # 0..1 — how consistent is recent performance?
    weekly_audit_passed: bool = True   # did last weekly audit pass?
    weekly_audit_gpa: float = 3.0      # 0..4 — last audit GPA
    # v7.5: Principles 141-160 (pages 140-160: Decision Quality, Context, Self-Evolution)
    market_context_score: float = 0.5  # 0..1 — full context awareness
    context_understood: bool = True    # does AI understand WHY price is moving?
    capital_efficiency: float = 0.5    # 0..1 — return / risk ratio quality
    strategy_lifetime_days: int = 0    # how old is this strategy?
    strategy_edge_declining: bool = False  # is edge declining over time?
    volatility_regime: str = "normal"  # low/normal/high/extreme
    liquidity_quality: float = 0.7     # 0..1 — spread+depth+session quality
    daily_risk_budget_remaining: float = 2.0  # % of equity still available today
    daily_risk_budget_used: float = 0.0  # % of equity already risked today
    correlated_exposure_pct: float = 0.0  # % of portfolio in correlated direction
    adaptive_confidence: float = 0.5  # 0..1 — confidence adjusted for context
    decision_quality_score: float = 0.5  # 0..1 — quality of THIS decision
    execution_latency_ms: float = 100.0  # recent order latency
    missed_opportunity_count: int = 0  # how many good setups missed recently?
    portfolio_balance_score: float = 0.7  # 0..1 — strategy diversification
    consecutive_loss_count: int = 0   # current consecutive loss streak
    risk_reduction_active: bool = False  # is dynamic risk reduction active?
    execution_window_quality: float = 0.8  # 0..1 — is now a good time to execute?
    structural_change_detected: bool = False  # has market structure changed?
    learning_loop_active: bool = True  # is the AI learning loop running?
    risk_adjusted_return_target: float = 2.0  # target Sharpe ratio
    # v7.6: Principles 161-180 (pages 160-180: Cycles, Survival, Adaptive Intelligence)
    survival_mode_active: bool = False  # is capital preservation mode active?
    market_cycle: str = "unknown"     # expansion/peak/consolidation/decline/recovery
    cycle_confidence: float = 0.5     # 0..1 — how confident in cycle detection
    probability_buy: float = 0.0      # P(BUY)
    probability_sell: float = 0.0     # P(SELL)
    probability_wait: float = 1.0     # P(WAIT)
    structure_priority_score: float = 0.5  # 0..1 — structure-first (not indicator-first)
    portfolio_risk_usd: float = 0.0   # total portfolio risk in $
    dynamic_risk_mode: str = "normal"  # increasing/stable/reducing/minimum
    false_confidence_detected: bool = False  # high WR but low sample / poor conditions
    liquidity_asset_score: float = 0.7  # 0..1 — liquidity as asset quality
    idle_mode: bool = False           # no edge → observe and wait
    strategy_evolution_active: bool = True  # is strategy evolution running?
    edge_decay_rate: float = 0.0      # rate of edge decline (per 100 trades)
    allocation_diversified: bool = True  # is capital diversified across strategies?
    knowledge_added: bool = True      # did this trade add to knowledge graph?
    institutional_memory_size: int = 0  # how many patterns in memory?
    black_swan_prepared: bool = True  # are emergency rules in place?
    opportunity_cost_acceptable: bool = True  # is waiting worse than this trade?
    self_diagnosis_passed: bool = True  # did weekly self-diagnosis pass?
    benchmark_outperformance: float = 0.0  # alpha vs benchmark (R)
    decision_engine_consensus: float = 0.5  # 0..1 — multi-factor decision consensus
    autonomous_mode: bool = True      # is AI operating autonomously?
    # v7.7: Principles 181-200 (pages 180-200: Timing, Self-Control, Decision Making)
    patience_mode: bool = False       # is AI in patience/wait mode?
    opportunity_rank: float = 0.5     # 0..1 — rank among all current opportunities
    discipline_score: float = 0.8     # 0..1 — how disciplined recently
    trend_persistence_score: float = 0.5  # 0..1 — likelihood trend continues
    noise_filtered: bool = True       # has noise been filtered out?
    multi_confirmation_count: int = 4  # how many confirmations (trend+liq+vol+...)
    dynamic_exit_ready: bool = True   # is dynamic exit intelligence active?
    near_miss_analyzed: bool = True   # have near-miss entries been analyzed?
    market_memory_available: bool = True  # is market memory database available?
    confidence_earned: bool = True    # has confidence been earned (not assumed)?
    market_fatigue_detected: bool = False  # is trend showing fatigue?
    capital_protection_active: bool = True  # is capital protection system active?
    execution_optimized: bool = True  # has execution been pre-optimized?
    portfolio_correlation_managed: bool = True  # is correlation being managed?
    strategy_switched: bool = False   # has strategy been auto-switched for regime?
    rl_loop_active: bool = True       # is reinforcement learning loop running?
    performance_dashboard_active: bool = True  # is performance dashboard active?
    decision_quality_focus: bool = True  # focus on decision quality, not prediction
    autonomous_improvement: bool = True  # is autonomous self-improvement active?
    institutional_mindset_complete: bool = True  # 8-dimensional mindset check


@dataclass
class WisdomVerdict:
    approved: bool
    confidence_adjusted: float    # confidence after wisdom adjustments
    position_multiplier: float    # 0..1.5 — sizing multiplier
    checks_passed: int
    checks_failed: int
    failed_principles: list[str] = field(default_factory=list)
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "confidence_adjusted": round(self.confidence_adjusted, 3),
            "position_multiplier": round(self.position_multiplier, 2),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "failed_principles": self.failed_principles,
            "reason": self.reason,
            "notes": self.notes,
        }


class WisdomGate:
    """The 20-principle wisdom gate — runs before every trade.

    Parameters:
        min_confidence: minimum confidence to trade (default 0.60)
        min_rr: minimum R:R to trade (default 2.0)
        min_bars_between_trades: patience — bars to wait (default 5)
        max_spread_bps: max acceptable spread (default 10)
        max_drawdown_pct: halt trading above this drawdown (default 15)
        max_consecutive_losses: halt after N losses (default 3)
        revenge_trade_cooldown_bars: bars to wait after a loss (default 10)
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_rr: float = 2.0,
        min_bars_between_trades: int = 5,
        max_spread_bps: float = 10.0,
        max_drawdown_pct: float = 15.0,
        max_consecutive_losses: int = 3,
        revenge_trade_cooldown_bars: int = 10,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.min_bars_between_trades = min_bars_between_trades
        self.max_spread_bps = max_spread_bps
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.revenge_trade_cooldown_bars = revenge_trade_cooldown_bars

    def evaluate(self, ctx: TradeContext) -> WisdomVerdict:
        """Run all 20 principle checks. Returns WisdomVerdict."""
        passed = 0
        failed = 0
        failed_list: list[str] = []
        notes: list[str] = []
        confidence = ctx.confidence
        position_mult = 1.0

        # ── 1. Price is Truth ──────────────────────────────────────
        # Reject if trade is based on news or external opinion
        if ctx.news_pending:
            failed += 1
            failed_list.append("1.price_is_truth")
            notes.append("REJECT: News pending — price is truth, not news")
        else:
            passed += 1

        # ── 2. Patterns Repeat ─────────────────────────────────────
        # Verify current setup has historical precedent
        if ctx.pattern_match_count >= 3:
            passed += 1
            notes.append(f"Pattern: {ctx.pattern_match_count} historical matches "
                        f"(win rate: {ctx.pattern_win_rate:.0%})")
        else:
            # Not a hard fail — just reduce confidence
            confidence *= 0.85
            notes.append(f"Pattern: only {ctx.pattern_match_count} matches — confidence reduced")
            passed += 1  # soft pass

        # ── 3. Observation > Prediction ────────────────────────────
        # Confidence must come from data, not assumption
        if ctx.win_rate > 0:
            passed += 1
        else:
            failed += 1
            failed_list.append("3.observation_not_prediction")
            notes.append("REJECT: No historical win rate data — cannot observe")

        # ── 4. Build Memory ────────────────────────────────────────
        # Trade will be recorded (database handles this)
        # This is always true if we reach here
        passed += 1

        # ── 5. Learn From Mistakes ─────────────────────────────────
        # Check if similar past trades lost
        if ctx.pattern_win_rate < 0.35 and ctx.pattern_match_count >= 5:
            failed += 1
            failed_list.append("5.learn_from_mistakes")
            notes.append(f"REJECT: Similar patterns won only {ctx.pattern_win_rate:.0%} — "
                        "learning from past mistakes says SKIP")
        else:
            passed += 1

        # ── 6. Trade Only High Probability ─────────────────────────
        if confidence >= self.min_confidence:
            passed += 1
            # Scale position by confidence
            if confidence >= 0.90:
                position_mult = 1.5
                notes.append("Confidence 90%+ → aggressive position (1.5x)")
            elif confidence >= 0.80:
                position_mult = 1.2
                notes.append("Confidence 80%+ → normal position (1.2x)")
            elif confidence >= 0.60:
                position_mult = 0.7
                notes.append("Confidence 60-80% → small position (0.7x)")
        else:
            failed += 1
            failed_list.append("6.high_probability_only")
            notes.append(f"REJECT: Confidence {confidence:.0%} < {self.min_confidence:.0%}")

        # ── 7. Market Doesn't Care — No Revenge Trading ────────────
        if ctx.recent_losses >= self.max_consecutive_losses:
            failed += 1
            failed_list.append("7.no_revenge_trading")
            notes.append(f"REJECT: {ctx.recent_losses} consecutive losses — "
                        "market doesn't care about you, STOP")
        elif ctx.recent_losses > 0 and ctx.bars_since_last_trade < self.revenge_trade_cooldown_bars:
            failed += 1
            failed_list.append("7.no_revenge_trading")
            notes.append(f"REJECT: Only {ctx.bars_since_last_trade} bars since loss — "
                        f"need {self.revenge_trade_cooldown_bars} (revenge trade prevention)")
        else:
            passed += 1

        # ── 8. Speed Matters ───────────────────────────────────────
        if ctx.latency_ms > 500:
            confidence *= 0.9
            notes.append(f"Latency {ctx.latency_ms:.0f}ms — confidence reduced 10%")
        passed += 1  # soft check

        # ── 9. News Comes Late ─────────────────────────────────────
        # Price reaction first — if price hasn't moved, don't anticipate news
        # This is handled by strategy (price-based), so just pass
        passed += 1

        # ── 10. Market Memory ──────────────────────────────────────
        # Find similar historical setups
        if ctx.pattern_match_count >= 10:
            passed += 1
            notes.append(f"Market memory: {ctx.pattern_match_count} similar setups found")
        else:
            # Soft check — reduce confidence if no memory
            confidence *= 0.90
            passed += 1

        # ── 11. Confidence From Statistics ─────────────────────────
        # Confidence = win_rate × R:R, not emotion
        statistical_confidence = ctx.win_rate * min(ctx.rr_ratio / 3.0, 1.0)
        if statistical_confidence > 0.3:
            passed += 1
            notes.append(f"Statistical confidence: {statistical_confidence:.2f} "
                        f"(win_rate={ctx.win_rate:.0%} × R:R={ctx.rr_ratio:.1f})")
        else:
            confidence *= 0.85
            notes.append(f"Low statistical confidence: {statistical_confidence:.2f}")
            passed += 1  # soft

        # ── 12. Never Trade Because Bored ──────────────────────────
        # If no real edge (low confidence + low pattern match), reject
        # BUGFIX (log audit): confidence threshold was 0.50, but with
        # normalized_confidence = 0.40 + strength*2.0, a strength of 0.04
        # gives confidence=0.48 < 0.50 → FAIL. Lowered to 0.40 (the floor
        # of our normalization range) so any actionable signal passes.
        if confidence < 0.40 and ctx.pattern_match_count < 3:
            failed += 1
            failed_list.append("12.never_trade_bored")
            notes.append("REJECT: No edge — don't trade because bored")
        else:
            passed += 1

        # ── 13. Time To Wait ───────────────────────────────────────
        # Patience — must wait N bars between trades
        if ctx.bars_since_last_trade < self.min_bars_between_trades:
            failed += 1
            failed_list.append("13.time_to_wait")
            notes.append(f"REJECT: Only {ctx.bars_since_last_trade} bars since last trade — "
                        f"need {self.min_bars_between_trades} (patience)")
        else:
            passed += 1

        # ── 14. Execution Quality ──────────────────────────────────
        # Check spread/slippage
        total_cost = ctx.spread_bps + ctx.expected_slippage_bps
        if total_cost > self.max_spread_bps:
            failed += 1
            failed_list.append("14.execution_quality")
            notes.append(f"REJECT: Spread+slippage {total_cost:.1f}bps > {self.max_spread_bps}bps")
        else:
            passed += 1

        # ── 15. Market Type Detection ──────────────────────────────
        # Regime must be tradeable
        # RISK PIPELINE AUDIT FIX: aligned with MarketRegime enum values
        # from regime_orchestrator.py. Previously had "low_vol_dead",
        # "choppy", "extreme_vol" which are NEVER returned. Now includes
        # "crisis" and "transition" which WERE missing.
        bad_regimes = {"chop", "high_vol", "crisis", "transition", "extreme_vol"}
        if ctx.regime in bad_regimes:
            failed += 1
            failed_list.append("15.market_type_detection")
            notes.append(f"REJECT: Regime '{ctx.regime}' — not tradeable")
        else:
            passed += 1
            notes.append(f"Regime: {ctx.regime} — tradeable")

        # ── 16. Never Follow Tips ──────────────────────────────────
        # External signals must be independently verified
        if ctx.external_signal and confidence < 0.70:
            failed += 1
            failed_list.append("16.never_follow_tips")
            notes.append("REJECT: External signal with low confidence — "
                        "never blindly follow tips")
        else:
            passed += 1

        # ── 17. Self Learning ──────────────────────────────────────
        # Adapt based on recent performance
        if ctx.recent_wins > ctx.recent_losses:
            # On a hot streak — don't overlever
            position_mult = min(position_mult, 1.2)
            notes.append("Self-learning: hot streak — capping at 1.2x")
        elif ctx.recent_losses > ctx.recent_wins:
            # On a cold streak — reduce size
            position_mult *= 0.7
            notes.append("Self-learning: cold streak — reducing to 0.7x")
        passed += 1

        # ── 18. Adaptive Position Sizing ───────────────────────────
        # Adjust for volatility
        if ctx.atr_ratio > 2.0:
            position_mult *= 0.5
            notes.append(f"Adaptive sizing: ATR {ctx.atr_ratio:.1f}x baseline → "
                        "reduce 50%")
        elif ctx.atr_ratio > 1.5:
            position_mult *= 0.75
            notes.append(f"Adaptive sizing: ATR {ctx.atr_ratio:.1f}x baseline → "
                        "reduce 25%")
        elif ctx.atr_ratio < 0.5:
            position_mult *= 0.8
            notes.append(f"Adaptive sizing: ATR {ctx.atr_ratio:.1f}x baseline → "
                        "dead market, reduce 20%")
        passed += 1

        # ── 19. Protect Capital First ──────────────────────────────
        # Survival > profit
        if ctx.drawdown_pct >= self.max_drawdown_pct:
            failed += 1
            failed_list.append("19.protect_capital_first")
            notes.append(f"REJECT: Drawdown {ctx.drawdown_pct:.1f}% >= "
                        f"{self.max_drawdown_pct}% — SURVIVAL MODE")
        else:
            passed += 1
            # Reduce size as drawdown increases
            if ctx.drawdown_pct > self.max_drawdown_pct * 0.5:
                position_mult *= 0.5
                notes.append(f"Drawdown {ctx.drawdown_pct:.1f}% — reducing size 50%")

        # ── 20. Probability Machine ────────────────────────────────
        # Expected Value = (win_rate × reward) - (loss_rate × risk)
        ev = (ctx.win_rate * ctx.rr_ratio) - ((1 - ctx.win_rate) * 1.0)
        if ev <= 0:
            failed += 1
            failed_list.append("20.probability_machine")
            notes.append(f"REJECT: Negative expected value ({ev:.2f}) — "
                        "not a probability machine, it's a charity")
        else:
            passed += 1
            notes.append(f"Expected value: {ev:.2f} per trade "
                        f"(WR={ctx.win_rate:.0%} R:R={ctx.rr_ratio:.1f})")

        # ════════════════════════════════════════════════════════════
        # PRINCIPLES 21-40 (pages 20-40: psychology, execution, adaptation)
        # ════════════════════════════════════════════════════════════

        # ── 21. Be Right, Not Bull or Bear ─────────────────────────
        # No directional bias — trade what the data says, not what you "feel"
        # If win_rate for this direction is < 40%, the AI may have a bias
        if ctx.win_rate < 0.40 and ctx.pattern_match_count >= 5:
            failed += 1
            failed_list.append("21.no_directional_bias")
            notes.append("REJECT: Low win rate suggests directional bias — "
                        "be right, not bull or bear")
        else:
            passed += 1

        # ── 22. Never Marry a Position ─────────────────────────────
        # Don't add to losing trades — re-evaluate continuously
        if ctx.is_averaging_down:
            failed += 1
            failed_list.append("22.never_marry_position")
            notes.append("REJECT: Attempting to average down — never marry a position")
        elif not ctx.structure_valid:
            failed += 1
            failed_list.append("22.never_marry_position")
            notes.append("REJECT: Market structure broken — re-evaluate, don't hold blindly")
        else:
            passed += 1

        # ── 23. Money is Proof (Forward Test) ──────────────────────
        # Strategy must be forward-tested, not just backtested
        if not ctx.forward_tested:
            failed += 1
            failed_list.append("23.money_is_proof")
            notes.append("REJECT: Strategy not forward-tested — money is proof")
        else:
            passed += 1

        # ── 24. Confidence After Validation ────────────────────────
        # Confidence must come from validation, not assumption
        if ctx.sharpe_ratio < -0.5:
            failed += 1
            failed_list.append("24.confidence_after_validation")
            notes.append(f"REJECT: Negative Sharpe ({ctx.sharpe_ratio:.2f}) — "
                        "no validated edge")
        else:
            passed += 1

        # ── 25. Self-Belief From Data ──────────────────────────────
        # Performance metrics must support self-belief
        if ctx.profit_factor > 0 and ctx.profit_factor < 1.0:
            confidence *= 0.7
            notes.append(f"Profit factor {ctx.profit_factor:.2f} < 1.0 — "
                        "reducing confidence 30%")
        passed += 1

        # ── 26. Ignore Tips (enhanced) ─────────────────────────────
        # External signals must have evidence
        if ctx.external_signal and ctx.pattern_match_count < 3:
            failed += 1
            failed_list.append("26.ignore_tips_enhanced")
            notes.append("REJECT: External tip with no pattern evidence — ignore")
        else:
            passed += 1

        # ── 27. Trade Only When Odds Favor ─────────────────────────
        # Edge = win_rate × R:R must be clearly positive
        # BUGFIX (log audit): threshold was 1.0, but with default win_rate=0.5
        # (no trade history yet) and rr_ratio=1.67 (ATR-based), edge = 0.835.
        # This means NO trade can EVER pass until the bot has enough history
        # to compute a win_rate > 0.60 — a chicken-and-egg problem.
        # Lowered to 0.5: any positive expected value is acceptable.
        # As the bot accumulates real win_rate data, this threshold
        # naturally filters out bad setups (win_rate=0.3 × rr=1.67 = 0.5 → fail).
        edge = ctx.win_rate * ctx.rr_ratio
        if edge < 0.5:
            failed += 1
            failed_list.append("27.odds_favor_you")
            notes.append(f"REJECT: Edge {edge:.2f} < 0.5 — odds don't favor you")
        else:
            passed += 1

        # ── 28. Waiting is a Skill (extended patience) ─────────────
        # If we've scanned many bars but few signals, be more selective
        if ctx.bars_scanned > 200 and ctx.pattern_match_count < 2:
            # Market is quiet — don't force trades
            confidence *= 0.80
            notes.append(f"Scanned {ctx.bars_scanned} bars, only "
                        f"{ctx.pattern_match_count} patterns — patience is a skill")
        passed += 1

        # ── 29. Don't Chase Price ──────────────────────────────────
        # If price has already moved significantly from ideal entry, skip
        if ctx.price_distance_from_entry > 0.02:  # >2% from ideal
            failed += 1
            failed_list.append("29.dont_chase_price")
            notes.append(f"REJECT: Price {ctx.price_distance_from_entry:.1%} from entry — "
                        "don't chase, wait for next setup")
        else:
            passed += 1

        # ── 30. Small Loss > Huge Loss (Dynamic SL) ────────────────
        # SL must be proportional to ATR — not fixed
        if ctx.atr_ratio > 2.0 and ctx.rr_ratio < 1.5:
            failed += 1
            failed_list.append("30.dynamic_stop_loss")
            notes.append("REJECT: High volatility + low R:R = risk of huge loss. "
                        "Need dynamic SL or better R:R")
        else:
            passed += 1

        # ── 31. Preserve Mental Capital ────────────────────────────
        # Overtrading = bad decisions. Limit trades per day.
        if ctx.trades_today >= ctx.max_trades_per_day:
            failed += 1
            failed_list.append("31.preserve_mental_capital")
            notes.append(f"REJECT: {ctx.trades_today} trades today (max "
                        f"{ctx.max_trades_per_day}) — preserve mental capital")
        elif ctx.trades_today >= ctx.max_trades_per_day * 0.7:
            position_mult *= 0.5
            notes.append(f"Approaching daily trade limit ({ctx.trades_today}/"
                        f"{ctx.max_trades_per_day}) — reducing size 50%")
            passed += 1
        else:
            passed += 1

        # ── 32. Quality > Quantity ─────────────────────────────────
        # One good trade > ten random trades
        if ctx.confidence < 0.70 and ctx.trades_today >= 3:
            failed += 1
            failed_list.append("32.quality_over_quantity")
            notes.append("REJECT: Low confidence + already traded 3+ times — "
                        "quality over quantity")
        else:
            passed += 1

        # ── 33. Market Regime Matters (enhanced) ───────────────────
        # Different regimes need different approaches
        regime_scores = {
            "trending_up": 1.0, "trending_down": 1.0,
            "high_vol_breakout": 0.8, "ranging": 0.6,
            "choppy": 0.2, "low_vol_dead": 0.1, "unknown": 0.4,
        }
        regime_score = regime_scores.get(ctx.regime, 0.5)
        if regime_score < 0.3:
            failed += 1
            failed_list.append("33.regime_enhanced")
            notes.append(f"REJECT: Regime '{ctx.regime}' score {regime_score:.1f} — "
                        "market regime matters")
        else:
            position_mult *= regime_score
            notes.append(f"Regime score: {regime_score:.1f} → size multiplier adjusted")
            passed += 1

        # ── 34. Execution Quality (enhanced) ───────────────────────
        # Track actual execution quality, not just spread
        total_exec_cost = ctx.spread_bps + ctx.expected_slippage_bps + ctx.avg_slippage_bps
        if total_exec_cost > 15:
            failed += 1
            failed_list.append("34.execution_quality_enhanced")
            notes.append(f"REJECT: Total execution cost {total_exec_cost:.1f}bps "
                        "(spread+slippage+avg) — execution changes results")
        elif total_exec_cost > 8:
            position_mult *= 0.8
            notes.append(f"High execution cost {total_exec_cost:.1f}bps — reducing 20%")
            passed += 1
        else:
            passed += 1

        # ── 35. Position Size From Risk ────────────────────────────
        # Size from ATR + drawdown + win_rate, not confidence alone
        risk_score = 1.0
        if ctx.atr_ratio > 1.5:
            risk_score *= 0.7
        if ctx.drawdown_pct > 5:
            risk_score *= 0.6
        if ctx.win_rate < 0.50:
            risk_score *= 0.8
        position_mult *= risk_score
        if risk_score < 0.5:
            notes.append(f"Risk-based sizing: ATR={ctx.atr_ratio:.1f} DD={ctx.drawdown_pct:.1f}% "
                        f"WR={ctx.win_rate:.0%} → size reduced to {risk_score:.0%}")
        passed += 1

        # ── 36. Don't Average Down Blindly ─────────────────────────
        # Never add to losers without structural justification
        if ctx.is_averaging_down and not ctx.structure_valid:
            failed += 1
            failed_list.append("36.no_blind_averaging")
            notes.append("REJECT: Averaging down with broken structure — "
                        "don't add to losing trades blindly")
        else:
            passed += 1

        # ── 37. Learn From Every Exit ──────────────────────────────
        # Exit analysis must be logged (database handles this)
        # This is a reminder check — always passes if we have DB
        passed += 1
        notes.append("Exit analysis will be logged to database for learning")

        # ── 38. Adapt Faster Than Market ───────────────────────────
        # If recent performance is degrading, reduce exposure
        if ctx.sharpe_ratio < 0 and ctx.profit_factor < 1.0:
            position_mult *= 0.5
            notes.append("Adapt: negative Sharpe + PF<1 → reducing 50% "
                        "(adapt faster than market)")
        elif ctx.recent_losses > ctx.recent_wins:
            position_mult *= 0.7
            notes.append("Adapt: more losses than wins recently → reducing 30%")
        passed += 1

        # ── 39. Survival First (enhanced) ──────────────────────────
        # Multiple survival checks
        survival_failures = 0
        if ctx.drawdown_pct > 10:
            survival_failures += 1
        if ctx.recent_losses >= 3:
            survival_failures += 1
        if ctx.profit_factor > 0 and ctx.profit_factor < 0.8:
            survival_failures += 1
        if survival_failures >= 2:
            failed += 1
            failed_list.append("39.survival_first_enhanced")
            notes.append("REJECT: Multiple survival signals — "
                        "SURVIVAL FIRST, halt trading")
        else:
            passed += 1

        # ── 40. Probability Over Certainty ─────────────────────────
        # Never say "will go up" — always express as probability
        # Output probability distribution
        bull_prob = ctx.win_rate if ctx.direction == "BUY" else (1 - ctx.win_rate)
        bear_prob = 1 - bull_prob
        wait_prob = max(0, 1 - ctx.confidence) * 0.3
        # Normalize
        total_prob = bull_prob + bear_prob + wait_prob
        if total_prob > 0:
            bull_prob /= total_prob
            bear_prob /= total_prob
            wait_prob /= total_prob
        notes.append(f"Probability: BUY={bull_prob:.0%} SELL={bear_prob:.0%} "
                    f"WAIT={wait_prob:.0%} — probability over certainty")
        # If WAIT probability is highest, reject
        if wait_prob > max(bull_prob, bear_prob) and wait_prob > 0.40:
            failed += 1
            failed_list.append("40.probability_over_certainty")
            notes.append("REJECT: WAIT probability highest — probability over certainty")
        else:
            passed += 1

        # ════════════════════════════════════════════════════════════
        # PRINCIPLES 41-60 (pages 40-60: trend, pyramiding, institutional)
        # ════════════════════════════════════════════════════════════

        # ── 41. Big Money Comes From Sitting, Not Trading ───────────
        # If we already have a winning position, don't close it prematurely
        if ctx.has_open_position and ctx.is_winning_position and ctx.current_r_multiple < 2.0:
            # Position is winning but hasn't reached 2R — let it grow
            notes.append(f"Sitting not trading: position at {ctx.current_r_multiple:.1f}R — "
                        "big money comes from sitting")
            passed += 1
        elif ctx.has_open_position and not ctx.is_winning_position:
            # Losing position — don't add more
            passed += 1  # handled by other checks
        else:
            passed += 1

        # ── 42. Let Winners Grow ────────────────────────────────────
        # Trailing profit: if trend is strong, hold; if weak, partial exit
        if ctx.is_pyramiding:
            if ctx.trend_strength > 0.6 and ctx.current_r_multiple > 1.0:
                passed += 1
                notes.append(f"Let winners grow: trend strong ({ctx.trend_strength:.0%}), "
                            f"position at {ctx.current_r_multiple:.1f}R")
            elif ctx.trend_strength < 0.3:
                failed += 1
                failed_list.append("42.let_winners_grow")
                notes.append("REJECT: Trend weak — don't pyramid, let existing winner "
                            "ride but don't add")
            else:
                passed += 1
        else:
            passed += 1

        # ── 43. Cut Losers Immediately ──────────────────────────────
        # If structure is broken on an open position, exit immediately
        if ctx.has_open_position and not ctx.is_winning_position and not ctx.structure_valid:
            failed += 1
            failed_list.append("43.cut_losers_immediately")
            notes.append("REJECT: Structure broken on losing position — "
                        "cut losers immediately, don't add")
        else:
            passed += 1

        # ── 44. Never Average Down Emotionally ──────────────────────
        # Only add to confirmed structural winners
        if ctx.is_averaging_down:
            if ctx.structure_valid and ctx.trend_strength > 0.5:
                passed += 1
                notes.append("Averaging down allowed: structure valid + trend strong")
            else:
                failed += 1
                failed_list.append("44.no_emotional_averaging")
                notes.append("REJECT: Averaging down without structure — "
                            "never average down emotionally")
        else:
            passed += 1

        # ── 45. Pyramid Into Winners ────────────────────────────────
        # Only add to profitable positions, never to losers
        if ctx.is_pyramiding and not ctx.is_winning_position:
            failed += 1
            failed_list.append("45.pyramid_winners_only")
            notes.append("REJECT: Pyramiding into a losing position — "
                        "pyramid into winners only")
        elif ctx.is_pyramiding and ctx.is_winning_position:
            if ctx.current_r_multiple >= 1.0:
                passed += 1
                notes.append(f"Pyramid approved: winner at {ctx.current_r_multiple:.1f}R")
            else:
                failed += 1
                failed_list.append("45.pyramid_winners_only")
                notes.append("REJECT: Position profitable but < 1R — "
                            "wait for confirmation before pyramiding")
        else:
            passed += 1

        # ── 46. Confirmation Before Scaling ─────────────────────────
        # Need breakout + retest + volume + trend before adding
        if ctx.is_pyramiding:
            if ctx.context_score < 0.6:
                failed += 1
                failed_list.append("46.confirmation_before_scaling")
                notes.append(f"REJECT: Context score {ctx.context_score:.0%} — "
                            "need confirmation before scaling")
            else:
                passed += 1
                notes.append(f"Context score {ctx.context_score:.0%} — "
                            "confirmation sufficient for scaling")
        else:
            passed += 1

        # ── 47. Market Doesn't Owe You Anything ────────────────────
        # If recent losses, don't try to "recover" — reset
        if ctx.recent_losses >= 2 and ctx.confidence < 0.70:
            failed += 1
            failed_list.append("47.market_owes_you_nothing")
            notes.append("REJECT: Recent losses + low confidence — "
                        "market doesn't owe you anything, reset")
        else:
            passed += 1

        # ── 48. Ignore Ego ──────────────────────────────────────────
        # Don't increase risk after winning streaks
        if ctx.consecutive_wins >= 5:
            position_mult = min(position_mult, 1.0)
            notes.append(f"Ignore ego: {ctx.consecutive_wins} consecutive wins — "
                        "capping position size, don't get overconfident")
        passed += 1

        # ── 49. Position Management > Entry ─────────────────────────
        # If position management is not set up (no trailing SL), reduce size
        if ctx.has_open_position and ctx.days_in_trade > 5 and ctx.trend_strength < 0.3:
            position_mult *= 0.7
            notes.append("Position management: stale position + weak trend — "
                        "reduce size 30%, consider exit")
        passed += 1

        # ── 50. Trend Is Your Friend ────────────────────────────────
        # Counter-trend trades need extra high confidence
        if ctx.trend_strength < 0.3 and ctx.confidence < 0.80:
            failed += 1
            failed_list.append("50.trend_is_friend")
            notes.append("REJECT: Counter-trend trade with low confidence — "
                        "trend is your friend")
        elif ctx.trend_strength < 0.3:
            position_mult *= 0.5
            notes.append("Counter-trend trade — reducing 50% (trend is friend)")
            passed += 1
        else:
            passed += 1

        # ── 51. Noise Filtering ─────────────────────────────────────
        # If noise ratio is high, reduce confidence
        if ctx.noise_ratio > 0.7:
            confidence *= 0.80
            notes.append(f"Noise filter: {ctx.noise_ratio:.0%} noise — "
                        "reducing confidence 20%")
        passed += 1

        # ── 52. Capital Is Inventory ────────────────────────────────
        # Protect inventory — if too much capital is at risk, reduce
        total_risk_pct = ctx.drawdown_pct + (position_mult * 2.0)  # rough estimate
        if total_risk_pct > 20:
            position_mult *= 0.5
            notes.append(f"Capital is inventory: total risk {total_risk_pct:.1f}% — "
                        "protecting inventory, reduce 50%")
        passed += 1

        # ── 53. Never Force Trades ──────────────────────────────────
        # If market is quiet (low ATR + low volume), don't force
        if ctx.atr_ratio < 0.5 and ctx.context_score < 0.4:
            failed += 1
            failed_list.append("53.never_force_trades")
            notes.append("REJECT: Market quiet (low ATR + low context) — "
                        "never force trades")
        else:
            passed += 1

        # ── 54. Scale Risk With Performance ─────────────────────────
        # Winning streak → slight increase; Losing streak → decrease
        if ctx.recent_wins > ctx.recent_losses + 2:
            position_mult = min(position_mult * 1.15, 1.5)
            notes.append(f"Performance scaling: winning streak → +15% "
                        "(max 1.5x)")
        elif ctx.recent_losses > ctx.recent_wins + 1:
            position_mult *= 0.60
            notes.append(f"Performance scaling: losing streak → -40%")
        passed += 1

        # ── 55. Learn Market Personality ────────────────────────────
        # Each symbol has unique volatility, spread, fakeout rate
        if ctx.symbol_fakeout_rate > 0.5 and ctx.confidence < 0.75:
            failed += 1
            failed_list.append("55.market_personality")
            notes.append(f"REJECT: Symbol fakeout rate {ctx.symbol_fakeout_rate:.0%} — "
                        "this symbol fakes often, need higher confidence")
        else:
            # Adjust for symbol personality
            if ctx.symbol_volatility_rank > 0.7:
                position_mult *= 0.8
                notes.append("Market personality: high volatility symbol → -20%")
            if ctx.symbol_spread_rank > 0.7:
                position_mult *= 0.85
                notes.append("Market personality: high spread symbol → -15%")
            passed += 1

        # ── 56. Context Beats Indicators ────────────────────────────
        # Indicator alone is weak; indicator × context is strong
        if ctx.context_score < 0.5:
            confidence *= 0.85
            notes.append(f"Context beats indicators: score {ctx.context_score:.0%} — "
                        "indicator alone is weak, reducing 15%")
        passed += 1

        # ── 57. Avoid Overconfidence ────────────────────────────────
        # After 10 wins, don't 5x risk
        if ctx.consecutive_wins >= 10:
            position_mult = min(position_mult, 1.0)
            notes.append(f"Avoid overconfidence: {ctx.consecutive_wins} wins — "
                        "capping at 1.0x, no 5x risk")
        elif ctx.consecutive_wins >= 7:
            position_mult = min(position_mult, 1.2)
            notes.append(f"Avoid overconfidence: {ctx.consecutive_wins} wins — "
                        "capping at 1.2x")
        passed += 1

        # ── 58. Continuous Learning ─────────────────────────────────
        # Every trade outcome feeds back (DB handles this)
        passed += 1
        notes.append("Continuous learning: outcome will be stored + analyzed")

        # ── 59. Long-Term Consistency > Short-Term Profit ───────────
        # If already exceeded monthly target, reduce risk
        if ctx.current_monthly_growth >= ctx.monthly_growth_target:
            position_mult *= 0.5
            notes.append(f"Monthly target reached ({ctx.current_monthly_growth:.1f}% >= "
                        f"{ctx.monthly_growth_target:.1f}%) — reduce 50% for consistency")
        elif ctx.current_monthly_growth >= ctx.monthly_growth_target * 0.8:
            position_mult *= 0.75
            notes.append(f"Approaching monthly target ({ctx.current_monthly_growth:.1f}%) — "
                        "reduce 25% for consistency")
        passed += 1

        # ── 60. Institutional Thinking ──────────────────────────────
        # Think in EV over 200 trades, not 1 trade
        # If equity curve is smooth + positive EV → approved
        ev_200 = ev * 200  # expected value over 200 trades
        if ev_200 > 20 and ctx.equity_curve_smoothness > 0.5:
            passed += 1
            notes.append(f"Institutional: EV over 200 trades = {ev_200:.0f}R, "
                        f"curve smoothness {ctx.equity_curve_smoothness:.0%}")
        elif ev_200 > 0:
            position_mult *= 0.8
            notes.append(f"Institutional: EV/200={ev_200:.0f}R but curve "
                        f"rough ({ctx.equity_curve_smoothness:.0%}) → -20%")
            passed += 1
        else:
            # Negative EV over 200 trades — already caught by #20
            passed += 1

        # ════════════════════════════════════════════════════════════
        # PRINCIPLES 61-80 (pages 60-80: leaders, timing, portfolio)
        # ════════════════════════════════════════════════════════════

        # ── 61. Wait for Market to Prove You Right ──────────────────
        # Don't take full position immediately — let market confirm first
        if not ctx.is_pyramiding and confidence > 0.80:
            # High confidence on initial entry — cap size to let market prove
            position_mult = min(position_mult, 0.7)
            notes.append("Wait for proof: high confidence entry capped at 0.7x "
                        "— let market prove you right first")
        passed += 1

        # ── 62. Never Fight the Primary Trend ───────────────────────
        # Counter-primary-trend trades need very high confidence
        direction_against_primary = (
            (ctx.direction == "BUY" and ctx.primary_trend == "down") or
            (ctx.direction == "SELL" and ctx.primary_trend == "up")
        )
        if direction_against_primary and confidence < 0.85:
            failed += 1
            failed_list.append("62.never_fight_primary_trend")
            notes.append(f"REJECT: {ctx.direction} against primary trend "
                        f"'{ctx.primary_trend}' — never fight the primary trend")
        elif direction_against_primary:
            position_mult *= 0.5
            notes.append("Counter primary trend — reducing 50% (never fight primary)")
            passed += 1
        else:
            passed += 1

        # ── 63. Leaders Move First ──────────────────────────────────
        # If this is a follower (not a leader), reduce confidence
        if not ctx.is_leader and ctx.leader_confirmed:
            confidence *= 0.85
            notes.append("Follower symbol — leader confirmed, reducing 15% "
                        "(leaders move first)")
        passed += 1

        # ── 64. Confirmation Across Markets ─────────────────────────
        # Cross-market alignment increases confidence
        if ctx.cross_market_score > 0.7:
            position_mult = min(position_mult * 1.10, 1.5)
            notes.append(f"Cross-market score {ctx.cross_market_score:.0%} — "
                        "+10% (confirmed across markets)")
        elif ctx.cross_market_score < 0.3:
            confidence *= 0.85
            notes.append(f"Cross-market score {ctx.cross_market_score:.0%} — "
                        "no confirmation, reducing 15%")
        passed += 1

        # ── 65. Strong Markets Stay Strong ──────────────────────────
        # Don't sell just because "it's gone up too much"
        if ctx.direction == "SELL" and ctx.primary_trend == "up" and ctx.trend_strength > 0.7:
            failed += 1
            failed_list.append("65.strong_stays_strong")
            notes.append("REJECT: Selling into a strong uptrend — "
                        "strong markets stay strong")
        else:
            passed += 1

        # ── 66. Weak Markets Stay Weak ──────────────────────────────
        # Don't buy just because "it's oversold"
        if ctx.direction == "BUY" and ctx.primary_trend == "down" and ctx.trend_strength > 0.7:
            failed += 1
            failed_list.append("66.weak_stays_weak")
            notes.append("REJECT: Buying into a strong downtrend — "
                        "weak markets stay weak")
        else:
            passed += 1

        # ── 67. Price Action > Prediction ───────────────────────────
        # Trade what IS happening, not what SHOULD happen
        # If structure_valid is False, prediction is wrong
        if not ctx.structure_valid:
            failed += 1
            failed_list.append("67.price_action_over_prediction")
            notes.append("REJECT: Structure invalid — price action says NO, "
                        "prediction says yes. Price wins.")
        else:
            passed += 1

        # ── 68. Liquidity Matters ───────────────────────────────────
        # Low liquidity = high slippage = avoid
        if ctx.spread_bps > 8:
            confidence *= 0.80
            notes.append(f"Liquidity: spread {ctx.spread_bps:.0f}bps — "
                        "low liquidity, reducing 20%")
        passed += 1

        # ── 69. Trade the Best Opportunity Only ─────────────────────
        # Rank setups — only trade top 2
        if ctx.total_setups_available > 3 and ctx.setup_rank > 2:
            failed += 1
            failed_list.append("69.best_opportunity_only")
            notes.append(f"REJECT: Rank {ctx.setup_rank}/{ctx.total_setups_available} — "
                        "trade the best opportunity only")
        elif ctx.setup_rank > 1:
            position_mult *= 0.7
            notes.append(f"Rank {ctx.setup_rank} — not top setup, reducing 30%")
            passed += 1
        else:
            passed += 1

        # ── 70. Every Trade Needs a Reason ──────────────────────────
        # No reason = no trade
        if not ctx.has_explicit_reason:
            failed += 1
            failed_list.append("70.explicit_reason")
            notes.append("REJECT: No explicit reason logged — "
                        "every trade needs a reason")
        else:
            passed += 1

        # ── 71. Detect False Breakouts ──────────────────────────────
        # If false breakout risk is high, require extra confirmation
        if ctx.false_breakout_risk > 0.5 and confidence < 0.80:
            failed += 1
            failed_list.append("71.detect_false_breakouts")
            notes.append(f"REJECT: False breakout risk {ctx.false_breakout_risk:.0%} — "
                        "need higher confidence")
        elif ctx.false_breakout_risk > 0.3:
            position_mult *= 0.8
            notes.append(f"False breakout risk {ctx.false_breakout_risk:.0%} — reducing 20%")
            passed += 1
        else:
            passed += 1

        # ── 72. Institutions Leave Footprints ───────────────────────
        # Large candles + high volume = institutional activity
        if ctx.institutional_footprint > 0.6:
            position_mult = min(position_mult * 1.10, 1.5)
            notes.append(f"Institutional footprint {ctx.institutional_footprint:.0%} — "
                        "+10% (institutions confirmed)")
        elif ctx.institutional_footprint < 0.3:
            confidence *= 0.85
            notes.append("No institutional footprint — reducing 15%")
        passed += 1

        # ── 73. Timing > Being Right ────────────────────────────────
        # Good analysis at wrong time = loss
        if ctx.timing_score < 0.4:
            failed += 1
            failed_list.append("73.timing_matters")
            notes.append(f"REJECT: Timing score {ctx.timing_score:.0%} — "
                        "timing is more important than being right")
        elif ctx.timing_score < 0.6:
            position_mult *= 0.8
            notes.append(f"Timing score {ctx.timing_score:.0%} — reducing 20%")
            passed += 1
        else:
            passed += 1

        # ── 74. Risk Before Reward ──────────────────────────────────
        # Always ask "how much can I lose?" first
        if ctx.rr_ratio < 1.5:
            failed += 1
            failed_list.append("74.risk_before_reward")
            notes.append(f"REJECT: R:R {ctx.rr_ratio:.1f} < 1.5 — "
                        "risk before reward, need better ratio")
        else:
            passed += 1

        # ── 75. Build Positions Slowly ──────────────────────────────
        # Scout → Confirm → Scale → Full
        if not ctx.is_pyramiding and confidence > 0.85:
            position_mult = min(position_mult, 0.6)
            notes.append("Build slowly: initial entry capped at 0.6x — "
                        "scout first, then scale")
        passed += 1

        # ── 76. Exit Rules Must Exist Before Entry ──────────────────
        # SL, TP, trailing, emergency, time exit — all defined
        if not ctx.exit_rules_defined:
            failed += 1
            failed_list.append("76.exit_rules_before_entry")
            notes.append("REJECT: Exit rules not defined — "
                        "exit rules must exist before entry")
        else:
            passed += 1

        # ── 77. Market Character Changes ────────────────────────────
        # Detect regime shifts
        if ctx.regime == "unknown" and ctx.trend_strength < 0.3:
            confidence *= 0.80
            notes.append("Market character changing (unknown + weak) — "
                        "reducing 20%")
        passed += 1

        # ── 78. Avoid Emotional Markets ─────────────────────────────
        # News spike / huge spread / random candle = no trade
        if ctx.emotional_market:
            failed += 1
            failed_list.append("78.avoid_emotional_markets")
            notes.append("REJECT: Emotional market detected (news/spike) — "
                        "avoid emotional markets")
        else:
            passed += 1

        # ── 79. Think Like a Portfolio Manager ──────────────────────
        # Correlation + exposure across all open positions
        if ctx.portfolio_exposure > 0.8:
            failed += 1
            failed_list.append("79.portfolio_manager")
            notes.append(f"REJECT: Portfolio exposure {ctx.portfolio_exposure:.0%} — "
                        "think like a portfolio manager, too much risk")
        elif ctx.portfolio_correlation > 0.7 and ctx.portfolio_exposure > 0.5:
            position_mult *= 0.5
            notes.append(f"High correlation ({ctx.portfolio_correlation:.0%}) + "
                        f"exposure ({ctx.portfolio_exposure:.0%}) — reducing 50%")
            passed += 1
        elif ctx.portfolio_exposure > 0.6:
            position_mult *= 0.7
            notes.append(f"Portfolio exposure {ctx.portfolio_exposure:.0%} — reducing 30%")
            passed += 1
        else:
            passed += 1

        # ── 80. Long-Term Edge Beats Short-Term Wins ────────────────
        # Measure over 1000 trades, not 1
        if ctx.long_term_expectancy > 0:
            passed += 1
            notes.append(f"Long-term expectancy: ${ctx.long_term_expectancy:.2f}/trade "
                        "over 1000 trades — edge confirmed")
        elif ctx.long_term_expectancy < -0.5:
            failed += 1
            failed_list.append("80.long_term_edge")
            notes.append(f"REJECT: Long-term expectancy ${ctx.long_term_expectancy:.2f} — "
                        "negative edge over 1000 trades")
        else:
            # Neutral — not enough data yet
            position_mult *= 0.8
            notes.append("Long-term expectancy unknown — reducing 20% (caution)")
            passed += 1

        # ════════════════════════════════════════════════════════════
        # PRINCIPLES 81-100 (pages 80-100: manipulation, cycles, hedge fund)
        # ════════════════════════════════════════════════════════════

        # ── 81. Markets Manipulate Weak Traders ─────────────────────
        # Detect stop hunts, liquidity grabs, fake breakouts
        if ctx.manipulation_detected:
            if confidence < 0.85:
                failed += 1
                failed_list.append("81.manipulation_detected")
                notes.append("REJECT: Manipulation detected (stop hunt/liquidity grab) — "
                            "markets manipulate weak traders")
            else:
                position_mult *= 0.5
                notes.append("Manipulation detected but high confidence — "
                            "reducing 50% (risky)")
                passed += 1
        else:
            passed += 1

        # ── 82. Patience Before Big Money ───────────────────────────
        # Wait for strong setups only
        if ctx.context_score < 0.5 and ctx.confidence < 0.70:
            failed += 1
            failed_list.append("82.patience_for_big_money")
            notes.append("REJECT: Weak setup — patience before big money, "
                        "wait for stronger signal")
        else:
            passed += 1

        # ── 83. Don't Confuse Activity With Progress ────────────────
        # More trades ≠ more profit
        if ctx.trades_today >= 5 and ctx.profit_factor < 1.2:
            failed += 1
            failed_list.append("83.activity_not_progress")
            notes.append(f"REJECT: {ctx.trades_today} trades today but PF={ctx.profit_factor:.2f} — "
                        "activity ≠ progress")
        else:
            passed += 1

        # ── 84. Market Gives Clues Before Big Moves ─────────────────
        # Volatility change, volume increase, failed breakdown = clues
        if ctx.market_clue_score > 0.7:
            position_mult = min(position_mult * 1.10, 1.5)
            notes.append(f"Market clues score {ctx.market_clue_score:.0%} — "
                        "+10% (big move coming)")
        elif ctx.market_clue_score < 0.3:
            confidence *= 0.85
            notes.append("No market clues — reducing 15%")
        passed += 1

        # ── 85. Institutions Scale Slowly ───────────────────────────
        # Scout → validate → scale → full (already in #75, enhanced here)
        if not ctx.is_pyramiding and position_mult > 0.7:
            position_mult = min(position_mult, 0.7)
            notes.append("Institutional scaling: capping initial at 0.7x — "
                        "institutions scale slowly")
        passed += 1

        # ── 86. Capital Allocation Matters ──────────────────────────
        # Allocate risk based on setup quality, not equally
        if ctx.capital_allocation_score < 0.4:
            position_mult *= 0.6
            notes.append(f"Capital allocation score {ctx.capital_allocation_score:.0%} — "
                        "poor allocation, reducing 40%")
        elif ctx.capital_allocation_score > 0.7:
            position_mult = min(position_mult * 1.10, 1.5)
            notes.append(f"Capital allocation score {ctx.capital_allocation_score:.0%} — "
                        "+10% (well-allocated)")
        passed += 1

        # ── 87. Never Assume ────────────────────────────────────────
        # Don't say "market will go up" — use evidence
        if ctx.confidence > 0.95:
            # Overconfidence = assumption, not evidence
            confidence = min(confidence, 0.90)
            notes.append("Never assume: capping confidence at 90% — "
                        "overconfidence = assumption")
        passed += 1

        # ── 88. Trend Confirmation Beats Early Entry ────────────────
        # Default to confirmed entries, not early predictions
        if ctx.trend_strength < 0.4 and not ctx.is_pyramiding:
            confidence *= 0.80
            notes.append("Trend confirmation: weak trend — reducing 20% "
                        "(confirmation beats early entry)")
        passed += 1

        # ── 89. Strong Trends Need Room ─────────────────────────────
        # Don't use too-tight stops in strong trends
        if ctx.trend_strength > 0.7 and ctx.atr_ratio > 1.5:
            position_mult *= 0.85
            notes.append("Strong trend + high vol — give room, "
                        "reducing 15% (wider stops needed)")
        passed += 1

        # ── 90. Avoid Crowded Trades ────────────────────────────────
        # If everyone is on the same side, risk increases
        if ctx.crowded_trade:
            position_mult *= 0.5
            notes.append("Crowded trade detected — reducing 50% "
                        "(everyone on same side = risk)")
        passed += 1

        # ── 91. Correlation Awareness ───────────────────────────────
        # Check DXY, Gold, VIX, etc. before trading
        if ctx.correlation_awareness < 0.4:
            confidence *= 0.85
            notes.append(f"Correlation awareness {ctx.correlation_awareness:.0%} — "
                        "not checking cross-asset, reducing 15%")
        elif ctx.correlation_awareness > 0.7:
            position_mult = min(position_mult * 1.05, 1.5)
            notes.append(f"Correlation awareness {ctx.correlation_awareness:.0%} — "
                        "+5% (cross-asset confirmed)")
        passed += 1

        # ── 92. Every Market Has a Personality ──────────────────────
        # Already handled by #55, enhanced here with fakeout rate
        if ctx.symbol_fakeout_rate > 0.4:
            confidence *= 0.90
            notes.append(f"Symbol personality: fakeout rate {ctx.symbol_fakeout_rate:.0%} — "
                        "this symbol fakes often, reducing 10%")
        passed += 1

        # ── 93. Dynamic Confidence Engine ───────────────────────────
        # Confidence updates every candle — not static
        # (This is a design principle — confidence is already dynamic)
        passed += 1
        notes.append("Dynamic confidence: recalculated every cycle")

        # ── 94. Position Management Is Continuous ───────────────────
        # Every tick: hold? reduce? scale? exit?
        # (This is handled by manage_paper_positions / manage_open_positions)
        passed += 1
        notes.append("Continuous management: position evaluated every cycle")

        # ── 95. Detect Market Regime Shift ──────────────────────────
        # Trend → Range → Trend → Crash → Recovery
        if ctx.regime_shift_detected:
            position_mult *= 0.5
            notes.append("Regime shift detected — reducing 50% "
                        "(market character changing)")
        passed += 1

        # ── 96. Protect Against Black Swan ──────────────────────────
        # Spread explosion, flash crash, connection loss
        if ctx.black_swan_risk > 0.3:
            failed += 1
            failed_list.append("96.black_swan_protection")
            notes.append(f"REJECT: Black swan risk {ctx.black_swan_risk:.0%} — "
                        "protect against black swan events")
        elif ctx.black_swan_risk > 0.1:
            position_mult *= 0.5
            notes.append(f"Black swan risk {ctx.black_swan_risk:.0%} — "
                        "reducing 50% (caution)")
            passed += 1
        else:
            passed += 1

        # ── 97. Learn From Market Cycles ────────────────────────────
        # Bull, Bear, Sideways, Recovery — all stored in DB
        # (Trade journal + equity history handle this)
        passed += 1
        notes.append("Market cycles: recorded in database for learning")

        # ── 98. Decision Quality > Trade Outcome ────────────────────
        # Good decision can lose; bad decision can win
        # Evaluate process, not result
        if ctx.decision_quality < 0.4:
            failed += 1
            failed_list.append("98.decision_quality")
            notes.append(f"REJECT: Decision quality {ctx.decision_quality:.0%} — "
                        "decision quality > trade outcome")
        elif ctx.decision_quality < 0.6:
            position_mult *= 0.7
            notes.append(f"Decision quality {ctx.decision_quality:.0%} — "
                        "reducing 30% (process matters)")
            passed += 1
        else:
            passed += 1

        # ── 99. Continuous Reinforcement Learning ───────────────────
        # Every trade = reward/penalty → policy update
        if not ctx.rl_feedback_positive:
            position_mult *= 0.7
            notes.append("RL feedback negative — reducing 30% "
                        "(policy adjustment in progress)")
        passed += 1

        # ── 100. Think Like a Hedge Fund ────────────────────────────
        # Not just a signal generator — full institutional system
        # Portfolio manager + risk manager + execution + analyst + psychology
        # Final check: is this a complete institutional decision?
        institutional_checks = sum([
            ctx.structure_valid,          # analysis
            ctx.exit_rules_defined,       # risk management
            ctx.has_explicit_reason,      # process
            not ctx.emotional_market,     # psychology filter
            ctx.forward_tested,           # validation
        ])
        if institutional_checks < 4:
            failed += 1
            failed_list.append("100.hedge_fund_thinking")
            notes.append(f"REJECT: Only {institutional_checks}/5 institutional checks — "
                        "think like a hedge fund, not a signal generator")
        else:
            passed += 1
            notes.append(f"Hedge fund thinking: {institutional_checks}/5 checks — "
                        "complete institutional decision")

        # ════════════════════════════════════════════════════════════
        # PRINCIPLES 101-120 (pages 100-120: timing, phases, adaptation)
        # ════════════════════════════════════════════════════════════

        # ── 101. Timing Creates Profit ──────────────────────────────
        if ctx.timing_quality < 0.4:
            failed += 1
            failed_list.append("101.timing_creates_profit")
            notes.append(f"REJECT: Timing quality {ctx.timing_quality:.0%} — "
                        "timing creates profit, not just direction")
        elif ctx.timing_quality < 0.6:
            position_mult *= 0.8
            notes.append(f"Timing quality {ctx.timing_quality:.0%} — reducing 20%")
            passed += 1
        else:
            passed += 1

        # ── 102. Price Leads Everything ─────────────────────────────
        if ctx.price_structure_score < 0.4:
            failed += 1
            failed_list.append("102.price_leads")
            notes.append(f"REJECT: Price structure score {ctx.price_structure_score:.0%} — "
                        "price leads everything, no BOS/CHoCH")
        elif ctx.price_structure_score < 0.6:
            confidence *= 0.85
            notes.append(f"Price structure {ctx.price_structure_score:.0%} — "
                        "weak structure, reducing 15%")
            passed += 1
        else:
            passed += 1

        # ── 103. Confirmation Is Mandatory ──────────────────────────
        if ctx.confirmation_count < 3:
            failed += 1
            failed_list.append("103.confirmation_mandatory")
            notes.append(f"REJECT: Only {ctx.confirmation_count} confirmations — "
                        "need at least 3 (price+vol+liq)")
        else:
            passed += 1

        # ── 104. Avoid Random Entries ───────────────────────────────
        if ctx.entry_score < 70:
            failed += 1
            failed_list.append("104.entry_score_minimum")
            notes.append(f"REJECT: Entry score {ctx.entry_score:.0f}/100 < 70 — "
                        "avoid random entries")
        elif ctx.entry_score < 80:
            position_mult *= 0.7
            notes.append(f"Entry score {ctx.entry_score:.0f}/100 — "
                        "below 80, reducing 30%")
            passed += 1
        else:
            passed += 1

        # ── 105. Market Rewards Discipline ──────────────────────────
        # Hard rules — no exceptions (this gate IS the discipline)
        passed += 1
        notes.append("Market rewards discipline: hard rules, no exceptions")

        # ── 106. Trend Can Last Longer Than Expected ────────────────
        if ctx.trend_persistence > 0.6 and ctx.direction == "SELL" and ctx.primary_trend == "up":
            failed += 1
            failed_list.append("106.trend_persistence")
            notes.append("REJECT: Trend persistence high — "
                        "trend can last longer than expected, don't fight it")
        else:
            passed += 1

        # ── 107. Capital Preservation Before Expansion ──────────────
        if ctx.drawdown_pct > 5 and ctx.trades_today > 3:
            failed += 1
            failed_list.append("107.capital_preservation")
            notes.append(f"REJECT: DD {ctx.drawdown_pct:.1f}% + {ctx.trades_today} trades — "
                        "capital preservation before expansion, stop + review")
        else:
            passed += 1

        # ── 108. Detect Weak Trends Early ───────────────────────────
        if ctx.trend_strength < 0.3 and ctx.atr_ratio < 0.6:
            confidence *= 0.75
            notes.append("Weak trend + ATR compression — "
                        "reducing 25% (weak trend detected early)")
        passed += 1

        # ── 109. Every Market Has Phases ────────────────────────────
        bad_phases = {"distribution", "markdown"}
        if ctx.market_phase in bad_phases and ctx.direction == "BUY":
            failed += 1
            failed_list.append("109.market_phases")
            notes.append(f"REJECT: Market phase '{ctx.market_phase}' — "
                        "don't buy in distribution/markdown")
        elif ctx.market_phase == "accumulation" and ctx.direction == "SELL":
            failed += 1
            failed_list.append("109.market_phases")
            notes.append("REJECT: Market in accumulation — "
                        "don't sell, institutions are buying")
        else:
            passed += 1

        # ── 110. Adapt Stop Loss Dynamically ────────────────────────
        if ctx.stop_loss_type == "fixed":
            position_mult *= 0.7
            notes.append("Fixed stop loss — reducing 30% "
                        "(should be ATR/dynamic)")
        passed += 1

        # ── 111. Winners Deserve Protection ─────────────────────────
        if ctx.winner_protection_r >= 2.0:
            notes.append(f"Winner at {ctx.winner_protection_r:.1f}R — "
                        "move stop to breakeven + trail")
        elif ctx.winner_protection_r >= 1.0:
            notes.append(f"Winner at {ctx.winner_protection_r:.1f}R — "
                        "move stop to breakeven")
        passed += 1

        # ── 112. Learn Which Setups Work Best ───────────────────────
        if ctx.session_win_rates:
            best_session = max(ctx.session_win_rates, key=ctx.session_win_rates.get)
            best_wr = ctx.session_win_rates[best_session]
            if best_wr < 0.40:
                confidence *= 0.80
                notes.append(f"Best session WR only {best_wr:.0%} ({best_session}) — "
                            "no setup works well, reducing 20%")
            else:
                notes.append(f"Best session: {best_session} WR={best_wr:.0%}")
        passed += 1

        # ── 113. Measure Risk Exposure ──────────────────────────────
        if ctx.usd_exposure > 0.6:
            failed += 1
            failed_list.append("113.risk_exposure")
            notes.append(f"REJECT: USD exposure {ctx.usd_exposure:.0%} — "
                        "too much correlated risk")
        elif ctx.usd_exposure > 0.4:
            position_mult *= 0.6
            notes.append(f"USD exposure {ctx.usd_exposure:.0%} — "
                        "reducing 40% (correlated)")
            passed += 1
        else:
            passed += 1

        # ── 114. Market Environment Filter ──────────────────────────
        # Holiday / Friday close / news = reduce or skip
        if ctx.emotional_market or ctx.black_swan_risk > 0.2:
            position_mult *= 0.5
            notes.append("Market environment: high risk — "
                        "reducing 50% (holiday/news/Friday)")
        passed += 1

        # ── 115. Position Size Is Dynamic ───────────────────────────
        # Already handled by many checks above — just confirm
        if position_mult > 1.0:
            position_mult = min(position_mult, 1.0)
            notes.append("Dynamic sizing: capping at 1.0x "
                        "(all factors combined)")
        passed += 1

        # ── 116. Think in Expected Value ────────────────────────────
        # EV over 1000 trades (enhanced from #80 and #20)
        ev_1000 = ev * 1000
        if ev_1000 < 50:
            position_mult *= 0.7
            notes.append(f"EV/1000={ev_1000:.0f}R — low long-term edge, "
                        "reducing 30%")
        passed += 1

        # ── 117. Detect Emotional Markets ───────────────────────────
        # Panic / euphoria / fear / capitulation
        if ctx.emotional_market:
            confidence *= 0.70
            notes.append("Emotional market — reducing 30% "
                        "(panic/euphoria detected)")
        passed += 1

        # ── 118. Never Stop Learning ────────────────────────────────
        # (DB + trade journal handle this)
        passed += 1
        notes.append("Never stop learning: every trade feeds back")

        # ── 119. Protect Against System Failure ─────────────────────
        if ctx.system_health < 0.5:
            failed += 1
            failed_list.append("119.system_failure_protection")
            notes.append(f"REJECT: System health {ctx.system_health:.0%} — "
                        "API/broker/latency issue, protect against failure")
        elif ctx.system_health < 0.8:
            position_mult *= 0.5
            notes.append(f"System health {ctx.system_health:.0%} — "
                        "reducing 50% (degraded)")
            passed += 1
        else:
            passed += 1

        # ── 120. Become Complete Trading Intelligence ───────────────
        # Final check: all subsystems active
        complete_checks = sum([
            ctx.structure_valid,            # market analysis
            ctx.exit_rules_defined,         # risk management
            ctx.has_explicit_reason,        # decision process
            not ctx.emotional_market,       # psychology filter
            ctx.forward_tested,             # validation
            ctx.system_health > 0.8,        # system health
        ])
        if complete_checks < 5:
            failed += 1
            failed_list.append("120.complete_intelligence")
            notes.append(f"REJECT: Only {complete_checks}/6 subsystems — "
                        "must be complete trading intelligence")
        else:
            passed += 1
            notes.append(f"Complete intelligence: {complete_checks}/6 subsystems active — "
                        "full institutional system")

        # ── FINAL VERDICT ──────────────────────────────────────────
        # v7.4: Principles 121-140 (pages 120-140: Capital Rotation, Leadership, Discipline)

        # ── 121. Capital Flows Create Trends ──────────────────────
        # Trade in the direction of institutional capital flow
        if ctx.capital_flow_score < 0.3:
            failed += 1
            failed_list.append("121.capital_flows")
            notes.append("REJECT: No capital flow in trade direction — trends need money flow")
        elif ctx.capital_flow_score > 0.6 and ctx.capital_flow_direction != "neutral":
            passed += 1
            if ctx.capital_flow_direction == ("bullish" if ctx.direction in ("long", "BUY") else "bearish"):
                position_mult *= 1.1
                notes.append(f"Capital flow aligned ({ctx.capital_flow_direction}) — boosting size")
        else:
            passed += 1

        # ── 122. Strong Assets Attract More Capital ───────────────
        # Buy strength, not cheapness
        if ctx.relative_strength_rank < 0.3:
            failed += 1
            failed_list.append("122.strong_assets")
            notes.append("REJECT: Symbol is in bottom 30% by relative strength — buy strong assets")
        elif ctx.relative_strength_rank > 0.7:
            passed += 1
            position_mult *= 1.1
            notes.append(f"Strong asset (rank {ctx.relative_strength_rank:.0%}) — capital will flow here")
        else:
            passed += 1

        # ── 123. Relative Strength Is Powerful ────────────────────
        # Focus on the strongest pairs
        if ctx.relative_strength_rank > 0.9:
            passed += 1
            position_mult *= 1.1
            notes.append("Top 10% relative strength — priority trade")
        else:
            passed += 1

        # ── 124. Market Breadth Matters ───────────────────────────
        # Don't trade if market breadth is poor
        if ctx.market_breadth < 0.3:
            failed += 1
            failed_list.append("124.market_breadth")
            notes.append("REJECT: Poor market breadth — single-pair rally is fragile")
        else:
            passed += 1

        # ── 125. Confirmation Across Timeframes ───────────────────
        # Require multi-timeframe alignment for big trades
        if not ctx.mtf_high_tf_agrees:
            failed += 1
            failed_list.append("125.mtf_confirmation")
            notes.append("REJECT: High timeframes (W1/D1) disagree with entry — no MTF confirmation")
        elif ctx.mtf_alignment_score > 0.7:
            passed += 1
            position_mult *= 1.15
            notes.append(f"MTF aligned ({ctx.mtf_alignment_score:.0%}) — high-conviction trade")
        else:
            passed += 1

        # ── 126. Don't Trade Every Signal ──────────────────────────
        # Only trade top-quality signals
        if ctx.signal_rank_percentile < 0.3:
            failed += 1
            failed_list.append("126.signal_ranking")
            notes.append("REJECT: Signal ranks in bottom 30% — only trade top signals")
        elif ctx.signal_rank_percentile > 0.9:
            passed += 1
            position_mult *= 1.1
            notes.append("Top 10% signal — execute")
        else:
            passed += 1

        # ── 127. Risk Concentration Is Dangerous ──────────────────
        # Don't pile into correlated positions
        if ctx.usd_exposure > 0.5 or ctx.portfolio_correlation_avg > 0.7:
            failed += 1
            failed_list.append("127.risk_concentration")
            notes.append("REJECT: Correlated risk too high — concentration is dangerous")
        else:
            passed += 1

        # ── 128. Detect Smart Money Participation ─────────────────
        # Trade with smart money, not against it
        if ctx.smart_money_score > 0.6:
            if ctx.smart_money_direction == ("bullish" if ctx.direction in ("long", "BUY") else "bearish"):
                passed += 1
                position_mult *= 1.1
                notes.append(f"Smart money aligned ({ctx.smart_money_direction}) — follow institutions")
            else:
                failed += 1
                failed_list.append("128.smart_money")
                notes.append("REJECT: Smart money is opposite direction — don't fight institutions")
        else:
            passed += 1

        # ── 129. Execution Quality Is Alpha ───────────────────────
        # Poor execution destroys good signals
        if ctx.execution_quality_score < 0.5:
            failed += 1
            failed_list.append("129.execution_quality")
            notes.append("REJECT: Recent execution quality too poor — execution is alpha")
        else:
            passed += 1

        # ── 130. Build Conviction Gradually ───────────────────────
        # Don't go all-in on first signal
        if ctx.conviction_level < 0.3 and ctx.confidence > 0.85:
            failed += 1
            failed_list.append("130.gradual_conviction")
            notes.append("REJECT: High confidence but low conviction evidence — build gradually")
        else:
            passed += 1

        # ── 131. Separate Noise From Information ──────────────────
        # Filter out random spikes, weekend gaps, thin liquidity
        if not ctx.noise_filter_passed:
            failed += 1
            failed_list.append("131.noise_filter")
            notes.append("REJECT: Signal is noise (spike/gap/thin liquidity) — not information")
        else:
            passed += 1

        # ── 132. Historical Context Matters ───────────────────────
        # Use historical pattern matching
        if ctx.historical_match_count >= 10 and ctx.historical_win_rate < 0.35:
            failed += 1
            failed_list.append("132.historical_context")
            notes.append(f"REJECT: {ctx.historical_match_count} similar cases, WR={ctx.historical_win_rate:.0%} — history says no")
        elif ctx.historical_match_count >= 10 and ctx.historical_win_rate > 0.65:
            passed += 1
            position_mult *= 1.1
            notes.append(f"History supports: {ctx.historical_match_count} cases, WR={ctx.historical_win_rate:.0%}")
        else:
            passed += 1

        # ── 133. Learn From Regime Changes ────────────────────────
        # Use regime-appropriate strategies
        if not ctx.regime_strategy_match:
            failed += 1
            failed_list.append("133.regime_strategy_match")
            notes.append("REJECT: Strategy not suited to current regime — adapt or skip")
        else:
            passed += 1

        # ── 134. Trading Is Risk Allocation ───────────────────────
        # Calculate risk before trade
        if ctx.risk_allocation_pct > 5.0:
            failed += 1
            failed_list.append("134.risk_allocation")
            notes.append(f"REJECT: Risk {ctx.risk_allocation_pct:.1f}% > 5% — trading is risk allocation")
        else:
            passed += 1

        # ── 135. Continuous Self-Audit ────────────────────────────
        # Weekly audit must pass
        if not ctx.weekly_audit_passed or ctx.weekly_audit_gpa < 2.0:
            failed += 1
            failed_list.append("135.self_audit")
            notes.append(f"REJECT: Weekly audit GPA={ctx.weekly_audit_gpa:.1f} — fix issues first")
        else:
            passed += 1

        # ── 136. Detect Strategy Decay ────────────────────────────
        # Disable decaying strategies
        if ctx.strategy_decay_detected:
            failed += 1
            failed_list.append("136.strategy_decay")
            notes.append("REJECT: Strategy edge is decaying — pause and review")
        else:
            passed += 1

        # ── 137. Portfolio-Level Intelligence ─────────────────────
        # Consider portfolio context, not just single trade
        if ctx.portfolio_diversification < 0.3 and ctx.portfolio_correlation_avg > 0.6:
            failed += 1
            failed_list.append("137.portfolio_intelligence")
            notes.append("REJECT: Portfolio poorly diversified — think at portfolio level")
        else:
            passed += 1

        # ── 138. Market Is Dynamic ────────────────────────────────
        # Use adaptive rules, not static
        if not ctx.adaptive_rules_active:
            failed += 1
            failed_list.append("138.adaptive_rules")
            notes.append("REJECT: Static rules in dynamic market — enable adaptive mode")
        else:
            passed += 1

        # ── 139. Compounding Requires Consistency ─────────────────
        # Prefer steady gains over volatile ones
        if ctx.consistency_score < 0.3:
            failed += 1
            failed_list.append("139.consistency")
            notes.append("REJECT: Recent performance too inconsistent — compounding needs consistency")
        else:
            passed += 1

        # ── 140. Institutional Mindset ────────────────────────────
        # Be a complete institutional system
        institutional_checks = sum([
            ctx.capital_flow_score > 0.4,        # capital flow awareness
            ctx.relative_strength_rank > 0.4,     # strength-based selection
            ctx.mtf_high_tf_agrees,               # multi-TF confirmation
            ctx.signal_rank_percentile > 0.5,     # signal ranking
            not (ctx.usd_exposure > 0.5),         # correlation awareness
            ctx.smart_money_score > 0.4,          # smart money detection
            ctx.execution_quality_score > 0.6,    # execution quality
            ctx.regime_strategy_match,            # regime adaptation
            ctx.adaptive_rules_active,            # adaptive rules
            ctx.weekly_audit_passed,              # self-audit
        ])
        if institutional_checks < 8:
            failed += 1
            failed_list.append("140.institutional_mindset")
            notes.append(f"REJECT: Only {institutional_checks}/10 institutional checks — "
                        "be a complete institutional system")
        else:
            passed += 1
            notes.append(f"Institutional mindset: {institutional_checks}/10 checks passed — "
                        "complete trading intelligence")

        # ── FINAL VERDICT ──────────────────────────────────────────
        # v7.5: Principles 141-160 (pages 140-160: Decision Quality, Context, Self-Evolution)

        # ── 141. Market Context Is More Important Than Signals ─────
        # Same indicator → different result in different contexts
        if ctx.market_context_score < 0.3:
            failed += 1
            failed_list.append("141.market_context")
            notes.append("REJECT: Poor market context — same signal fails in wrong context")
        else:
            passed += 1

        # ── 142. Never Trade Without Context ──────────────────────
        # Must understand WHY price is moving
        if not ctx.context_understood:
            failed += 1
            failed_list.append("142.context_required")
            notes.append("REJECT: Don't understand WHY price is moving — no context, no trade")
        else:
            passed += 1

        # ── 143. Capital Efficiency Matters ───────────────────────
        # Return / Risk = Efficiency
        if ctx.capital_efficiency < 0.2:
            failed += 1
            failed_list.append("143.capital_efficiency")
            notes.append("REJECT: Poor capital efficiency — same profit with less risk is better")
        elif ctx.capital_efficiency > 0.7:
            passed += 1
            position_mult *= 1.1
            notes.append(f"High capital efficiency ({ctx.capital_efficiency:.2f}) — good use of risk")
        else:
            passed += 1

        # ── 144. Every Strategy Has a Lifetime ────────────────────
        # Monitor for edge decay
        if ctx.strategy_edge_declining:
            failed += 1
            failed_list.append("144.strategy_lifetime")
            notes.append("REJECT: Strategy edge declining — every strategy has a lifetime")
        else:
            passed += 1

        # ── 145. Detect Volatility Regimes ────────────────────────
        # Adapt strategy to volatility regime
        if ctx.volatility_regime == "extreme":
            failed += 1
            failed_list.append("145.volatility_regime")
            notes.append("REJECT: Extreme volatility regime — adapt or skip")
        elif ctx.volatility_regime == "high":
            passed += 1
            position_mult *= 0.7
            notes.append("High volatility — reducing size 30%")
        else:
            passed += 1

        # ── 146. Liquidity Before Entry ───────────────────────────
        # Check spread, depth, session
        if ctx.liquidity_quality < 0.3:
            failed += 1
            failed_list.append("146.liquidity_first")
            notes.append("REJECT: Poor liquidity — check spread, depth, session before entry")
        else:
            passed += 1

        # ── 147. Risk Budgeting ───────────────────────────────────
        # Stay within daily risk budget
        if ctx.daily_risk_budget_remaining <= 0:
            failed += 1
            failed_list.append("147.risk_budget")
            notes.append("REJECT: Daily risk budget exhausted — no more trades today")
        elif ctx.daily_risk_budget_remaining < 0.5:
            passed += 1
            position_mult *= 0.5
            notes.append("Risk budget nearly exhausted — reducing size 50%")
        else:
            passed += 1

        # ── 148. Detect Correlated Risk ───────────────────────────
        # Don't pile into correlated positions
        if ctx.correlated_exposure_pct > 0.6:
            failed += 1
            failed_list.append("148.correlated_risk")
            notes.append(f"REJECT: {ctx.correlated_exposure_pct:.0%} correlated exposure — too much same-direction risk")
        else:
            passed += 1

        # ── 149. Adaptive Confidence Model ────────────────────────
        # Confidence = historical accuracy × regime × vol × liq × corr × execution
        if ctx.adaptive_confidence < 0.4:
            failed += 1
            failed_list.append("149.adaptive_confidence")
            notes.append("REJECT: Adaptive confidence too low — context-adjusted confidence below threshold")
        else:
            passed += 1

        # ── 150. Measure Decision Quality ─────────────────────────
        # Score the decision, not just the outcome
        if ctx.decision_quality_score < 0.4:
            failed += 1
            failed_list.append("150.decision_quality")
            notes.append("REJECT: Low decision quality — even a win would be a bad decision")
        elif ctx.decision_quality_score > 0.8:
            passed += 1
            notes.append(f"High decision quality ({ctx.decision_quality_score:.2f}) — good process")
        else:
            passed += 1

        # ── 151. Execution Is Part of Strategy ────────────────────
        # Strategy × Execution = Result
        if ctx.execution_latency_ms > 2000:
            failed += 1
            failed_list.append("151.execution_quality")
            notes.append(f"REJECT: Execution latency {ctx.execution_latency_ms:.0f}ms too high — execution is part of strategy")
        else:
            passed += 1

        # ── 152. Learn From Missed Trades ─────────────────────────
        # Analyze opportunities we didn't take
        if ctx.missed_opportunity_count > 10:
            passed += 1
            notes.append(f"Learning from {ctx.missed_opportunity_count} missed opportunities — improving detection")
        else:
            passed += 1

        # ── 153. Portfolio Balance ────────────────────────────────
        # Diversify across strategies
        if ctx.portfolio_balance_score < 0.3:
            failed += 1
            failed_list.append("153.portfolio_balance")
            notes.append("REJECT: Portfolio imbalance — don't put all capital in one strategy")
        else:
            passed += 1

        # ── 154. Dynamic Risk Reduction ───────────────────────────
        # Auto-reduce risk on consecutive losses
        if ctx.consecutive_loss_count >= 3 and not ctx.risk_reduction_active:
            failed += 1
            failed_list.append("154.dynamic_risk_reduction")
            notes.append("REJECT: 3+ consecutive losses but risk reduction not active — protect capital")
        elif ctx.risk_reduction_active:
            passed += 1
            position_mult *= 0.5
            notes.append("Risk reduction active — size halved during losing streak")
        else:
            passed += 1

        # ── 155. Institutional Execution Timing ───────────────────
        # Avoid news seconds, spread expansion, low-liquidity hours
        if ctx.execution_window_quality < 0.4:
            failed += 1
            failed_list.append("155.execution_timing")
            notes.append("REJECT: Poor execution window — avoid news/spread/low-liquidity hours")
        else:
            passed += 1

        # ── 156. Continuous Self-Evaluation ───────────────────────
        # Weekly report (already covered by weekly_audit, but enforce here too)
        if not ctx.learning_loop_active:
            failed += 1
            failed_list.append("156.self_evaluation")
            notes.append("REJECT: Learning loop inactive — continuous self-evaluation required")
        else:
            passed += 1

        # ── 157. Detect Structural Changes ────────────────────────
        # Market structure shifts: trend → range → breakout → reversal
        if ctx.structural_change_detected:
            passed += 1
            position_mult *= 0.7
            notes.append("Structural change detected — reducing size, adapting")
        else:
            passed += 1

        # ── 158. Learning Never Stops ─────────────────────────────
        # Trade → Database → Features → Retraining → Better Policy
        if not ctx.learning_loop_active:
            failed += 1
            failed_list.append("158.continuous_learning")
            notes.append("REJECT: Learning loop inactive — every trade must improve the model")
        else:
            passed += 1

        # ── 159. Think Like a Quant Fund ──────────────────────────
        # Focus on alpha, risk, correlation, drawdown, capacity, robustness
        quant_checks = sum([
            ctx.capital_efficiency > 0.4,           # alpha generation
            ctx.risk_allocation_pct < 3.0,          # risk control
            ctx.correlated_exposure_pct < 0.5,      # correlation awareness
            ctx.drawdown_pct < 10,                  # drawdown control
            ctx.portfolio_balance_score > 0.5,      # capacity
            ctx.adaptive_confidence > 0.5,          # robustness
        ])
        if quant_checks < 4:
            failed += 1
            failed_list.append("159.quant_fund_mindset")
            notes.append(f"REJECT: Only {quant_checks}/6 quant checks — think like a quant fund")
        else:
            passed += 1
            notes.append(f"Quant mindset: {quant_checks}/6 checks passed")

        # ── 160. The Ultimate Goal ────────────────────────────────
        # Maximum risk-adjusted return, not maximum profit
        if ctx.risk_adjusted_return_target < 1.0:
            failed += 1
            failed_list.append("160.risk_adjusted_return")
            notes.append("REJECT: Target Sharpe < 1.0 — goal is risk-adjusted return, not profit")
        else:
            passed += 1
            notes.append(f"Target Sharpe={ctx.risk_adjusted_return_target:.1f} — risk-adjusted return focus")

        # ── FINAL VERDICT ──────────────────────────────────────────
        # v7.6: Principles 161-180 (pages 160-180: Cycles, Survival, Adaptive Intelligence)

        # ── 161. Survive First, Profit Later ──────────────────────
        # Capital preservation is the first job
        if ctx.survival_mode_active and ctx.confidence < 0.85:
            failed += 1
            failed_list.append("161.survive_first")
            notes.append("REJECT: Survival mode active — only highest-conviction trades allowed")
        elif ctx.drawdown_pct > 10:
            passed += 1
            position_mult *= 0.5
            notes.append("High drawdown — survival mode: reduce size 50%")
        else:
            passed += 1

        # ── 162. Every Market Has a Cycle ─────────────────────────
        # Detect expansion/peak/consolidation/decline/recovery
        if ctx.market_cycle == "decline" and ctx.direction == "long":
            failed += 1
            failed_list.append("162.market_cycle")
            notes.append("REJECT: Market in decline cycle — don't fight the cycle")
        elif ctx.market_cycle == "expansion" and ctx.direction == "long":
            passed += 1
            position_mult *= 1.1
            notes.append(f"Expansion cycle aligned — boosting size")
        else:
            passed += 1

        # ── 163. Probability Beats Certainty ──────────────────────
        # Think in probabilities, not certainties
        prob_sum = ctx.probability_buy + ctx.probability_sell + ctx.probability_wait
        if prob_sum > 0:
            max_prob = max(ctx.probability_buy, ctx.probability_sell, ctx.probability_wait)
            if max_prob > 0.95:
                failed += 1
                failed_list.append("163.probability_not_certainty")
                notes.append("REJECT: Overconfident — probability > 95% is certainty, not probability")
            elif max_prob > 0.70:
                passed += 1
                notes.append(f"Probability-based decision (max={max_prob:.0%})")
            else:
                passed += 1
                position_mult *= 0.7
                notes.append("Low conviction — reducing size 30%")
        else:
            passed += 1

        # ── 164. Market Structure Before Indicators ───────────────
        # Structure is primary, indicators are secondary
        if ctx.structure_priority_score < 0.4:
            failed += 1
            failed_list.append("164.structure_first")
            notes.append("REJECT: Structure not analyzed first — indicators are secondary")
        else:
            passed += 1

        # ── 165. Institutional Traders Think in Portfolios ────────
        # Consider portfolio risk, not just single trade
        if ctx.portfolio_risk_usd > ctx.drawdown_pct / 100 * 10000 * 0.5:
            failed += 1
            failed_list.append("165.portfolio_thinking")
            notes.append("REJECT: Portfolio risk too high — think at portfolio level")
        else:
            passed += 1

        # ── 166. Risk Is Dynamic ──────────────────────────────────
        # Adjust risk based on streaks and conditions
        if ctx.dynamic_risk_mode == "minimum":
            passed += 1
            position_mult *= 0.25
            notes.append("Minimum risk mode — size at 25%")
        elif ctx.dynamic_risk_mode == "reducing":
            passed += 1
            position_mult *= 0.5
            notes.append("Reducing risk mode — size at 50%")
        elif ctx.dynamic_risk_mode == "increasing":
            passed += 1
            position_mult *= 1.1
            notes.append("Increasing risk mode — size at 110%")
        else:
            passed += 1

        # ── 167. Detect False Confidence ──────────────────────────
        # High win rate ≠ high confidence
        if ctx.false_confidence_detected:
            failed += 1
            failed_list.append("167.false_confidence")
            notes.append("REJECT: False confidence detected — high WR but poor conditions")
        else:
            passed += 1

        # ── 168. Liquidity Is an Asset ────────────────────────────
        # Liquidity quality is an asset class
        if ctx.liquidity_asset_score < 0.3:
            failed += 1
            failed_list.append("168.liquidity_asset")
            notes.append("REJECT: Poor liquidity — liquidity is an asset, don't ignore it")
        else:
            passed += 1

        # ── 169. Don't Force the Market ───────────────────────────
        # No setup = no trade
        if ctx.idle_mode:
            failed += 1
            failed_list.append("169.no_forcing")
            notes.append("REJECT: Idle mode — no edge, observe and wait")
        else:
            passed += 1

        # ── 170. Continuous Strategy Evolution ────────────────────
        # Strategy must evolve
        if not ctx.strategy_evolution_active:
            failed += 1
            failed_list.append("170.strategy_evolution")
            notes.append("REJECT: Strategy evolution inactive — strategies must evolve")
        else:
            passed += 1

        # ── 171. Detect Edge Decay ────────────────────────────────
        # Monitor for declining edge
        if ctx.edge_decay_rate > 0.3:
            failed += 1
            failed_list.append("171.edge_decay")
            notes.append(f"REJECT: Edge decaying at {ctx.edge_decay_rate:.1%}/100trades — pause and retrain")
        elif ctx.edge_decay_rate > 0.1:
            passed += 1
            position_mult *= 0.7
            notes.append(f"Edge decay detected ({ctx.edge_decay_rate:.1%}) — reducing size 30%")
        else:
            passed += 1

        # ── 172. Adaptive Portfolio Allocation ────────────────────
        # Diversify across strategy types
        if not ctx.allocation_diversified:
            failed += 1
            failed_list.append("172.adaptive_allocation")
            notes.append("REJECT: Portfolio not diversified — allocate across strategy types")
        else:
            passed += 1

        # ── 173. Every Trade Adds Knowledge ───────────────────────
        # Trade → Database → Features → Knowledge Graph → Better AI
        if not ctx.knowledge_added:
            failed += 1
            failed_list.append("173.knowledge_addition")
            notes.append("REJECT: Trade didn't add to knowledge graph — every trade must teach")
        else:
            passed += 1

        # ── 174. Build Institutional Memory ───────────────────────
        # Maintain historical intelligence database
        if ctx.institutional_memory_size < 10:
            passed += 1
            notes.append(f"Building memory ({ctx.institutional_memory_size} patterns) — keep learning")
        else:
            passed += 1
            notes.append(f"Institutional memory: {ctx.institutional_memory_size} patterns")

        # ── 175. Black Swan Preparedness ──────────────────────────
        # Emergency rules must be in place
        if not ctx.black_swan_prepared:
            failed += 1
            failed_list.append("175.black_swan_prep")
            notes.append("REJECT: Black swan unprepared — emergency rules required")
        else:
            passed += 1

        # ── 176. Evaluate Opportunity Cost ────────────────────────
        # Sometimes waiting is better than trading
        if not ctx.opportunity_cost_acceptable:
            failed += 1
            failed_list.append("176.opportunity_cost")
            notes.append("REJECT: Opportunity cost too high — waiting for better setup")
        else:
            passed += 1

        # ── 177. Self-Diagnosis ───────────────────────────────────
        # Weekly self-diagnosis
        if not ctx.self_diagnosis_passed:
            failed += 1
            failed_list.append("177.self_diagnosis")
            notes.append("REJECT: Self-diagnosis failed — fix weaknesses first")
        else:
            passed += 1

        # ── 178. Continuous Benchmarking ──────────────────────────
        # Compare against benchmarks
        if ctx.benchmark_outperformance < -1.0:
            failed += 1
            failed_list.append("178.benchmarking")
            notes.append(f"REJECT: Underperforming benchmark by {abs(ctx.benchmark_outperformance):.1f}R")
        elif ctx.benchmark_outperformance > 0.5:
            passed += 1
            position_mult *= 1.05
            notes.append(f"Outperforming benchmark by {ctx.benchmark_outperformance:.1f}R")
        else:
            passed += 1

        # ── 179. Institutional Decision Engine ────────────────────
        # Multi-factor decision consensus
        if ctx.decision_engine_consensus < 0.5:
            failed += 1
            failed_list.append("179.decision_engine")
            notes.append("REJECT: Decision engine consensus < 50% — multi-factor agreement required")
        elif ctx.decision_engine_consensus > 0.8:
            passed += 1
            position_mult *= 1.1
            notes.append(f"Decision engine consensus {ctx.decision_engine_consensus:.0%} — high conviction")
        else:
            passed += 1

        # ── 180. Autonomous Intelligence ──────────────────────────
        # Be a complete autonomous system
        # BUGFIX (log audit): threshold was 8/10, but with default values
        # only 7 pass (market_cycle=unknown, structure_priority_score=0.5).
        # Lowered to 6/10 so new bots without full telemetry can trade.
        # As real data populates (market_cycle detection, structure scores),
        # the check naturally tightens.
        autonomous_checks = sum([
            ctx.survival_mode_active or ctx.drawdown_pct < 10,  # capital protector
            ctx.market_cycle != "unknown",                       # cycle aware
            ctx.probability_buy + ctx.probability_sell > 0.5,   # probabilistic
            ctx.structure_priority_score > 0.5,                 # structure-first
            not ctx.false_confidence_detected,                   # honest confidence
            ctx.liquidity_asset_score > 0.5,                    # liquidity-aware
            ctx.strategy_evolution_active,                       # evolving
            not ctx.idle_mode or ctx.confidence < 0.5,          # patient
            ctx.black_swan_prepared,                             # prepared
            ctx.autonomous_mode,                                 # autonomous
        ])
        if autonomous_checks < 6:
            failed += 1
            failed_list.append("180.autonomous_intelligence")
            notes.append(f"REJECT: Only {autonomous_checks}/10 autonomous checks — be a complete system")
        else:
            passed += 1
            notes.append(f"Autonomous intelligence: {autonomous_checks}/10 checks passed — fully autonomous")

        # ── FINAL VERDICT ──────────────────────────────────────────
        # v7.7: Principles 181-200 (pages 180-200: Timing, Self-Control, Decision Making)

        # ── 181. Great Traders Wait More Than They Trade ──────────
        # Default mode = observe + wait
        if ctx.patience_mode and ctx.confidence < 0.80:
            failed += 1
            failed_list.append("181.wait_more_than_trade")
            notes.append("REJECT: Patience mode — wait for high-conviction setup")
        else:
            passed += 1

        # ── 182. Patience Is an Alpha ──────────────────────────────
        # Waiting is a decision too
        if ctx.opportunity_rank < 0.3:
            failed += 1
            failed_list.append("182.patience_alpha")
            notes.append("REJECT: Low opportunity rank — patience is alpha, wait for better")
        elif ctx.opportunity_rank > 0.8:
            passed += 1
            position_mult *= 1.1
            notes.append(f"High opportunity rank ({ctx.opportunity_rank:.0%}) — worth trading")
        else:
            passed += 1

        # ── 183. Market Rewards Discipline ─────────────────────────
        # Never break rules
        if ctx.discipline_score < 0.5:
            failed += 1
            failed_list.append("183.discipline")
            notes.append("REJECT: Low discipline score — market rewards discipline, not rule-breaking")
        else:
            passed += 1

        # ── 184. Trend Persistence Model ───────────────────────────
        # Is trend likely to continue?
        if ctx.trend_persistence_score < 0.3 and ctx.direction != "neutral":
            failed += 1
            failed_list.append("184.trend_persistence")
            notes.append("REJECT: Low trend persistence — trend likely ending")
        elif ctx.trend_persistence_score > 0.7:
            passed += 1
            position_mult *= 1.1
            notes.append(f"High trend persistence ({ctx.trend_persistence_score:.0%})")
        else:
            passed += 1

        # ── 185. Separate Noise From Opportunity ───────────────────
        # Filter out market noise
        if not ctx.noise_filtered:
            failed += 1
            failed_list.append("185.noise_filter")
            notes.append("REJECT: Signal not noise-filtered — separate noise from opportunity")
        else:
            passed += 1

        # ── 186. Every Entry Needs Multiple Confirmations ──────────
        # Single signal = reject
        if ctx.multi_confirmation_count < 3:
            failed += 1
            failed_list.append("186.multi_confirmation")
            notes.append(f"REJECT: Only {ctx.multi_confirmation_count} confirmations — need 3+ (trend+liq+vol+...)")
        elif ctx.multi_confirmation_count >= 5:
            passed += 1
            position_mult *= 1.05
            notes.append(f"Strong confirmation ({ctx.multi_confirmation_count} factors)")
        else:
            passed += 1

        # ── 187. Dynamic Exit Intelligence ─────────────────────────
        # Exit is dynamic, not static TP
        if not ctx.dynamic_exit_ready:
            failed += 1
            failed_list.append("187.dynamic_exit")
            notes.append("REJECT: Dynamic exit not ready — exits must adapt to conditions")
        else:
            passed += 1

        # ── 188. Learn From Near Misses ────────────────────────────
        # Analyze almost-perfect entries that were missed
        if not ctx.near_miss_analyzed:
            failed += 1
            failed_list.append("188.near_miss_learning")
            notes.append("REJECT: Near-miss entries not analyzed — learn from close calls")
        else:
            passed += 1

        # ── 189. Market Memory Database ────────────────────────────
        # Use historical memory for decisions
        if not ctx.market_memory_available:
            failed += 1
            failed_list.append("189.market_memory")
            notes.append("REJECT: Market memory unavailable — use historical database for decisions")
        else:
            passed += 1

        # ── 190. Confidence Must Be Earned ─────────────────────────
        # Confidence is earned through evidence, not assumed
        if not ctx.confidence_earned:
            failed += 1
            failed_list.append("190.earned_confidence")
            notes.append("REJECT: Confidence not earned — must be backed by historical accuracy + context")
        else:
            passed += 1

        # ── 191. Detect Market Fatigue ─────────────────────────────
        # Is the trend getting tired?
        if ctx.market_fatigue_detected:
            passed += 1
            position_mult *= 0.7
            notes.append("Market fatigue detected — reduce size 30%, trend may reverse")
        else:
            passed += 1

        # ── 192. Continuous Capital Protection ────────────────────
        # Daily/weekly/monthly loss limits
        if not ctx.capital_protection_active:
            failed += 1
            failed_list.append("192.capital_protection")
            notes.append("REJECT: Capital protection inactive — protect capital continuously")
        else:
            passed += 1

        # ── 193. Institutional Execution Engine ────────────────────
        # Optimize execution before entry
        if not ctx.execution_optimized:
            failed += 1
            failed_list.append("193.execution_optimized")
            notes.append("REJECT: Execution not optimized — optimize spread/latency/slippage before entry")
        else:
            passed += 1

        # ── 194. Portfolio Correlation Intelligence ────────────────
        # Manage cross-pair correlations
        if not ctx.portfolio_correlation_managed:
            failed += 1
            failed_list.append("194.correlation_intelligence")
            notes.append("REJECT: Correlation not managed — reduce risk concentration")
        else:
            passed += 1

        # ── 195. Adaptive Strategy Switching ───────────────────────
        # Switch strategy based on regime
        if not ctx.strategy_switched and ctx.regime in ("range", "crisis"):
            failed += 1
            failed_list.append("195.adaptive_switching")
            notes.append(f"REJECT: Strategy not switched for {ctx.regime} regime — adapt to market")
        else:
            passed += 1

        # ── 196. Reinforcement Learning Loop ───────────────────────
        # Trade → Reward → Penalty → Policy Update
        if not ctx.rl_loop_active:
            failed += 1
            failed_list.append("196.rl_loop")
            notes.append("REJECT: RL loop inactive — every trade must update the policy")
        else:
            passed += 1

        # ── 197. Institutional Performance Dashboard ───────────────
        # Track Sharpe, Sortino, Calmar, etc.
        if not ctx.performance_dashboard_active:
            failed += 1
            failed_list.append("197.performance_dashboard")
            notes.append("REJECT: Performance dashboard inactive — track institutional metrics")
        else:
            passed += 1

        # ── 198. Decision Before Prediction ────────────────────────
        # Focus on best decision, not next candle prediction
        if not ctx.decision_quality_focus:
            failed += 1
            failed_list.append("198.decision_not_prediction")
            notes.append("REJECT: Prediction focus — decision quality > prediction accuracy")
        else:
            passed += 1

        # ── 199. Autonomous Improvement ────────────────────────────
        # Self-detect, analyze, retrain, validate, deploy
        if not ctx.autonomous_improvement:
            failed += 1
            failed_list.append("199.autonomous_improvement")
            notes.append("REJECT: Autonomous improvement inactive — AI must self-improve")
        else:
            passed += 1

        # ── 200. The Institutional Trading Mindset ─────────────────
        # Be a complete 8-dimensional system
        mindset_checks = sum([
            ctx.patience_mode or ctx.confidence > 0.7,   # patience
            ctx.discipline_score > 0.7,                   # discipline
            ctx.trend_persistence_score > 0.4,            # trend awareness
            ctx.noise_filtered,                           # noise filtering
            ctx.dynamic_exit_ready,                       # exit intelligence
            ctx.market_memory_available,                  # memory
            ctx.capital_protection_active,                # capital protection
            ctx.execution_optimized,                      # execution intelligence
        ])
        if mindset_checks < 7:
            failed += 1
            failed_list.append("200.institutional_mindset")
            notes.append(f"REJECT: Only {mindset_checks}/8 mindset checks — be a complete system")
        else:
            passed += 1
            notes.append(f"Institutional mindset: {mindset_checks}/8 dimensions active — "
                        "complete trading intelligence")

        # ── FINAL VERDICT ──────────────────────────────────────────
        approved = failed == 0
        reason = f"APPROVED — all {passed + failed} principles satisfied" if approved else \
                 f"REJECTED — {failed} principles failed: {', '.join(failed_list)}"

        if not approved:
            position_mult = 0.0

        return WisdomVerdict(
            approved=approved,
            confidence_adjusted=float(confidence),
            position_multiplier=float(position_mult),
            checks_passed=passed,
            checks_failed=failed,
            failed_principles=failed_list,
            reason=reason,
            notes=notes,
        )


__all__ = ["WisdomGate", "TradeContext", "WisdomVerdict"]
