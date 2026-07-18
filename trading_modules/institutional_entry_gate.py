"""
Institutional Entry Gate — The "Should I Trade Now?" Brain
============================================================

THE PROBLEM THIS MODULE SOLVES
-------------------------------
Most AI trading bots fail not because their analysis is wrong, but because
they cannot answer one question: **"Should I trade RIGHT NOW?"**

A professional trader will sit through 50 candles and take ZERO trades,
because none of them are A+ setups. An amateur AI trades 30 of those 50
candles — and loses money on 22 of them.

This module is the institutional trader's brain. It answers:

    1. Is there a trend?           (20 pts)
    2. Is price at a key level?    (20 pts)
    3. Where is the liquidity?     (20 pts)
    4. Is volume confirming?       (15 pts)
    5. Is momentum confirming?     (15 pts)
    6. Is Risk:Reward acceptable?  (10 pts)
                                  ----
                          TOTAL     100 pts

    Score < 70  → NO_TRADE (WAIT)
    70 ≤ s < 85 → WEAK      (skip unless confluence is exceptional)
    Score ≥ 85  → STRONG    (entry approved)

THREE-DECISION SYSTEM
---------------------
Unlike naive bots that always output BUY or SELL, this gate outputs:
    - BUY       (strong long setup)
    - SELL      (strong short setup)
    - NO_TRADE  (WAIT — the most important decision)

NO-TRADE CONDITIONS (instant skip, no scoring needed)
-----------------------------------------------------
    - Market in middle of range (no edge)
    - Major news within ±15 min
    - Low-liquidity session (Asian dead hours for crypto)
    - Spread > 2x normal
    - ATR < 0.3x baseline (dead market) or ATR > 3x baseline (chaotic)
    - Fake breakout detected
    - R:R < 2:1
    - No structure confirmation (no BOS / CHoCH / retest)
    - HTF bias conflicts with LTF signal

INSTITUTIONAL 10/10 CHECKLIST (all must pass)
---------------------------------------------
    1.  Trend aligned (LTF + HTF same direction)
    2.  HTF (H4/D1) bias clear
    3.  Price at key level (S/R / OB / FVG / PDH/PDL / WHL/WHL)
    4.  Liquidity taken (sweep / grab)
    5.  BOS or CHoCH confirmed
    6.  Retest complete
    7.  Volume supports move (≥ 1.5x average)
    8.  Spread acceptable
    9.  R:R ≥ 2:1 (prefer 3:1+)
   10.  Risk within limits (Kelly fraction, max position size)

If ANY of the 10 fails → NO_TRADE.

Usage:
    from trading_modules.institutional_entry_gate import (
        InstitutionalEntryGate, EntryInput, EntryDecision
    )

    gate = InstitutionalEntryGate(config_dict)
    decision = gate.evaluate(EntryInput(
        symbol="BTCUSD",
        direction="BUY",
        df_m15=candles_15m,
        df_h1=candles_1h,
        df_h4=candles_4h,
        df_d1=candles_1d,
        entry_price=65200.0,
        stop_loss=64800.0,
        take_profit=66400.0,
        spread_points=15,
        news_in_minutes=120,    # next high-impact news in 120 min
        session="london",       # sydney/tokyo/london/newyork/overlap/off
    ))

    if decision.action == "BUY":
        execute_order(decision)
    elif decision.action == "SELL":
        execute_order(decision)
    else:  # NO_TRADE
        log.info(f"Skipping: {decision.skip_reason}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any

import numpy as np
import pandas as pd

# BUG FIX: regime_allows_trading() is called in evaluate() (Step 1.5) but
# was never imported — every call would raise NameError at that point,
# crashing entry evaluation unconditionally once regime detection ran.
from trading_modules.market_regime import regime_allows_trading

# Critical #1 fix: all confluence module imports are now LAZY (inside
# __init__) so that a missing or broken module doesn't crash the entire
# entry gate on import. Each module is imported individually with a
# try/except — if it fails, a warning is logged and that module is
# skipped, but the gate still functions with the remaining modules.
#
# This graceful degradation is critical for production reliability:
# a single broken import should not prevent the bot from trading.

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Decision types
# ----------------------------------------------------------------------
class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"                    # setup not yet complete — price approaching level
    SKIP = "SKIP"                    # setup quality is bad — don't trade this
    # Backward-compat aliases
    NO_TRADE = "SKIP"
    WEAK = "WAIT"


# ----------------------------------------------------------------------
# Input container
# ----------------------------------------------------------------------
@dataclass
class EntryInput:
    """All inputs the institutional gate needs to make a decision."""
    # ── Required ──
    symbol: str
    direction: str                      # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    # ── OHLCV dataframes (column names: open, high, low, close, volume) ──
    df_m15: Optional[pd.DataFrame] = None
    df_h1: Optional[pd.DataFrame] = None
    df_h4: Optional[pd.DataFrame] = None
    df_d1: Optional[pd.DataFrame] = None
    # ── Market context ──
    spread_points: float = 0.0          # current spread in points
    news_in_minutes: int = 9999         # min until next high-impact news
    session: str = "off"                # sydney/tokyo/london/newyork/overlap/off
    # ── Pre-computed hints (optional) ──
    at_key_zone: Optional[bool] = None  # caller may pre-mark this
    liquidity_sweep: Optional[bool] = None
    structure_break: Optional[str] = None   # BOS / CHoCH / None
    pattern: Optional[str] = None
    pattern_rating: Optional[int] = None    # 1-5
    candle_closed: bool = True
    # ── v6.1: Cross-asset data for CrossAssetConfirmation ──
    related_dfs: Optional[dict] = None  # {"DXY": df, "GOLD": df, ...}


# ----------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------
@dataclass
class EntryDecision:
    action: str                         # Action enum value
    score: float                        # 0-100
    grade: str                          # A+ / A / B / C / F
    checklist: dict[str, bool]          # 10-item institutional checklist
    failed_checks: list[str]            # list of failed check names
    skip_reason: Optional[str] = None   # why SKIP / WAIT
    # Score breakdown
    score_breakdown: dict[str, float] = field(default_factory=dict)
    # Risk metrics
    risk_pips: Optional[float] = None
    reward_pips: Optional[float] = None
    rr_ratio: Optional[float] = None
    # SMC / structure context
    trend_ltf: Optional[str] = None     # bullish / bearish / ranging
    trend_htf: Optional[str] = None
    at_key_level: bool = False
    liquidity_taken: bool = False
    confirmation: Optional[str] = None  # BOS / CHoCH / ENGULFING / RETEST
    # v5.5 additions
    confidence_pct: float = 0.0         # 0..100 — overall confidence in the trade
    win_probability: float = 0.0        # 0..1 — estimated probability of TP hit
    regime: Optional[str] = None        # market regime from MarketRegimeDetector
    candle_quality_score: Optional[float] = None  # 0..1 from CandleQualityAnalyzer
    candle_quality_label: Optional[str] = None
    htf_alignment: bool = False         # D1+H4+H1+M15 all aligned?
    # v6.1: Confluence module results (full transparency)
    confluence_results: dict = field(default_factory=dict)
    confluence_bonuses: dict = field(default_factory=dict)  # {module_name: bonus_value}
    confluence_penalties: dict = field(default_factory=dict)
    # Meta
    symbol: str = ""
    direction: str = ""
    timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def should_execute(self) -> bool:
        """True only if action is BUY or SELL."""
        return self.action in (Action.BUY.value, Action.SELL.value)

    def is_skip(self) -> bool:
        """True if action is SKIP (bad quality — don't trade this)."""
        return self.action == Action.SKIP.value

    def is_wait(self) -> bool:
        """True if action is WAIT (setup not yet complete — price approaching)."""
        return self.action == Action.WAIT.value

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": round(self.score, 1),
            "grade": self.grade,
            "confidence_pct": round(self.confidence_pct, 1),
            "win_probability": round(self.win_probability, 3),
            "checklist": self.checklist,
            "failed_checks": self.failed_checks,
            "skip_reason": self.skip_reason,
            "score_breakdown": {k: round(v, 1) for k, v in self.score_breakdown.items()},
            "risk_pips": self.risk_pips,
            "reward_pips": self.reward_pips,
            "rr_ratio": round(self.rr_ratio, 2) if self.rr_ratio else None,
            "trend_ltf": self.trend_ltf,
            "trend_htf": self.trend_htf,
            "htf_alignment": self.htf_alignment,
            "at_key_level": self.at_key_level,
            "liquidity_taken": self.liquidity_taken,
            "confirmation": self.confirmation,
            "regime": self.regime,
            "candle_quality_score": round(self.candle_quality_score, 2)
                if self.candle_quality_score is not None else None,
            "candle_quality_label": self.candle_quality_label,
            "confluence_bonuses": self.confluence_bonuses,
            "confluence_penalties": self.confluence_penalties,
            "confluence_modules_run": len(self.confluence_results),
            "symbol": self.symbol,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


# ----------------------------------------------------------------------
# The Gate
# ----------------------------------------------------------------------
class InstitutionalEntryGate:
    """
    The institutional "should I trade now?" brain.

    A gate, not a signal generator. The signal comes from your strategy
    layer. This gate decides whether the signal is good enough to execute.
    """

    # Score weights — sum to 100 (v5.5 revised per institutional framework)
    WEIGHTS = {
        "trend": 15.0,          # was 20
        "htf_alignment": 15.0,  # NEW — separate from trend
        "key_level": 20.0,      # was 20 (unchanged)
        "liquidity": 15.0,      # was 20
        "confirmation": 15.0,   # NEW — separated from volume
        "volume": 10.0,         # was 15
        "session": 5.0,         # NEW
        "news": 5.0,            # NEW
    }

    # Score thresholds (v5.5 — stricter)
    THRESHOLD_A_PLUS = 90.0     # A+ setup → 1.5x position size
    THRESHOLD_GOOD = 80.0       # Good setup → 1.0x position size
    THRESHOLD_ACCEPTABLE = 70.0 # Acceptable → 0.5x position size
    THRESHOLD_WEAK = 55.0       # Below this → SKIP (bad quality)

    # 10/10 institutional checklist
    CHECKLIST_ITEMS = [
        "trend_aligned",
        "htf_bias_clear",
        "price_at_key_level",
        "liquidity_taken",
        "structure_confirmed",
        "retest_complete",
        "volume_supports",
        "spread_acceptable",
        "rr_acceptable",
        "risk_within_limits",
    ]

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        ie_cfg = config.get("institutional_entry", {}) if isinstance(config, dict) else {}

        # No-trade filters
        self.min_rr = float(ie_cfg.get("min_rr", 2.0))
        self.strong_rr = float(ie_cfg.get("strong_rr", 3.0))
        self.spread_max_multiple = float(ie_cfg.get("spread_max_multiple", 2.0))
        self.atr_low_multiple = float(ie_cfg.get("atr_low_multiple", 0.3))
        self.atr_high_multiple = float(ie_cfg.get("atr_high_multiple", 3.0))
        self.news_blackout_minutes = int(ie_cfg.get("news_blackout_minutes", 15))
        self.range_middle_skip_pct = float(ie_cfg.get("range_middle_skip_pct", 0.30))
        # ^ if price within middle 30% of recent range → skip

        # Sessions considered low-liquidity for crypto
        low_liq = ie_cfg.get("low_liquidity_sessions", ["sydney", "off"])
        self.low_liquidity_sessions = set(low_liq) if low_liq else set()

        # ATR / EMA params
        self.atr_period = int(ie_cfg.get("atr_period", 14))
        self.ema_fast = int(ie_cfg.get("ema_fast", 20))
        self.ema_slow = int(ie_cfg.get("ema_slow", 50))
        self.adx_period = int(ie_cfg.get("adx_period", 14))
        self.adx_trend_threshold = float(ie_cfg.get("adx_trend_threshold", 18.0))
        self.volume_period = int(ie_cfg.get("volume_period", 20))
        self.volume_min_ratio = float(ie_cfg.get("volume_min_ratio", 1.5))

        # Key-level proximity (in ATR multiples)
        self.key_level_atr_multiple = float(ie_cfg.get("key_level_atr_multiple", 0.5))

        # Lookback windows for swing detection
        self.swing_lookback = int(ie_cfg.get("swing_lookback", 20))
        self.range_lookback = int(ie_cfg.get("range_lookback", 50))

        # v5.5 helper modules — lazy imports for graceful degradation.
        # Critical #1 fix: each module is imported individually so a
        # single missing/broken module doesn't prevent the gate from
        # functioning with the remaining modules.
        self.regime_detector = None
        self.candle_analyzer = None
        self.volume_profile_analyzer = None
        self.vwap_analyzer = None
        self.fibonacci_analyzer = None
        self.wyckoff_analyzer = None
        self.fake_breakout_detector = None
        self.ema_ribbon_analyzer = None
        self.cross_asset_checker = None
        self.amt_analyzer = None
        self.chart_pattern_detector = None
        self.liquidation_heatmap = None
        self.cme_gap_detector = None
        self.anomaly_detector = None
        self.change_point_detector = None

        n_modules = 0

        try:
            from .market_regime import MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector(
                adx_period=int(ie_cfg.get("adx_period", 14)),
                adx_trend_threshold=float(ie_cfg.get("adx_trend_threshold", 25.0)),
                atr_period=int(ie_cfg.get("atr_period", 14)),
                ema_fast=self.ema_fast,
                ema_slow=self.ema_slow,
            )
            n_modules += 1
        except Exception as e:
            log.warning("InstitutionalEntryGate: market_regime unavailable: %r", e)

        try:
            from .candle_quality import CandleQualityAnalyzer
            self.candle_analyzer = CandleQualityAnalyzer(atr_period=self.atr_period)
            n_modules += 1
        except Exception as e:
            log.warning("InstitutionalEntryGate: candle_quality unavailable: %r", e)

        # v6.1: Instantiate all confluence contributors (lazy)
        for module_path, class_name, init_args in [
            (".volume_profile", "VolumeProfileAnalyzer", {"num_bins": int(ie_cfg.get("num_bins", 50)), "min_rows": 20}),
            (".vwap", "VWAPAnalyzer", {}),
            (".fibonacci", "FibonacciAnalyzer", {"swing_window": self.swing_lookback, "atr_period": self.atr_period}),
            (".wyckoff", "WyckoffAnalyzer", {"atr_period": self.atr_period}),
            ("engine.candlestick.false_breakout", "FalseBreakoutDetector", {"atr_period": self.atr_period}),
            (".ema_ribbon", "EMARibbonAnalyzer", {}),
            (".cross_asset", "CrossAssetConfirmation", {}),
            (".auction_market_theory", "AMTAnalyzer", {"atr_period": self.atr_period}),
            (".chart_patterns", "ChartPatternDetector", {"swing_window": 5, "atr_period": self.atr_period}),
            (".liquidation_heatmap", "LiquidationHeatmap", {"atr_period": self.atr_period}),
            (".cme_gap", "CMEGapDetector", {}),
            (".anomaly_detection", "AnomalyDetector", {}),
            (".change_point_detection", "ChangePointDetector", {}),
        ]:
            attr_name = {
                ".volume_profile": "volume_profile_analyzer",
                ".vwap": "vwap_analyzer",
                ".fibonacci": "fibonacci_analyzer",
                ".wyckoff": "wyckoff_analyzer",
                "engine.candlestick.false_breakout": "fake_breakout_detector",
                ".ema_ribbon": "ema_ribbon_analyzer",
                ".cross_asset": "cross_asset_checker",
                ".auction_market_theory": "amt_analyzer",
                ".chart_patterns": "chart_pattern_detector",
                ".liquidation_heatmap": "liquidation_heatmap",
                ".cme_gap": "cme_gap_detector",
                ".anomaly_detection": "anomaly_detector",
                ".change_point_detection": "change_point_detector",
            }[module_path]
            try:
                if module_path.startswith("."):
                    mod = __import__(f"trading_modules{module_path}", fromlist=[class_name])
                else:
                    mod = __import__(module_path, fromlist=[class_name])
                cls = getattr(mod, class_name)
                setattr(self, attr_name, cls(**init_args))
                n_modules += 1
            except Exception as e:
                log.warning("InstitutionalEntryGate: %s unavailable: %r", module_path, e)

        log.info("InstitutionalEntryGate v6.1 initialized "
                 "(min_rr=%.1f, A+≥%.0f, good≥%.0f, accept≥%.0f, %d/%d confluence modules wired)",
                 self.min_rr,
                 self.THRESHOLD_A_PLUS, self.THRESHOLD_GOOD,
                 self.THRESHOLD_ACCEPTABLE, n_modules, 15)

    # ==================================================================
    # PUBLIC ENTRY POINT
    # ==================================================================
    def evaluate(self, inp: EntryInput) -> EntryDecision:
        """Run the full institutional entry evaluation (v5.5)."""
        ts = datetime.now(timezone.utc).isoformat()
        direction = inp.direction.upper()
        if direction not in ("BUY", "SELL"):
            return self._reject(inp, "Invalid direction", ts)

        # ---- Step 1: hard no-trade filters (instant SKIP) ----
        skip = self._check_no_trade_filters(inp)
        if skip:
            return self._reject(inp, skip, ts)

        # ---- Step 1.5: Market Regime detection (v5.5) ----
        regime_result = self.regime_detector.detect(inp.df_m15, inp.df_h1)
        if not regime_allows_trading(regime_result.regime):
            return self._reject(
                inp,
                f"Regime {regime_result.regime.value} — no strategies work here "
                f"({regime_result.description})",
                ts,
            )

        # ---- Step 2: trend detection (LTF + HTF) ----
        ltf_df = inp.df_m15 if (inp.df_m15 is not None and not inp.df_m15.empty) else inp.df_h1
        htf_df = inp.df_h4 if (inp.df_h4 is not None and not inp.df_h4.empty) else inp.df_d1
        trend_ltf = self._detect_trend(ltf_df)
        trend_htf = self._detect_trend(htf_df)
        # v5.5: also detect H4 and D1 trends for true MTF
        trend_h4 = self._detect_trend(inp.df_h4) if inp.df_h4 is not None else trend_htf
        trend_d1 = self._detect_trend(inp.df_d1) if inp.df_d1 is not None else trend_htf
        # HTF alignment: all of D1, H4, H1 must agree with M15
        htf_alignment = (
            trend_ltf in ("bullish", "bearish") and
            trend_ltf == trend_h4 == trend_d1
        )

        # ---- Step 3: R:R calculation ----
        rr = self._calculate_rr(inp)
        if rr is None or rr < self.min_rr:
            rr_str = f"{rr:.2f}" if rr is not None else "N/A"
            return self._reject(inp, f"R:R {rr_str} < {self.min_rr}", ts)

        # ---- Step 4: key level proximity ----
        at_key_level, key_level_type = self._check_key_level(inp)

        # ---- Step 5: liquidity analysis ----
        liquidity_taken, liquidity_type = self._analyze_liquidity(inp)

        # ---- Step 6: confirmation (BOS/CHoCH/Engulfing/Retest/Volume) ----
        confirmation = self._detect_confirmation(inp)

        # ---- Step 6.5: Candle quality (v5.5) ----
        candle_quality = self.candle_analyzer.analyze(inp.df_m15, direction=direction)

        # ---- Step 7: volume + momentum ----
        vol_ratio = self._volume_ratio(inp.df_m15)
        momentum_score = self._momentum_score(inp.df_m15, direction)

        # ---- Step 8: scoring (v5.5 revised weights) ----
        breakdown = self._score_v55(
            trend_ltf=trend_ltf,
            trend_htf=trend_htf,
            htf_alignment=htf_alignment,
            at_key_level=at_key_level,
            liquidity_taken=liquidity_taken,
            confirmation=confirmation,
            vol_ratio=vol_ratio,
            candle_quality_score=candle_quality.score,
            session=inp.session,
            news_in_minutes=inp.news_in_minutes,
        )
        score = sum(breakdown.values())

        # ---- Step 8.5: Run all 13 confluence modules (v6.1) ----
        confluence_results, conf_bonuses, conf_penalties = self._run_confluence_modules(inp)
        # Apply confluence adjustments to the score (capped)
        total_bonus = sum(b[1] for b in conf_bonuses)
        total_penalty = sum(p[1] for p in conf_penalties)
        # Cap adjustments at ±15 points so confluence can shift grade but not dominate
        confluence_adjustment = max(-15.0, min(15.0, total_bonus + total_penalty))
        score = max(0.0, min(100.0, score + confluence_adjustment))
        # Build bonus/penalty dicts for transparency
        bonuses_dict = {b[0]: b[1] for b in conf_bonuses}
        penalties_dict = {p[0]: p[1] for p in conf_penalties}

        # ---- Step 9: 10/10 checklist ----
        checklist = self._build_checklist(
            inp=inp,
            trend_ltf=trend_ltf,
            trend_htf=trend_htf,
            at_key_level=at_key_level,
            liquidity_taken=liquidity_taken,
            confirmation=confirmation,
            vol_ratio=vol_ratio,
            rr=rr,
        )
        # Add candle quality to checklist (replaces retest_complete with stronger check)
        checklist["candle_quality_ok"] = candle_quality.score >= 0.4
        # v6.1: Add confluence-based checklist items
        checklist["confluence_positive"] = len(conf_bonuses) >= 2
        checklist["no_confluence_contradiction"] = len(conf_penalties) <= 2
        failed = [k for k, v in checklist.items() if not v]

        # ---- Step 9.5: Confidence estimation (v5.5) ----
        confidence_pct, win_probability = self._estimate_confidence(
            score=score,
            rr=rr,
            candle_quality_score=candle_quality.score,
            htf_alignment=htf_alignment,
            regime_confidence=regime_result.confidence,
        )

        # ---- Step 10: final decision (v5.5 — 4-decision system) ----
        action, grade, skip_reason = self._decide_v55(
            score=score,
            checklist_failed=failed,
            trend_ltf=trend_ltf,
            trend_htf=trend_htf,
            direction=direction,
            at_key_level=at_key_level,
            candle_quality_score=candle_quality.score,
        )

        notes = []
        if key_level_type:
            notes.append(f"key_level={key_level_type}")
        if liquidity_type:
            notes.append(f"liquidity={liquidity_type}")
        if confirmation:
            notes.append(f"confirmation={confirmation}")
        notes.append(f"vol_ratio={vol_ratio:.2f}")
        notes.append(f"momentum={momentum_score:.2f}")
        notes.append(f"candle_q={candle_quality.label}({candle_quality.score:.2f})")
        notes.append(f"regime={regime_result.regime.value}({regime_result.confidence:.2f})")
        notes.append(f"htf_align={htf_alignment}")
        # v6.1: Add confluence summary to notes
        notes.append(f"confluence: +{total_bonus:.0f}/{total_penalty:.0f} ({len(conf_bonuses)}B/{len(conf_penalties)}P)")

        return EntryDecision(
            action=action,
            score=score,
            grade=grade,
            checklist=checklist,
            failed_checks=failed,
            skip_reason=skip_reason,
            score_breakdown=breakdown,
            risk_pips=abs(inp.entry_price - inp.stop_loss),
            reward_pips=abs(inp.take_profit - inp.entry_price),
            rr_ratio=rr,
            trend_ltf=trend_ltf,
            trend_htf=trend_htf,
            at_key_level=at_key_level,
            liquidity_taken=liquidity_taken,
            confirmation=confirmation,
            confidence_pct=confidence_pct,
            win_probability=win_probability,
            regime=regime_result.regime.value,
            candle_quality_score=candle_quality.score,
            candle_quality_label=candle_quality.label,
            htf_alignment=htf_alignment,
            confluence_results=confluence_results,
            confluence_bonuses=bonuses_dict,
            confluence_penalties=penalties_dict,
            symbol=inp.symbol,
            direction=direction,
            timestamp=ts,
            notes=notes,
        )

    # ==================================================================
    # STEP 1 — No-trade filters
    # ==================================================================
    def _check_no_trade_filters(self, inp: EntryInput) -> Optional[str]:
        """Return skip reason if any hard filter triggers, else None."""
        # News blackout
        if inp.news_in_minutes < self.news_blackout_minutes:
            return f"News in {inp.news_in_minutes}min < blackout {self.news_blackout_minutes}min"

        # Low-liquidity session
        if inp.session in self.low_liquidity_sessions:
            return f"Low-liquidity session: {inp.session}"

        # ATR extremes (need m15 data)
        if inp.df_m15 is not None and len(inp.df_m15) > self.atr_period + 5:
            atr_now = self._atr(inp.df_m15, self.atr_period).iloc[-1]
            atr_baseline = self._atr(inp.df_m15, self.atr_period).rolling(50).mean().iloc[-1]
            if atr_baseline and atr_baseline > 0:
                ratio = atr_now / atr_baseline
                if ratio < self.atr_low_multiple:
                    return f"ATR too low ({ratio:.2f}x baseline) — dead market"
                if ratio > self.atr_high_multiple:
                    return f"ATR too high ({ratio:.2f}x baseline) — chaotic market"

        # Spread (caller must pass spread_points; baseline inferred from symbol)
        # We skip spread filter here if spread_points==0 (caller didn't supply)
        # — actual spread check is in the checklist

        # Range middle skip — ONLY if market is not trending
        if inp.df_m15 is not None and len(inp.df_m15) > self.range_lookback:
            # First check if there's a trend — if yes, don't apply middle-of-range skip
            trend_check = self._detect_trend(inp.df_m15)
            recent = inp.df_m15.tail(self.range_lookback)
            hi = recent["high"].max()
            lo = recent["low"].min()
            rng = hi - lo
            if rng > 0 and trend_check == "ranging":
                pos = (inp.entry_price - lo) / rng  # 0..1
                middle_start = self.range_middle_skip_pct
                middle_end = 1 - self.range_middle_skip_pct
                if middle_start < pos < middle_end:
                    # Only skip if no key level proximity (which would override)
                    # — caller's key_zone flag overrides
                    if inp.at_key_zone is False or inp.at_key_zone is None:
                        return (f"Price in middle of range "
                                f"(pos={pos:.2f}, skip zone {middle_start}-{middle_end})")

        # R:R already enforced in step 3 — but reject early here too
        rr = self._calculate_rr(inp)
        if rr is None:
            return "Cannot calculate R:R (entry/SL/TP missing)"
        if rr < self.min_rr:
            return f"R:R {rr:.2f} < minimum {self.min_rr}"

        return None

    # ==================================================================
    # STEP 2 — Trend detection
    # ==================================================================
    def _detect_trend(self, df: Optional[pd.DataFrame]) -> str:
        """Return 'bullish', 'bearish', or 'ranging'.
        
        Uses EMA stack + EMA slope + ADX confirmation. A trend is only
        confirmed when EMAs are stacked correctly AND slope is non-zero
        AND ADX (if computable) is above threshold.
        """
        if df is None or df is None or len(df) < self.ema_slow + 10:
            return "unknown"
        close = df["close"]
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        
        last_close = close.iloc[-1]
        last_ema_f = ema_f.iloc[-1]
        last_ema_s = ema_s.iloc[-1]
        
        # EMA slope over last 10 bars (in %)
        slope = (ema_s.iloc[-1] - ema_s.iloc[-10]) / ema_s.iloc[-10] if ema_s.iloc[-10] != 0 else 0
        slope_pct = slope * 100
        
        # ADX (best-effort)
        try:
            adx = self._adx(df, self.adx_period)
            last_adx = float(adx.iloc[-1]) if not adx.empty and not np.isnan(adx.iloc[-1]) else 0
        except Exception:
            last_adx = 25.0  # neutral default
        
        # Bullish: EMA fast > slow, close above both, positive slope
        if (last_ema_f > last_ema_s and 
            last_close > last_ema_f and 
            slope_pct > 0.05):  # at least 0.05% rise over 10 bars
            if last_adx >= self.adx_trend_threshold:
                return "bullish"
            else:
                return "bullish"  # accept even weak ADX if EMA stack is clean
        # Bearish: EMA fast < slow, close below both, negative slope
        if (last_ema_f < last_ema_s and 
            last_close < last_ema_f and 
            slope_pct < -0.05):
            if last_adx >= self.adx_trend_threshold:
                return "bearish"
            else:
                return "bearish"
        return "ranging"

    # ==================================================================
    # STEP 3 — R:R
    # ==================================================================
    def _calculate_rr(self, inp: EntryInput) -> Optional[float]:
        risk = abs(inp.entry_price - inp.stop_loss)
        reward = abs(inp.take_profit - inp.entry_price)
        if risk <= 0:
            return None
        return reward / risk

    # ==================================================================
    # STEP 4 — Key level proximity
    # ==================================================================
    def _check_key_level(self, inp: EntryInput) -> tuple[bool, Optional[str]]:
        """Check if price is near a key institutional level."""
        # Caller override
        if inp.at_key_zone is True:
            return True, "caller_flagged"

        df = inp.df_m15
        if df is None or len(df) < self.swing_lookback:
            return False, None

        atr = self._atr(df, self.atr_period).iloc[-1]
        if atr <= 0:
            return False, None
        threshold = atr * self.key_level_atr_multiple
        price = inp.entry_price

        # Recent swing highs / lows
        recent = df.tail(self.range_lookback)
        swing_highs = self._swing_highs(recent["high"].values, self.swing_lookback // 2)
        swing_lows = self._swing_lows(recent["low"].values, self.swing_lookback // 2)

        for sh in swing_highs:
            if abs(price - sh) <= threshold:
                return True, "swing_high"
        for sl in swing_lows:
            if abs(price - sl) <= threshold:
                return True, "swing_low"

        # Previous day high / low
        if inp.df_d1 is not None and len(inp.df_d1) >= 2:
            pdh = inp.df_d1["high"].iloc[-2]
            pdl = inp.df_d1["low"].iloc[-2]
            if abs(price - pdh) <= threshold:
                return True, "PDH"
            if abs(price - pdl) <= threshold:
                return True, "PDL"

        # Weekly high / low (use D1 last 7 bars)
        if inp.df_d1 is not None and len(inp.df_d1) >= 7:
            week = inp.df_d1.tail(7)
            whl = week["high"].max()
            wll = week["low"].min()
            if abs(price - whl) <= threshold:
                return True, "WHL"
            if abs(price - wll) <= threshold:
                return True, "WLL"

        return False, None

    # ==================================================================
    # STEP 5 — Liquidity analysis
    # ==================================================================
    def _analyze_liquidity(self, inp: EntryInput) -> tuple[bool, Optional[str]]:
        """Detect liquidity sweep / grab / equal highs-lows."""
        if inp.liquidity_sweep is True:
            return True, "caller_flagged"

        df = inp.df_m15
        if df is None or len(df) < 30:
            return False, None

        recent = df.tail(30).reset_index(drop=True)
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values
        atr = self._atr(df, self.atr_period).iloc[-1]
        if atr <= 0:
            return False, None

        # Equal highs / lows (within 0.1 ATR)
        eq_high_threshold = atr * 0.1
        eq_low_threshold = atr * 0.1

        # Find pairs of similar highs
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) <= eq_high_threshold:
                    # Liquidity grab: last candle pierced high but closed below
                    if j == len(highs) - 1 and closes[-1] < highs[i]:
                        return True, "equal_highs_grab"
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) <= eq_low_threshold:
                    if j == len(lows) - 1 and closes[-1] > lows[i]:
                        return True, "equal_lows_grab"

        # Stop hunt: last candle wick beyond prior swing but close back inside
        prior_swing_hi = max(highs[:-1]) if len(highs) > 1 else 0
        prior_swing_lo = min(lows[:-1]) if len(lows) > 1 else 0
        if highs[-1] > prior_swing_hi and closes[-1] < prior_swing_hi:
            return True, "stop_hunt_high"
        if lows[-1] < prior_swing_lo and closes[-1] > prior_swing_lo:
            return True, "stop_hunt_low"

        return False, None

    # ==================================================================
    # STEP 6 — Confirmation
    # ==================================================================
    def _detect_confirmation(self, inp: EntryInput) -> Optional[str]:
        """Detect BOS / CHoCH / Engulfing / Retest."""
        if inp.structure_break:
            return inp.structure_break

        df = inp.df_m15
        if df is None or len(df) < 10:
            return None

        recent = df.tail(5).reset_index(drop=True)
        # Bullish engulfing
        if (recent["open"].iloc[-2] > recent["close"].iloc[-2] and  # prior red
            recent["close"].iloc[-1] > recent["open"].iloc[-1] and  # current green
            recent["close"].iloc[-1] > recent["open"].iloc[-2] and
            recent["open"].iloc[-1] < recent["close"].iloc[-2]):
            return "BULLISH_ENGULFING"
        # Bearish engulfing
        if (recent["open"].iloc[-2] < recent["close"].iloc[-2] and
            recent["close"].iloc[-1] < recent["open"].iloc[-1] and
            recent["close"].iloc[-1] < recent["open"].iloc[-2] and
            recent["open"].iloc[-1] > recent["close"].iloc[-2]):
            return "BEARISH_ENGULFING"

        # Volume spike (last bar volume > 2x average of prior 10)
        if len(df) >= 12:
            v_now = df["volume"].iloc[-1]
            v_avg = df["volume"].iloc[-12:-1].mean()
            if v_avg > 0 and v_now > 2.0 * v_avg:
                return "VOLUME_SPIKE"

        return None

    # ==================================================================
    # STEP 7 — Volume + momentum
    # ==================================================================
    def _volume_ratio(self, df: Optional[pd.DataFrame]) -> float:
        if df is None or len(df) < self.volume_period + 1:
            return 1.0
        v_now = df["volume"].iloc[-1]
        v_avg = df["volume"].iloc[-self.volume_period - 1:-1].mean()
        if v_avg <= 0:
            return 1.0
        return float(v_now / v_avg)

    def _momentum_score(self, df: Optional[pd.DataFrame], direction: str) -> float:
        """Return momentum score 0..1 in the direction of the trade."""
        if df is None or len(df) < 14:
            return 0.5
        close = df["close"]
        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        if np.isnan(rsi):
            return 0.5
        if direction == "BUY":
            return float(rsi / 100.0)  # higher RSI = better for buy
        else:
            return float((100 - rsi) / 100.0)

    # ==================================================================
    # STEP 8 — Scoring
    # ==================================================================
    # ==================================================================
    # STEP 8 — v5.5 Scoring (revised weights per institutional framework)
    # ==================================================================
    # Major #5 fix: removed legacy _score() and _decide() methods (dead code).
    # The v55 versions are the only ones called by evaluate().
    def _score_v55(
        self,
        trend_ltf: str,
        trend_htf: str,
        htf_alignment: bool,
        at_key_level: bool,
        liquidity_taken: bool,
        confirmation: Optional[str],
        vol_ratio: float,
        candle_quality_score: float,
        session: str,
        news_in_minutes: int,
    ) -> dict[str, float]:
        """v5.5 scoring with 8 dimensions per user's revised framework."""
        w = self.WEIGHTS

        # Trend (15) — LTF direction strength
        if trend_ltf in ("bullish", "bearish"):
            trend_score = w["trend"]
        else:
            trend_score = 0.0

        # HTF alignment (15) — D1+H4+H1+M15 all agree
        if htf_alignment:
            htf_score = w["htf_alignment"]
        elif trend_ltf == trend_htf and trend_ltf in ("bullish", "bearish"):
            htf_score = w["htf_alignment"] * 0.5  # partial — only 2 TF agree
        else:
            htf_score = 0.0

        # Key level (20) — binary
        key_level_score = w["key_level"] if at_key_level else 0.0

        # Liquidity (15) — binary
        liquidity_score = w["liquidity"] if liquidity_taken else 0.0

        # Confirmation (15) — BOS / CHoCH / Engulfing / Volume spike
        confirmation_strong = confirmation in (
            "BOS", "CHoCH", "BULLISH_ENGULFING", "BEARISH_ENGULFING"
        )
        confirmation_partial = confirmation == "VOLUME_SPIKE"
        if confirmation_strong:
            confirmation_pts = w["confirmation"]
        elif confirmation_partial:
            confirmation_pts = w["confirmation"] * 0.6
        else:
            confirmation_pts = 0.0

        # Volume (10) — also incorporates candle quality as a multiplier
        if vol_ratio >= 2.0:
            volume_score = w["volume"]
        elif vol_ratio >= 1.5:
            volume_score = w["volume"] * 0.8
        elif vol_ratio >= 1.0:
            volume_score = w["volume"] * 0.4
        else:
            volume_score = 0.0
        # Scale volume by candle quality (poor candle = discount volume signal)
        volume_score *= max(0.3, candle_quality_score)

        # Session (5) — london/newyork/overlap = full; tokyo = half; sydney/off = 0
        if session in ("london", "newyork", "overlap"):
            session_score = w["session"]
        elif session == "tokyo":
            session_score = w["session"] * 0.5
        else:
            session_score = 0.0

        # News (5) — full marks if no news within 60 min, partial if 15-60 min, 0 if < 15
        if news_in_minutes >= 60:
            news_score = w["news"]
        elif news_in_minutes >= self.news_blackout_minutes:
            news_score = w["news"] * 0.5
        else:
            news_score = 0.0

        return {
            "trend": trend_score,
            "htf_alignment": htf_score,
            "key_level": key_level_score,
            "liquidity": liquidity_score,
            "confirmation": confirmation_pts,
            "volume": volume_score,
            "session": session_score,
            "news": news_score,
        }

    # ==================================================================
    # STEP 9.5 — Confidence estimation (v5.5)
    # ==================================================================
    def _estimate_confidence(
        self,
        score: float,
        rr: float,
        candle_quality_score: float,
        htf_alignment: bool,
        regime_confidence: float,
    ) -> tuple[float, float]:
        """Estimate overall confidence (0..100) and win probability (0..1).

        Confidence = blend of:
          - gate score (0..100) → 50% weight
          - candle quality (0..1 → 0..100) → 20% weight
          - HTF alignment (0/1 → 0/100) → 15% weight
          - regime confidence (0..1 → 0..100) → 15% weight

        Win probability = sigmoid transform of (score/100 + rr_bonus)
        """
        score_pct = max(0.0, min(100.0, score))
        candle_pct = max(0.0, min(100.0, candle_quality_score * 100.0))
        htf_pct = 100.0 if htf_alignment else 30.0
        regime_pct = max(0.0, min(100.0, regime_confidence * 100.0))

        confidence_pct = (
            0.50 * score_pct +
            0.20 * candle_pct +
            0.15 * htf_pct +
            0.15 * regime_pct
        )

        # Win probability — base from score, bonus from R:R
        # Empirically: high-score setups with high R:R have ~60-65% win prob
        # Use a logistic curve: win_prob = 1 / (1 + exp(-(score/30 + rr/2 - 2)))
        raw = (score / 30.0) + (rr / 2.0) - 2.0
        win_prob = 1.0 / (1.0 + pow(2.71828, -raw))
        # Cap between 0.35 and 0.75 — we are NEVER more than 75% confident
        win_prob = max(0.35, min(0.75, win_prob))

        return float(confidence_pct), float(win_prob)

    # ==================================================================
    # STEP 8.5 — Run all confluence modules (v6.1)
    # ==================================================================
    def _run_confluence_modules(self, inp: EntryInput) -> tuple[dict, list, list]:
        """Run all 13 confluence contributor modules.

        Returns:
            (results_dict, bonus_signals, penalty_signals)
            - results_dict: {module_name: result.to_dict() or dict}
            - bonus_signals: list of (name, value, note) tuples (positive score adjustments)
            - penalty_signals: list of (name, value, note) tuples (negative adjustments)
        """
        results: dict = {}
        bonuses: list[tuple[str, float, str]] = []
        penalties: list[tuple[str, float, str]] = []
        df = inp.df_m15
        direction = inp.direction.upper()
        price = inp.entry_price

        # 1. Volume Profile
        try:
            vp = self.volume_profile_analyzer.analyze(df)
            results["volume_profile"] = vp.to_dict()
            if vp.rejection_from_hvn:
                bonuses.append(("vp_rejection_hvn", 3.0,
                                f"rejected from HVN @ {vp.poc:.2f}" if vp.poc else "HVN rejection"))
            if vp.fast_move_through_lvn:
                bonuses.append(("vp_fast_lvn", 2.0, "fast move through LVN — momentum"))
            if vp.price_in_value_area:
                bonuses.append(("vp_in_value_area", 1.0, "price in value area — accepted"))
        except Exception as e:
            log.debug("VP failed: %s", e)

        # 2. VWAP
        try:
            vwap_r = self.vwap_analyzer.analyze(df)
            results["vwap"] = vwap_r.to_dict()
            if direction == "BUY" and vwap_r.bullish_rejection:
                bonuses.append(("vwap_bull_rejection", 3.0, "bullish VWAP rejection"))
            elif direction == "SELL" and vwap_r.bearish_rejection:
                bonuses.append(("vwap_bear_rejection", 3.0, "bearish VWAP rejection"))
            # Distance from VWAP — too far = mean reversion risk
            if not np.isnan(vwap_r.distance_atr) and abs(vwap_r.distance_atr) > 3.0:
                penalties.append(("vwap_extended", -2.0,
                                  f"price {vwap_r.distance_atr:.1f} ATR from VWAP — extended"))
        except Exception as e:
            log.debug("VWAP failed: %s", e)

        # 3. Fibonacci
        try:
            fib = self.fibonacci_analyzer.analyze(df, current_price=price, direction=direction)
            results["fibonacci"] = fib.to_dict()
            if fib.at_nearest_level:
                bonuses.append(("fib_confluence", 3.0,
                                f"at Fib {fib.nearest_level_name} @ {fib.nearest_level_price:.2f}"))
            if fib.confluence_cluster:
                bonuses.append(("fib_cluster", 2.0,
                                f"cluster of {fib.confluence_cluster['cluster_size']} levels"))
        except Exception as e:
            log.debug("Fib failed: %s", e)

        # 4. Wyckoff
        try:
            wy = self.wyckoff_analyzer.analyze(df, direction=direction)
            results["wyckoff"] = wy.to_dict()
            if direction == "BUY" and wy.spring_detected:
                bonuses.append(("wyckoff_spring", 4.0, "Wyckoff spring — bullish reversal"))
            if direction == "BUY" and wy.sos_detected:
                bonuses.append(("wyckoff_sos", 3.0, "Sign of Strength"))
            if direction == "SELL" and wy.upthrust_detected:
                bonuses.append(("wyckoff_upthrust", 4.0, "Wyckoff upthrust — bearish reversal"))
            if direction == "SELL" and wy.sow_detected:
                bonuses.append(("wyckoff_sow", 3.0, "Sign of Weakness"))
        except Exception as e:
            log.debug("Wyckoff failed: %s", e)

        # 5. Fake Breakout (test recent swing high/low)
        try:
            recent = df.tail(self.range_lookback) if df is not None else None
            if recent is not None and not recent.empty:
                level = float(recent["high"].max()) if direction == "BUY" else float(recent["low"].min())
                fb = self.fake_breakout_detector.analyze(df, level=level, direction=direction)
                results["fake_breakout"] = fb.to_dict()
                if fb.is_real:
                    bonuses.append(("real_breakout", 3.0, f"confirmed real breakout of {level:.2f}"))
                if fb.is_fake:
                    penalties.append(("fake_breakout", -4.0, "fake breakout detected — trap"))
        except Exception as e:
            log.debug("FakeBreakout failed: %s", e)

        # 6. EMA Ribbon
        try:
            er = self.ema_ribbon_analyzer.analyze(df)
            results["ema_ribbon"] = er.to_dict()
            if direction == "BUY" and er.fully_stacked_bull:
                bonuses.append(("ema_bull_stack", 2.0, "fully stacked bullish ribbon"))
            if direction == "SELL" and er.fully_stacked_bear:
                bonuses.append(("ema_bear_stack", 2.0, "fully stacked bearish ribbon"))
            if er.compression_ratio >= 0.9:
                bonuses.append(("ema_compression", 1.0, "ribbon compressed — breakout pending"))
        except Exception as e:
            log.debug("EMARibbon failed: %s", e)

        # 7. Cross-Asset Confirmation
        try:
            if inp.related_dfs:
                ca = self.cross_asset_checker.check(
                    candidate_symbol=inp.symbol,
                    candidate_direction=direction,
                    related_dfs=inp.related_dfs,
                )
                results["cross_asset"] = ca.to_dict()
                if ca.confirmed:
                    bonuses.append(("cross_asset_confirmed", 3.0,
                                    f"confirmed by {len(ca.confirmations)} assets"))
                if ca.contradicted:
                    penalties.append(("cross_asset_contradicted", -3.0,
                                      f"contradicted by {len(ca.contradictions)} assets"))
        except Exception as e:
            log.debug("CrossAsset failed: %s", e)

        # 8. AMT (Auction Market Theory)
        try:
            amt = self.amt_analyzer.analyze(df)
            results["amt"] = amt.to_dict()
            if direction == "BUY" and amt.at_va_low_rejection:
                bonuses.append(("amt_va_low_rejection", 3.0, "rejected at VAL — bullish"))
            if direction == "SELL" and amt.at_va_high_rejection:
                bonuses.append(("amt_va_high_rejection", 3.0, "rejected at VAH — bearish"))
            if amt.excess_high and direction == "SELL":
                bonuses.append(("amt_excess_high", 2.0, "excess high — bearish"))
            if amt.excess_low and direction == "BUY":
                bonuses.append(("amt_excess_low", 2.0, "excess low — bullish"))
        except Exception as e:
            log.debug("AMT failed: %s", e)

        # 9. Chart Patterns
        try:
            patterns = self.chart_pattern_detector.detect_all(df)
            results["chart_patterns"] = patterns.to_dict()
            for p in patterns.patterns:
                if not p.detected:
                    continue
                if p.direction == direction.lower():
                    bonuses.append((f"pattern_{p.pattern_type}", 3.0,
                                    f"{p.pattern_type} detected (conf={p.confidence:.2f})"))
                elif p.direction != "neutral":
                    penalties.append((f"pattern_{p.pattern_type}", -3.0,
                                      f"{p.pattern_type} detected against direction"))
        except Exception as e:
            log.debug("ChartPatterns failed: %s", e)

        # 10. Liquidation Heatmap
        try:
            lh = self.liquidation_heatmap.analyze(df, current_price=price)
            results["liquidation_heatmap"] = lh.to_dict()
            if direction == "BUY" and lh.magnet_below:
                dist_pct = abs(price - lh.magnet_below) / price * 100
                if dist_pct < 1.0:
                    bonuses.append(("liq_magnet_below", 2.0,
                                    f"long liq magnet ${lh.magnet_below:.0f} ({dist_pct:.1f}%) below"))
            if direction == "SELL" and lh.magnet_above:
                dist_pct = abs(lh.magnet_above - price) / price * 100
                if dist_pct < 1.0:
                    bonuses.append(("liq_magnet_above", 2.0,
                                    f"short liq magnet ${lh.magnet_above:.0f} ({dist_pct:.1f}%) above"))
            if lh.cascade_risk == "high":
                penalties.append(("liq_cascade_risk", -2.0, "high cascade risk"))
        except Exception as e:
            log.debug("LiquidationHeatmap failed: %s", e)

        # 11. CME Gap (uses D1 if available)
        try:
            if inp.df_d1 is not None and not inp.df_d1.empty:
                cme = self.cme_gap_detector.analyze(inp.df_d1)
                results["cme_gap"] = cme.to_dict()
                if cme.nearest_open_gap:
                    g = cme.nearest_open_gap
                    gap_mid = (g["gap_low"] + g["gap_high"]) / 2
                    dist_pct = abs(price - gap_mid) / price * 100
                    if dist_pct < 2.0:
                        bonuses.append(("cme_gap_magnet", 2.0,
                                        f"CME gap {g['direction']} {dist_pct:.1f}% away — magnet"))
        except Exception as e:
            log.debug("CMEGap failed: %s", e)

        # 12. Anomaly Detection
        try:
            anom = self.anomaly_detector.detect(df["close"], method="auto")
            results["anomaly"] = anom.to_dict()
            # If current bar is anomalous, that's a warning
            if len(df) - 1 in anom.anomaly_indices:
                penalties.append(("anomaly_current", -2.0, "current bar is anomalous"))
        except Exception as e:
            log.debug("Anomaly failed: %s", e)

        # 13. Change Point Detection
        try:
            cps = self.change_point_detector.detect_all(df["close"])
            results["change_points"] = {
                "n_change_points": len(cps),
                "recent_cp": cps[-1].to_dict() if cps else None,
            }
            # If a change point was detected in the last 5 bars, regime may be shifting
            if cps and cps[-1].index >= len(df) - 5:
                penalties.append(("recent_change_point", -2.0,
                                  f"change point at idx {cps[-1].index}: {cps[-1].type}"))
        except Exception as e:
            log.debug("ChangePoint failed: %s", e)

        return results, bonuses, penalties

    # ==================================================================
    # STEP 9 — 10/10 Checklist
    # ==================================================================
    def _build_checklist(
        self,
        inp: EntryInput,
        trend_ltf: str,
        trend_htf: str,
        at_key_level: bool,
        liquidity_taken: bool,
        confirmation: Optional[str],
        vol_ratio: float,
        rr: Optional[float],
    ) -> dict[str, bool]:
        direction = inp.direction.upper()

        # 1. trend_aligned
        trend_aligned = (
            trend_ltf in ("bullish", "bearish") and
            trend_ltf == trend_htf
        )

        # 2. htf_bias_clear
        htf_bias_clear = trend_htf in ("bullish", "bearish")

        # 3. price_at_key_level
        price_at_key_level = at_key_level or inp.at_key_zone is True

        # 4. liquidity_taken
        liq_ok = liquidity_taken or inp.liquidity_sweep is True

        # 5. structure_confirmed
        structure_ok = confirmation in (
            "BOS", "CHoCH", "BULLISH_ENGULFING", "BEARISH_ENGULFING",
            "VOLUME_SPIKE", "RETEST"
        ) or inp.structure_break in ("BOS", "CHoCH")

        # 6. retest_complete (heuristic: structure break + price returned to break level)
        retest_complete = confirmation == "RETEST" or inp.structure_break == "BOS"

        # 7. volume_supports
        volume_supports = vol_ratio >= self.volume_min_ratio

        # 8. spread_acceptable (caller-provided spread)
        # We treat spread_points=0 as "unknown / not provided" → pass
        spread_acceptable = inp.spread_points == 0 or inp.spread_points <= 50  # 50 points = generic cap

        # 9. rr_acceptable
        rr_acceptable = rr is not None and rr >= self.min_rr

        # 10. risk_within_limits (basic — caller's responsibility for Kelly)
        # We approximate: risk_pips should not be > 2x ATR
        risk_within = True
        if inp.df_m15 is not None and len(inp.df_m15) > self.atr_period + 1:
            atr = self._atr(inp.df_m15, self.atr_period).iloc[-1]
            risk_pips = abs(inp.entry_price - inp.stop_loss)
            if atr > 0 and risk_pips > 3.0 * atr:
                risk_within = False

        return {
            "trend_aligned": bool(trend_aligned),
            "htf_bias_clear": bool(htf_bias_clear),
            "price_at_key_level": bool(price_at_key_level),
            "liquidity_taken": bool(liq_ok),
            "structure_confirmed": bool(structure_ok),
            "retest_complete": bool(retest_complete),
            "volume_supports": bool(volume_supports),
            "spread_acceptable": bool(spread_acceptable),
            "rr_acceptable": bool(rr_acceptable),
            "risk_within_limits": bool(risk_within),
        }

    # ==================================================================
    # STEP 10 — v5.5 Final decision (4-decision system)
    # ==================================================================
    # Major #5 fix: removed legacy _decide() (dead code).
    def _decide_v55(
        self,
        score: float,
        checklist_failed: list[str],
        trend_ltf: str,
        trend_htf: str,
        direction: str,
        at_key_level: bool,
        candle_quality_score: float,
    ) -> tuple[str, str, Optional[str]]:
        """4-decision system: BUY / SELL / WAIT / SKIP.

        - BUY/SELL: score ≥ 70 AND no critical checklist failures
        - WAIT: score ≥ 55 AND < 70 — setup is forming, price approaching level
        - SKIP: score < 55 OR critical checklist failure — bad quality, abandon
        """
        # Grade from score
        if score >= self.THRESHOLD_A_PLUS:
            grade = "A+"
        elif score >= self.THRESHOLD_GOOD:
            grade = "A"
        elif score >= self.THRESHOLD_ACCEPTABLE:
            grade = "B"
        elif score >= self.THRESHOLD_WEAK:
            grade = "C"
        else:
            grade = "F"

        # Critical checklist failures → SKIP (don't trade this setup)
        critical_failures = {
            "trend_aligned", "htf_bias_clear", "structure_confirmed",
            "rr_acceptable", "candle_quality_ok",
        }
        for c in checklist_failed:
            if c in critical_failures:
                return (Action.SKIP.value, grade,
                        f"Critical checklist failed: {c}")

        # Hard SKIP conditions (regardless of score)
        if candle_quality_score < 0.3:
            return (Action.SKIP.value, grade,
                    f"Candle quality too low ({candle_quality_score:.2f})")

        # Decision based on score + checklist health
        if score >= self.THRESHOLD_A_PLUS and len(checklist_failed) == 0:
            return (direction, grade, None)        # A+ entry
        if score >= self.THRESHOLD_GOOD and len(checklist_failed) <= 1:
            return (direction, grade, None)        # Good entry
        if score >= self.THRESHOLD_ACCEPTABLE and len(checklist_failed) <= 2:
            return (direction, grade, None)        # Acceptable entry

        # WAIT zone — setup is forming but not yet complete
        if score >= self.THRESHOLD_WEAK:
            # Distinguish WAIT (forming) from SKIP (bad)
            if at_key_level and candle_quality_score >= 0.4:
                # Price is at the level + decent candle → setup is forming
                return (Action.WAIT.value, grade,
                        f"Setup forming — score {score:.1f}, wait for confirmation")
            else:
                return (Action.WAIT.value, grade,
                        f"Score {score:.1f} below acceptable — wait for better setup")

        # Below weak threshold → SKIP (bad quality)
        return (Action.SKIP.value, grade,
                f"Score {score:.1f} below weak threshold ({self.THRESHOLD_WEAK}) — skip")

    # ==================================================================
    # Helpers
    # ==================================================================
    def _reject(self, inp: EntryInput, reason: str, ts: str) -> EntryDecision:
        return EntryDecision(
            action=Action.SKIP.value,
            score=0.0,
            grade="F",
            checklist={k: False for k in self.CHECKLIST_ITEMS},
            failed_checks=list(self.CHECKLIST_ITEMS),
            skip_reason=reason,
            symbol=inp.symbol,
            direction=inp.direction,
            timestamp=ts,
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        """Wilder's ADX (simplified)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
        minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def _swing_highs(highs: np.ndarray, k: int) -> list[float]:
        """Find local peaks: highs[i] is peak if greater than k bars on each side."""
        swings = []
        n = len(highs)
        for i in range(k, n - k):
            window = highs[i - k:i + k + 1]
            if highs[i] == window.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swings.append(float(highs[i]))
        return swings

    @staticmethod
    def _swing_lows(lows: np.ndarray, k: int) -> list[float]:
        swings = []
        n = len(lows)
        for i in range(k, n - k):
            window = lows[i - k:i + k + 1]
            if lows[i] == window.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swings.append(float(lows[i]))
        return swings


# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------
def create_gate(config: Optional[dict] = None) -> InstitutionalEntryGate:
    """Build an InstitutionalEntryGate from the global config dict."""
    return InstitutionalEntryGate(config)
