"""
Failure Analysis — classify why each trade failed
===================================================

Not all losses are equal. A loss from bad prediction is different from a
loss from bad execution or unexpected news. This module classifies each
losing trade into one of:

    1. Bad prediction       — the signal was wrong (regime/trend misread)
    2. Bad timing            — right direction, entered too early/late
    3. Bad execution         — slippage/spread ate the edge
    4. Bad exit              — right entry, exited too early/late
    5. Unexpected news       — black swan event
    6. Random noise          — no identifiable cause (stop hit by wick)
    7. Risk management       — position too large, killed by volatility
    8. Liquidity event       — flash crash, gap, no buyers

Each classification suggests a different corrective action.

Usage:
    from trading_modules.failure_analysis import FailureAnalyzer, FailedTrade
    analyzer = FailureAnalyzer()
    result = analyzer.classify(FailedTrade(
        symbol="BTCUSD", direction="BUY",
        entry=65000, exit=64500, stop=64400,
        pnl=-50, pnl_pct=-0.77,
        entry_time=..., exit_time=...,
        market_data_around_exit=df,
        had_news_during_trade=False,
    ))
    print(f"Failure type: {result.failure_type} — {result.recommendation}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FailedTrade:
    symbol: str
    direction: str               # "BUY" / "SELL"
    entry: float
    exit: float
    stop: float
    take_profit: Optional[float]
    pnl: float                   # negative for loss
    pnl_pct: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    market_data_around_exit: Optional[pd.DataFrame] = None
    had_news_during_trade: bool = False
    slippage: float = 0.0        # in price units
    spread_at_exit: float = 0.0
    hold_minutes: float = 0.0


@dataclass
class FailureResult:
    failure_type: str            # "bad_prediction" / "bad_timing" / etc.
    confidence: float            # 0..1
    explanation: str
    recommendation: str
    secondary_types: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "secondary_types": self.secondary_types,
            "notes": self.notes,
        }


class FailureAnalyzer:
    """Classify why a losing trade failed.

    Parameters:
        slippage_threshold_bps: slippage above this (bps) = execution issue (default 10)
        news_window_minutes: news within this window before exit = news-related (default 30)
        wick_threshold: stop hit by wick only (close recovered) = noise (default 0.7)
    """

    def __init__(
        self, slippage_threshold_bps: float = 10,
        news_window_minutes: int = 30,
        wick_threshold: float = 0.7,
    ) -> None:
        self.slippage_threshold_bps = slippage_threshold_bps
        self.news_window_minutes = news_window_minutes
        self.wick_threshold = wick_threshold

    def classify(self, trade: FailedTrade) -> FailureResult:
        """Classify a losing trade."""
        # Build evidence
        evidence: list[tuple[str, float, str]] = []
        # (failure_type, confidence, explanation)

        direction = trade.direction.upper()
        # ── 1. Bad prediction (price went against you immediately) ──
        # If price moved against you from the start, the signal was wrong
        if trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            # Look at bars between entry and exit
            if "close" in df.columns and len(df) > 5:
                closes = df["close"].to_numpy(dtype=float)
                if direction == "BUY":
                    adverse_move = float((trade.entry - closes.min()) / trade.entry)
                else:
                    adverse_move = float((closes.max() - trade.entry) / trade.entry)
                if adverse_move > 0.02:  # >2% adverse move
                    evidence.append((
                        "bad_prediction", 0.7,
                        f"Price moved {adverse_move:.1%} against signal from entry"
                    ))

        # ── 2. Bad timing (right direction, entered too early) ─────
        # If price eventually went your way but you got stopped out first
        if trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            if "close" in df.columns and len(df) > 10:
                # Look at price AFTER your exit
                post_exit = df["close"].iloc[-5:]
                if direction == "BUY" and post_exit.iloc[-1] > trade.entry:
                    evidence.append((
                        "bad_timing", 0.6,
                        "Price recovered above entry after you were stopped out — entered too early"
                    ))
                elif direction == "SELL" and post_exit.iloc[-1] < trade.entry:
                    evidence.append((
                        "bad_timing", 0.6,
                        "Price fell below entry after you were stopped out — entered too early"
                    ))

        # ── 3. Bad execution (slippage ate the edge) ───────────────
        if trade.slippage > 0:
            slippage_bps = (trade.slippage / trade.entry) * 10000
            if slippage_bps > self.slippage_threshold_bps:
                evidence.append((
                    "bad_execution", 0.7,
                    f"Slippage of {slippage_bps:.1f} bps exceeded threshold"
                ))

        # ── 4. Bad exit (exited too early or too late) ─────────────
        # If price hit TP level after you exited (exited too early)
        if trade.take_profit is not None and trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            if "high" in df.columns and "low" in df.columns:
                if direction == "BUY":
                    # Did price reach TP after exit?
                    post_exit_high = float(df["high"].iloc[-5:].max())
                    if post_exit_high >= trade.take_profit:
                        evidence.append((
                            "bad_exit", 0.65,
                            "Price reached take-profit after you exited — exited too early"
                        ))
                else:
                    post_exit_low = float(df["low"].iloc[-5:].min())
                    if post_exit_low <= trade.take_profit:
                        evidence.append((
                            "bad_exit", 0.65,
                            "Price reached take-profit after you exited — exited too early"
                        ))

        # ── 5. Unexpected news ─────────────────────────────────────
        if trade.had_news_during_trade:
            evidence.append((
                "unexpected_news", 0.8,
                "High-impact news event occurred during the trade"
            ))

        # ── 6. Random noise (stop hit by wick, close recovered) ────
        if trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            if "high" in df.columns and "low" in df.columns and "close" in df.columns:
                # Find the bar where stop was hit
                if direction == "BUY":
                    stop_hit_bars = df[df["low"] <= trade.stop]
                else:
                    stop_hit_bars = df[df["high"] >= trade.stop]
                if not stop_hit_bars.empty:
                    stop_bar = stop_hit_bars.iloc[0]
                    rng = float(stop_bar["high"]) - float(stop_bar["low"])
                    if rng > 0:
                        if direction == "BUY":
                            wick_below_stop = float(stop_bar["low"]) - trade.stop
                            recovery = float(stop_bar["close"]) - trade.stop
                        else:
                            wick_below_stop = trade.stop - float(stop_bar["high"])
                            recovery = trade.stop - float(stop_bar["close"])
                        # If close recovered well above stop, it was just a wick
                        if recovery > 0 and rng > 0:
                            recovery_ratio = recovery / rng
                            if recovery_ratio > self.wick_threshold:
                                evidence.append((
                                    "random_noise", 0.7,
                                    f"Stop hit by wick only — close recovered {recovery_ratio:.0%} of range"
                                ))

        # ── 7. Risk management (position too large for volatility) ─
        if trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            if "high" in df.columns and "low" in df.columns and len(df) > 14:
                # ATR at entry
                atr = float((df["high"] - df["low"]).tail(14).mean())
                stop_distance = abs(trade.entry - trade.stop)
                if atr > 0 and stop_distance < atr * 0.5:
                    evidence.append((
                        "risk_management", 0.6,
                        f"Stop distance ({stop_distance:.2f}) < 0.5 × ATR ({atr:.2f}) — too tight for volatility"
                    ))

        # ── 8. Liquidity event (gap / flash crash) ─────────────────
        if trade.market_data_around_exit is not None:
            df = trade.market_data_around_exit
            if "close" in df.columns and len(df) > 2:
                returns = df["close"].pct_change().dropna()
                if len(returns) > 0:
                    max_bar_move = float(returns.abs().max())
                    if max_bar_move > 0.05:  # >5% in a single bar
                        evidence.append((
                            "liquidity_event", 0.75,
                            f"Flash move of {max_bar_move:.1%} in a single bar — liquidity event"
                        ))

        # ── Default: random noise ──────────────────────────────────
        if not evidence:
            evidence.append((
                "random_noise", 0.4,
                "No identifiable cause — likely random market noise"
            ))

        # Sort by confidence
        evidence.sort(key=lambda x: x[1], reverse=True)
        primary = evidence[0]
        secondary = [e[0] for e in evidence[1:3]]  # top 2 secondary

        # Recommendation per failure type
        recommendations = {
            "bad_prediction": "Review signal logic — regime/trend may have been misread. Reduce position size on similar setups.",
            "bad_timing": "Wait for confirmation (e.g., retest, BOS) before entering. Consider limit orders at key levels.",
            "bad_execution": "Use limit orders instead of market orders. Trade during higher-liquidity sessions.",
            "bad_exit": "Use trailing stops or partial exits. Don't exit on fear — let the trade play out to TP/SL.",
            "unexpected_news": "Check economic calendar before trading. Implement news blackout filter.",
            "random_noise": "Widen stop slightly (1.5-2x ATR) to avoid wick stops. This is unavoidable noise.",
            "risk_management": "Size positions based on ATR, not fixed %. Stop should be at least 1.5x ATR.",
            "liquidity_event": "Avoid trading during low-liquidity hours. Use options to define risk instead of stops.",
        }

        recommendation = recommendations.get(primary[0], "Review trade and adjust strategy.")

        notes = [f"{e[0]}: {e[2]} (conf={e[1]:.2f})" for e in evidence]

        return FailureResult(
            failure_type=primary[0],
            confidence=float(primary[1]),
            explanation=primary[2],
            recommendation=recommendation,
            secondary_types=secondary,
            notes=notes,
        )


__all__ = ["FailureAnalyzer", "FailedTrade", "FailureResult"]
