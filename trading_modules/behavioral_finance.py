"""
Behavioral Finance — detect biases in your own + market's behavior
====================================================================

Behavioral finance studies how cognitive biases affect trading decisions.
This module detects:

    1. Disposition Effect    — selling winners too early, holding losers too long
    2. Anchoring Bias        — fixating on entry price or recent high/low
    3. Loss Aversion         — refusing to take a small loss, hoping it recovers
    4. Herd Behavior         — following the crowd into/out of trades
    5. Overreaction          — oversized response to recent news
    6. Underreaction         — failure to react to clear signals
    7. Prospect Theory       — risk-seeking in losses, risk-averse in gains

The analyzer examines your trade history + current market action to flag
these biases in real time.

Usage:
    from trading_modules.behavioral_finance import BehavioralAnalyzer, TraderHistory
    analyzer = BehavioralAnalyzer()
    bias = analyzer.analyze(trader_history, current_position_pnl=-50)
    if bias.disposition_effect_detected:
        log.warning("You're holding losers too long — consider cutting")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TraderHistory:
    """Snapshot of recent trade history for bias analysis."""
    # List of closed trades, each with: entry, exit, pnl, pnl_pct, hold_minutes, win
    closed_trades: list[dict] = field(default_factory=list)
    # Current open position (if any)
    open_position_pnl: Optional[float] = None
    open_position_pnl_pct: Optional[float] = None
    open_position_hold_minutes: Optional[float] = None
    # Current market state
    market_volatility_atr_ratio: float = 1.0
    market_volume_ratio: float = 1.0


@dataclass
class BehavioralResult:
    disposition_effect_detected: bool       # selling winners early, holding losers
    anchoring_bias_detected: bool
    loss_aversion_detected: bool
    herd_behavior_detected: bool
    overreaction_detected: bool
    underreaction_detected: bool
    prospect_theory_violation: bool
    bias_score: float                         # 0..1 (higher = more biased)
    avg_winner_hold_minutes: float = 0.0
    avg_loser_hold_minutes: float = 0.0
    avg_winner_pnl: float = 0.0
    avg_loser_pnl: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "disposition_effect_detected": self.disposition_effect_detected,
            "anchoring_bias_detected": self.anchoring_bias_detected,
            "loss_aversion_detected": self.loss_aversion_detected,
            "herd_behavior_detected": self.herd_behavior_detected,
            "overreaction_detected": self.overreaction_detected,
            "underreaction_detected": self.underreaction_detected,
            "prospect_theory_violation": self.prospect_theory_violation,
            "bias_score": round(self.bias_score, 3),
            "avg_winner_hold_minutes": round(self.avg_winner_hold_minutes, 1),
            "avg_loser_hold_minutes": round(self.avg_loser_hold_minutes, 1),
            "avg_winner_pnl": round(self.avg_winner_pnl, 2),
            "avg_loser_pnl": round(self.avg_loser_pnl, 2),
            "recommendations": self.recommendations,
            "notes": self.notes,
        }


class BehavioralAnalyzer:
    """Detect behavioral biases from trade history.

    Parameters:
        min_trades: minimum trades needed for reliable analysis (default 10)
        winner_loser_hold_ratio: losers held > this × winners = disposition effect (default 1.5)
        loss_aversion_threshold: losing trades held > this many minutes = loss aversion (default 240)
    """

    def __init__(
        self, min_trades: int = 10,
        winner_loser_hold_ratio: float = 1.5,
        loss_aversion_threshold: float = 240,
    ) -> None:
        self.min_trades = min_trades
        self.winner_loser_hold_ratio = winner_loser_hold_ratio
        self.loss_aversion_threshold = loss_aversion_threshold

    def analyze(self, history: TraderHistory) -> BehavioralResult:
        if not history.closed_trades or len(history.closed_trades) < self.min_trades:
            return BehavioralResult(
                False, False, False, False, False, False, False, 0.0,
                notes=[f"need at least {self.min_trades} trades, got {len(history.closed_trades)}"],
            )
        trades = history.closed_trades
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) < 0]
        if not winners or not losers:
            return BehavioralResult(
                False, False, False, False, False, False, False, 0.0,
                notes=["no winners or no losers in history — cannot detect biases"],
            )
        avg_winner_hold = float(np.mean([t.get("hold_minutes", 0) for t in winners]))
        avg_loser_hold = float(np.mean([t.get("hold_minutes", 0) for t in losers]))
        avg_winner_pnl = float(np.mean([t.get("pnl", 0) for t in winners]))
        avg_loser_pnl = float(np.mean([t.get("pnl", 0) for t in losers]))
        avg_winner_pct = float(np.mean([t.get("pnl_pct", 0) for t in winners]))
        avg_loser_pct = float(np.mean([abs(t.get("pnl_pct", 0)) for t in losers]))

        bias_count = 0
        recs: list[str] = []
        notes: list[str] = []

        # ── Disposition Effect ────────────────────────────────────
        # Holding losers longer than winners
        disposition = avg_loser_hold > avg_winner_hold * self.winner_loser_hold_ratio
        if disposition:
            bias_count += 1
            recs.append(
                f"Disposition effect: losers held {avg_loser_hold:.0f}min vs "
                f"winners {avg_winner_hold:.0f}min — cut losers faster"
            )
        notes.append(f"avg_winner_hold={avg_winner_hold:.1f}min, avg_loser_hold={avg_loser_hold:.1f}min")

        # ── Loss Aversion ─────────────────────────────────────────
        # Refusing to cut losses — losers held way too long
        loss_aversion = avg_loser_hold > self.loss_aversion_threshold
        if loss_aversion:
            bias_count += 1
            recs.append(
                f"Loss aversion: avg losing trade held {avg_loser_hold:.0f}min "
                f"(> {self.loss_aversion_threshold}min threshold)"
            )

        # ── Anchoring Bias ────────────────────────────────────────
        # If you have an open position, check if pnl is negative AND you've held long
        anchoring = False
        if (history.open_position_pnl is not None and
                history.open_position_pnl < 0 and
                history.open_position_hold_minutes is not None and
                history.open_position_hold_minutes > avg_winner_hold * 2):
            anchoring = True
            bias_count += 1
            recs.append(
                f"Anchoring bias: holding losing position for "
                f"{history.open_position_hold_minutes:.0f}min — review stop loss"
            )

        # ── Herd Behavior ─────────────────────────────────────────
        # Entering trades when market volume is unusually high (FOMO entries)
        herd = (history.market_volume_ratio > 1.5 and
                len([t for t in trades[-5:] if t.get("pnl", 0) < 0]) >= 3)
        if herd:
            bias_count += 1
            recs.append("Herd behavior: entering during high-volume spikes — recent 5 trades mostly losses")

        # ── Overreaction ──────────────────────────────────────────
        # Sizing up significantly after recent losses (chasing)
        recent_5 = trades[-5:]
        recent_sizes = [t.get("notional", 0) for t in recent_5]
        if len(recent_sizes) >= 3:
            size_ratio = max(recent_sizes) / max(min(recent_sizes), 1)
            recent_loss_streak = sum(1 for t in recent_5 if t.get("pnl", 0) < 0)
            overreaction = size_ratio > 2.0 and recent_loss_streak >= 2
        else:
            overreaction = False
        if overreaction:
            bias_count += 1
            recs.append("Overreaction: position sizing spiked after recent losses — revenge trading")

        # ── Underreaction ─────────────────────────────────────────
        # Win rate very low with very small position sizes
        win_rate = len(winners) / len(trades)
        underreaction = win_rate < 0.3 and avg_winner_pct < 0.005
        if underreaction:
            bias_count += 1
            recs.append("Underreaction: tiny position sizes + low win rate — increase conviction")

        # ── Prospect Theory Violation ─────────────────────────────
        # Risk-seeking in losses (holding losers), risk-averse in gains (cutting winners early)
        prospect_violation = (disposition and
                              avg_winner_pct < avg_loser_pct * 0.8)
        if prospect_violation:
            bias_count += 1
            recs.append("Prospect theory violation: cutting winners at small gains while holding large losers")

        # ── Bias Score ────────────────────────────────────────────
        bias_score = bias_count / 7.0  # 7 possible biases

        notes.append(f"win_rate={win_rate:.1%} ({len(winners)}W/{len(losers)}L)")
        notes.append(f"avg_winner_pnl=${avg_winner_pnl:.2f} ({avg_winner_pct:+.2%})")
        notes.append(f"avg_loser_pnl=${avg_loser_pnl:.2f} ({-avg_loser_pct:.2%})")

        if not recs:
            recs.append("No significant biases detected — discipline looks good")

        return BehavioralResult(
            disposition_effect_detected=bool(disposition),
            anchoring_bias_detected=bool(anchoring),
            loss_aversion_detected=bool(loss_aversion),
            herd_behavior_detected=bool(herd),
            overreaction_detected=bool(overreaction),
            underreaction_detected=bool(underreaction),
            prospect_theory_violation=bool(prospect_violation),
            bias_score=float(bias_score),
            avg_winner_hold_minutes=avg_winner_hold,
            avg_loser_hold_minutes=avg_loser_hold,
            avg_winner_pnl=avg_winner_pnl,
            avg_loser_pnl=avg_loser_pnl,
            recommendations=recs,
            notes=notes,
        )


__all__ = ["BehavioralAnalyzer", "TraderHistory", "BehavioralResult"]
