"""
Strategy Decay Detection — track edge erosion over time
========================================================

Every strategy has a lifespan. As more participants discover and trade
the same edge, alpha decays. This module tracks:

    1. Rolling win rate          — is it trending down?
    2. Rolling Sharpe ratio      — is risk-adjusted return declining?
    3. Average winner / loser    — is the strategy losing its edge?
    4. Trade frequency           — has the strategy stopped firing?
    5. Slippage trend            — is execution cost growing (more crowding)?
    6. Edge decay score          — composite 0..1

When decay is detected, the module flags the strategy for retirement or
parameter re-tuning.

Usage:
    from trading_modules.strategy_decay import StrategyDecayDetector
    detector = StrategyDecayDetector()
    decay = detector.analyze(trades_list)
    if decay.decay_severe:
        log.warning(f"Strategy '{name}' decaying — consider retiring")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecayResult:
    decay_score: float                  # 0..1 (1 = severe decay)
    decay_severe: bool                  # score > 0.7
    decay_moderate: bool                # 0.4 < score <= 0.7
    win_rate_trend: str                 # "declining" / "stable" / "improving"
    sharpe_trend: str
    recent_win_rate: float
    baseline_win_rate: float
    recent_sharpe: float
    baseline_sharpe: float
    recent_avg_winner: float
    recent_avg_loser: float
    trade_frequency_change: float       # recent / baseline (>1 = more, <1 = less)
    slippage_trend: str                 # "increasing" / "stable" / "decreasing"
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decay_score": round(self.decay_score, 3),
            "decay_severe": self.decay_severe,
            "decay_moderate": self.decay_moderate,
            "win_rate_trend": self.win_rate_trend,
            "sharpe_trend": self.sharpe_trend,
            "recent_win_rate": round(self.recent_win_rate, 3),
            "baseline_win_rate": round(self.baseline_win_rate, 3),
            "recent_sharpe": round(self.recent_sharpe, 3),
            "baseline_sharpe": round(self.baseline_sharpe, 3),
            "trade_frequency_change": round(self.trade_frequency_change, 3),
            "slippage_trend": self.slippage_trend,
            "recommendations": self.recommendations,
            "notes": self.notes,
        }


class StrategyDecayDetector:
    """Detect strategy edge decay from trade history.

    Parameters:
        baseline_window: # of trades for baseline (default 50)
        recent_window: # of trades for recent comparison (default 20)
        win_rate_decline_threshold: recent WR / baseline WR below this = decay (default 0.85)
        sharpe_decline_threshold: similar (default 0.7)
    """

    def __init__(
        self, baseline_window: int = 50, recent_window: int = 20,
        win_rate_decline_threshold: float = 0.85,
        sharpe_decline_threshold: float = 0.7,
    ) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.win_rate_decline_threshold = win_rate_decline_threshold
        self.sharpe_decline_threshold = sharpe_decline_threshold

    def analyze(self, trades: list[dict]) -> DecayResult:
        """Analyze trade history for decay.

        Args:
            trades: list of dicts with at least: pnl, pnl_pct, timestamp, slippage (optional)
        """
        if not trades or len(trades) < self.baseline_window + self.recent_window:
            return DecayResult(
                decay_score=0.0, decay_severe=False, decay_moderate=False,
                win_rate_trend="unknown", sharpe_trend="unknown",
                recent_win_rate=0, baseline_win_rate=0,
                recent_sharpe=0, baseline_sharpe=0,
                recent_avg_winner=0, recent_avg_loser=0,
                trade_frequency_change=1.0, slippage_trend="unknown",
                notes=[f"need {self.baseline_window + self.recent_window} trades, got {len(trades)}"],
            )
        # Split into baseline (older) and recent (newer)
        # Assume trades are in chronological order
        baseline_trades = trades[:-self.recent_window][-self.baseline_window:]
        recent_trades = trades[-self.recent_window:]

        # Win rates
        baseline_wins = [t for t in baseline_trades if t.get("pnl", 0) > 0]
        recent_wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
        baseline_wr = len(baseline_wins) / len(baseline_trades) if baseline_trades else 0
        recent_wr = len(recent_wins) / len(recent_trades) if recent_trades else 0

        # Sharpe (per-trade)
        baseline_rets = np.array([t.get("pnl_pct", 0) for t in baseline_trades])
        recent_rets = np.array([t.get("pnl_pct", 0) for t in recent_trades])
        baseline_sharpe = self._sharpe(baseline_rets)
        recent_sharpe = self._sharpe(recent_rets)

        # Average winner / loser
        recent_winners = [t for t in recent_trades if t.get("pnl", 0) > 0]
        recent_losers = [t for t in recent_trades if t.get("pnl", 0) < 0]
        recent_avg_winner = float(np.mean([t["pnl"] for t in recent_winners])) if recent_winners else 0
        recent_avg_loser = float(np.mean([t["pnl"] for t in recent_losers])) if recent_losers else 0

        # Trade frequency (trades per unit time)
        # Approximate: compare timestamps of first/last trade in each window
        try:
            bl_first_ts = baseline_trades[0].get("timestamp")
            bl_last_ts = baseline_trades[-1].get("timestamp")
            r_first_ts = recent_trades[0].get("timestamp")
            r_last_ts = recent_trades[-1].get("timestamp")
            if all(ts for ts in [bl_first_ts, bl_last_ts, r_first_ts, r_last_ts]):
                from datetime import datetime
                if isinstance(bl_first_ts, str):
                    bl_first_ts = datetime.fromisoformat(bl_first_ts)
                    bl_last_ts = datetime.fromisoformat(bl_last_ts)
                    r_first_ts = datetime.fromisoformat(r_first_ts)
                    r_last_ts = datetime.fromisoformat(r_last_ts)
                bl_duration = (bl_last_ts - bl_first_ts).total_seconds() / 86400
                r_duration = (r_last_ts - r_first_ts).total_seconds() / 86400
                if bl_duration > 0 and r_duration > 0:
                    bl_freq = len(baseline_trades) / bl_duration
                    r_freq = len(recent_trades) / r_duration
                    trade_freq_change = r_freq / bl_freq if bl_freq > 0 else 1.0
                else:
                    trade_freq_change = 1.0
            else:
                trade_freq_change = 1.0
        except Exception:
            trade_freq_change = 1.0

        # Slippage trend
        baseline_slip = [t.get("slippage", 0) for t in baseline_trades if "slippage" in t]
        recent_slip = [t.get("slippage", 0) for t in recent_trades if "slippage" in t]
        if baseline_slip and recent_slip:
            bl_slip_avg = float(np.mean(baseline_slip))
            r_slip_avg = float(np.mean(recent_slip))
            if r_slip_avg > bl_slip_avg * 1.2:
                slippage_trend = "increasing"
            elif r_slip_avg < bl_slip_avg * 0.8:
                slippage_trend = "decreasing"
            else:
                slippage_trend = "stable"
        else:
            slippage_trend = "unknown"

        # ── Trends ────────────────────────────────────────────────
        if baseline_wr > 0:
            wr_ratio = recent_wr / baseline_wr
            if wr_ratio < self.win_rate_decline_threshold:
                win_rate_trend = "declining"
            elif wr_ratio > 1.1:
                win_rate_trend = "improving"
            else:
                win_rate_trend = "stable"
        else:
            win_rate_trend = "unknown"
        if baseline_sharpe > 0:
            sharpe_ratio = recent_sharpe / baseline_sharpe
            if sharpe_ratio < self.sharpe_decline_threshold:
                sharpe_trend = "declining"
            elif sharpe_ratio > 1.1:
                sharpe_trend = "improving"
            else:
                sharpe_trend = "stable"
        else:
            sharpe_trend = "unknown"

        # ── Decay score (composite 0..1) ──────────────────────────
        decay_components = []
        if win_rate_trend == "declining":
            decay_components.append(min(1.0, 1 - wr_ratio if baseline_wr > 0 else 0))
        if sharpe_trend == "declining":
            decay_components.append(min(1.0, 1 - sharpe_ratio if baseline_sharpe > 0 else 0))
        if slippage_trend == "increasing":
            decay_components.append(0.5)
        if trade_freq_change < 0.5:
            decay_components.append(0.3)  # strategy stopped firing

        decay_score = float(np.mean(decay_components)) if decay_components else 0.0
        decay_severe = decay_score > 0.7
        decay_moderate = 0.4 < decay_score <= 0.7

        recommendations: list[str] = []
        if decay_severe:
            recommendations.append("SEVERE decay — retire strategy or re-tune parameters")
        elif decay_moderate:
            recommendations.append("Moderate decay — review strategy, consider reducing allocation")
        if win_rate_trend == "declining":
            recommendations.append(f"Win rate declining: {baseline_wr:.1%} → {recent_wr:.1%}")
        if sharpe_trend == "declining":
            recommendations.append(f"Sharpe declining: {baseline_sharpe:.2f} → {recent_sharpe:.2f}")
        if slippage_trend == "increasing":
            recommendations.append("Slippage increasing — strategy may be getting crowded")
        if not recommendations:
            recommendations.append("Strategy healthy — no decay detected")

        notes = [
            f"baseline: {len(baseline_trades)} trades, WR={baseline_wr:.1%}, Sharpe={baseline_sharpe:.2f}",
            f"recent: {len(recent_trades)} trades, WR={recent_wr:.1%}, Sharpe={recent_sharpe:.2f}",
            f"trade_freq_change: {trade_freq_change:.2f}x",
        ]

        return DecayResult(
            decay_score=decay_score,
            decay_severe=bool(decay_severe),
            decay_moderate=bool(decay_moderate),
            win_rate_trend=win_rate_trend,
            sharpe_trend=sharpe_trend,
            recent_win_rate=float(recent_wr),
            baseline_win_rate=float(baseline_wr),
            recent_sharpe=float(recent_sharpe),
            baseline_sharpe=float(baseline_sharpe),
            recent_avg_winner=recent_avg_winner,
            recent_avg_loser=recent_avg_loser,
            trade_frequency_change=float(trade_freq_change),
            slippage_trend=slippage_trend,
            recommendations=recommendations,
            notes=notes,
        )

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        std = float(returns.std(ddof=1))
        if std <= 0:
            return 0.0
        return float(returns.mean() / std * np.sqrt(252))


__all__ = ["StrategyDecayDetector", "DecayResult"]
