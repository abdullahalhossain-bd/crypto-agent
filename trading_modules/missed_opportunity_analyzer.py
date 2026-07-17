"""trading_modules/missed_opportunity_analyzer.py
=====================================================================
Missed Opportunity Analyzer (Principle #152)
=====================================================================
Tracks setups the bot DIDN'T trade, then backtests what WOULD have
happened. This teaches the bot to recognize patterns it's missing.

How It Works:
    1. When a signal is generated but REJECTED (by risk gate, wisdom
       gate, or context filter), record it as a "missed opportunity."
    2. Track what happened to price after the missed signal.
    3. After N bars, evaluate: would this trade have won or lost?
    4. Store the result with full context for later analysis.
    5. Periodically analyze: which rejection reasons are costing us money?

Insights Generated:
    - "We're rejecting too many good setups (overly conservative)"
    - "Wisdom Gate is rejecting 70% of trades that would have won"
    - "Context filter is blocking profitable regime entries"
    - "Risk gate is too tight — missed 5 winners of 2R+"

Usage:
    analyzer = MissedOpportunityAnalyzer()

    # When a signal is rejected:
    analyzer.record_missed(
        symbol="BTCUSD", action="BUY", entry_price=43250,
        sl=42500, tp=45000, rejection_reason="wisdom_gate",
        context={"rsi": 62, "regime": "trend_up"},
    )

    # Each cycle, update missed opportunities with current prices:
    analyzer.update_prices({"BTCUSD": 43800})

    # Periodically analyze:
    report = analyzer.analyze()
    # report = {
    #     "total_missed": 25,
    #     "would_have_won": 15,
    #     "would_have_lost": 7,
    #     "still_open": 3,
    #     "win_rate_if_traded": 0.68,
    #     "avg_r_if_traded": 1.4,
    #     "total_r_lost": 21.0,
    #     "by_reason": {"wisdom_gate": {"count": 10, "win_rate": 0.70}, ...}
    # }
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.missed_opportunity_analyzer")


@dataclass
class MissedOpportunity:
    """A signal that was rejected but tracked for analysis."""
    id: str = ""
    timestamp: float = 0.0
    symbol: str = ""
    action: str = ""           # BUY or SELL
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    rejection_reason: str = ""  # "wisdom_gate", "risk_pipeline", "context_filter"
    rejection_detail: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    # Tracking
    current_price: float = 0.0
    status: str = "open"      # open, would_win, would_loss, would_breakeven, expired
    would_be_pnl: float = 0.0
    would_be_r: float = 0.0
    bars_since: int = 0
    resolved_at: float = 0.0
    # Risk metrics at rejection time
    confidence_at_rejection: float = 0.0
    strength_at_rejection: float = 0.0


class MissedOpportunityAnalyzer:
    """Tracks and analyzes rejected signals.

    Helps answer: "Are we being too conservative?"
    """

    def __init__(self,
                 max_bars_to_resolve: int = 20,
                 max_tracked: int = 500):
        """Initialize analyzer.

        Args:
            max_bars_to_resolve: bars to wait before declaring outcome
            max_tracked: max missed opportunities to track
        """
        self.max_bars = max_bars_to_resolve
        self.max_tracked = max_tracked
        self._lock = threading.RLock()
        self._missed: List[MissedOpportunity] = []
        # Critical #8 fix: separate list for resolved opportunities.
        self._resolved: List[MissedOpportunity] = []
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Record a missed opportunity
    # ------------------------------------------------------------------
    def record_missed(self,
                      symbol: str, action: str,
                      entry_price: float, sl: float, tp: float,
                      rejection_reason: str, rejection_detail: str = "",
                      context: Optional[Dict[str, Any]] = None,
                      confidence: float = 0.0, strength: float = 0.0) -> str:
        """Record a signal that was rejected.

        Args:
            symbol: trading symbol
            action: "BUY" or "SELL"
            entry_price: intended entry price
            sl: stop loss price
            tp: take profit price
            rejection_reason: which gate rejected it
            rejection_detail: detailed reason
            context: market context at rejection time
            confidence: signal confidence at rejection
            strength: signal strength at rejection

        Returns:
            missed_id for tracking
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._missed) >= self.max_tracked:
                self._missed.pop(0)

            missed = MissedOpportunity(
                id=f"missed_{int(time.time()*1000)}_{len(self._missed)}",
                timestamp=time.time(),
                symbol=symbol, action=action,
                entry_price=entry_price, sl=sl, tp=tp,
                rejection_reason=rejection_reason,
                rejection_detail=rejection_detail,
                context=context or {},
                current_price=entry_price,
                confidence_at_rejection=confidence,
                strength_at_rejection=strength,
            )
            self._missed.append(missed)
            log.debug("missed_opp: recorded %s %s rejected by %s",
                      symbol, action, rejection_reason)
            return missed.id

    # ------------------------------------------------------------------
    # Update prices + resolve
    # ------------------------------------------------------------------
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices and resolve mature missed opportunities.

        Args:
            prices: {symbol: current_price}
        """
        with self._lock:
            self._cycle_count += 1
            for m in self._missed:
                if m.status != "open":
                    continue
                if m.symbol not in prices:
                    continue
                m.current_price = prices[m.symbol]
                m.bars_since += 1

                # Check if SL or TP would have been hit
                if m.action == "BUY":
                    if m.current_price <= m.sl:
                        m.status = "would_loss"
                        m.would_be_pnl = (m.current_price - m.entry_price) * 1  # 1 lot approx
                        risk = abs(m.entry_price - m.sl)
                        m.would_be_r = -1.0  # hit SL = -1R
                        m.resolved_at = time.time()
                    elif m.current_price >= m.tp:
                        m.status = "would_win"
                        m.would_be_pnl = (m.current_price - m.entry_price) * 1
                        reward = abs(m.tp - m.entry_price)
                        risk = abs(m.entry_price - m.sl)
                        m.would_be_r = reward / max(risk, 1e-10)
                        m.resolved_at = time.time()
                elif m.action == "SELL":
                    if m.current_price >= m.sl:
                        m.status = "would_loss"
                        m.would_be_pnl = (m.entry_price - m.current_price) * 1
                        m.would_be_r = -1.0
                        m.resolved_at = time.time()
                    elif m.current_price <= m.tp:
                        m.status = "would_win"
                        m.would_be_pnl = (m.entry_price - m.current_price) * 1
                        reward = abs(m.entry_price - m.tp)
                        risk = abs(m.sl - m.entry_price)
                        m.would_be_r = reward / max(risk, 1e-10)
                        m.resolved_at = time.time()

                # Expire if too many bars passed
                if m.bars_since >= self.max_bars and m.status == "open":
                    # Compute unrealized outcome
                    if m.action == "BUY":
                        unrealized_r = (m.current_price - m.entry_price) / max(abs(m.entry_price - m.sl), 1e-10)
                    else:
                        unrealized_r = (m.entry_price - m.current_price) / max(abs(m.sl - m.entry_price), 1e-10)

                    if unrealized_r > 0.1:
                        m.status = "would_win"
                    elif unrealized_r < -0.1:
                        m.status = "would_loss"
                    else:
                        m.status = "would_breakeven"
                    m.would_be_r = unrealized_r
                    m.resolved_at = time.time()

            # Critical #8 fix: move resolved opportunities to a separate list
            # to prevent O(n) growth. The _missed list only contains open
            # opportunities; resolved ones go to _resolved for historical
            # analysis without affecting performance.
            still_open = []
            for m in self._missed:
                if m.status == "open":
                    still_open.append(m)
                else:
                    self._resolved.append(m)
            self._missed = still_open
            # Cap resolved list to prevent unbounded growth.
            if len(self._resolved) > 1000:
                self._resolved = self._resolved[-1000:]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, Any]:
        """Analyze missed opportunities.

        Returns comprehensive report including:
            - Total missed, won, lost counts
            - Win rate if we had traded
            - Average R if traded
            - Total R lost by not trading
            - Breakdown by rejection reason
            - Recommendations
        """
        with self._lock:
            missed = list(self._missed)

        total = len(missed)
        resolved = [m for m in missed if m.status in ("would_win", "would_loss", "would_breakeven")]
        still_open = [m for m in missed if m.status == "open"]

        wins = [m for m in resolved if m.status == "would_win"]
        losses = [m for m in resolved if m.status == "would_loss"]
        breakevens = [m for m in resolved if m.status == "would_breakeven"]

        win_rate = len(wins) / max(len(resolved), 1)
        avg_r = sum(m.would_be_r for m in resolved) / max(len(resolved), 1)
        total_r_lost = sum(m.would_be_r for m in wins)  # R we missed by not trading winners

        # By rejection reason
        by_reason: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0, "wins": 0, "losses": 0, "total_r": 0.0, "win_rate": 0.0
        })
        for m in resolved:
            r = by_reason[m.rejection_reason]
            r["count"] += 1
            if m.status == "would_win":
                r["wins"] += 1
                r["total_r"] += m.would_be_r
            elif m.status == "would_loss":
                r["losses"] += 1
                r["total_r"] += m.would_be_r  # negative (we avoided a loss)
            r["win_rate"] = r["wins"] / max(r["count"], 1)

        # By symbol
        by_symbol: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0, "wins": 0, "win_rate": 0.0
        })
        for m in resolved:
            s = by_symbol[m.symbol]
            s["count"] += 1
            if m.status == "would_win":
                s["wins"] += 1
            s["win_rate"] = s["wins"] / max(s["count"], 1)

        # Insights + recommendations
        insights = self._generate_insights(win_rate, avg_r, by_reason)
        recommendations = self._recommend(win_rate, by_reason, total_r_lost)

        return {
            "total_missed": total,
            "resolved": len(resolved),
            "still_open": len(still_open),
            "would_have_won": len(wins),
            "would_have_lost": len(losses),
            "would_have_breakeven": len(breakevens),
            "win_rate_if_traded": round(win_rate, 3),
            "avg_r_if_traded": round(avg_r, 3),
            "total_r_lost": round(total_r_lost, 2),  # R missed by not trading winners
            "total_r_saved": round(sum(m.would_be_r for m in losses if m.would_be_r < 0), 2),
            "by_reason": dict(by_reason),
            "by_symbol": dict(by_symbol),
            "insights": insights,
            "recommendations": recommendations,
        }

    def _generate_insights(self, win_rate: float, avg_r: float,
                           by_reason: Dict) -> List[str]:
        """Generate insights from analysis."""
        insights = []
        if win_rate > 0.6 and avg_r > 0.3:
            insights.append(
                f"We're being too conservative — {win_rate:.0%} of rejected trades would have won "
                f"with avg {avg_r:.2f}R. Consider loosening filters."
            )
        if win_rate < 0.3:
            insights.append(
                f"Rejection filters are working well — only {win_rate:.0%} would have won."
            )
        # Per-reason insights
        for reason, stats in by_reason.items():
            if stats["count"] >= 5 and stats["win_rate"] > 0.65:
                insights.append(
                    f"{reason} is rejecting {stats['count']} trades with {stats['win_rate']:.0%} win rate — "
                    f"losing {stats['total_r']:.1f}R. REVIEW THIS FILTER."
                )
            if stats["count"] >= 5 and stats["win_rate"] < 0.25:
                insights.append(
                    f"{reason} is correctly rejecting {stats['count']} trades (only {stats['win_rate']:.0%} would win)."
                )
        return insights

    def _recommend(self, win_rate: float, by_reason: Dict,
                   total_r_lost: float) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        if win_rate > 0.6 and total_r_lost > 5:
            recs.append("LOOSEN filters — we're missing too many winners")
        if win_rate < 0.3:
            recs.append("Filters are well-calibrated — maintain current strictness")

        # Find the worst filter (rejecting the most winners)
        worst_filter = None
        worst_r_lost = 0
        for reason, stats in by_reason.items():
            if stats["count"] >= 5 and stats["total_r"] > worst_r_lost and stats["win_rate"] > 0.5:
                worst_filter = reason
                worst_r_lost = stats["total_r"]
        if worst_filter:
            recs.append(f"Review {worst_filter} — losing {worst_r_lost:.1f}R by over-filtering")

        if not recs:
            recs.append("No changes needed — filter performance is balanced")
        return recs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def recent_missed(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get recent missed opportunities."""
        with self._lock:
            return [
                {
                    "id": m.id, "symbol": m.symbol, "action": m.action,
                    "entry_price": m.entry_price, "rejection_reason": m.rejection_reason,
                    "status": m.status, "would_be_r": round(m.would_be_r, 3),
                    "bars_since": m.bars_since, "timestamp": m.timestamp,
                }
                for m in self._missed[-last_n:]
            ]

    def stats(self) -> Dict[str, Any]:
        """Quick stats."""
        with self._lock:
            return {
                "total_tracked": len(self._missed),
                "open": sum(1 for m in self._missed if m.status == "open"),
                "resolved": sum(1 for m in self._missed if m.status != "open"),
                "cycles_tracked": self._cycle_count,
            }
