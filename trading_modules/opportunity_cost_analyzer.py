"""trading_modules/opportunity_cost_analyzer.py
=====================================================================
Opportunity Cost Analyzer (Principle #176)
=====================================================================
Evaluates whether taking a trade NOW is better than WAITING for a
potentially better setup later.

Core Question:
    "Is this current setup good enough, or should I wait for a better one?"

Factors Considered:
    1. Current setup quality (0-100 score)
    2. Expected time until next better setup
    3. Capital cost of waiting (capital sitting idle)
    4. Risk of missing this setup entirely
    5. Market cycle phase (some phases produce more setups)
    6. Historical frequency of better setups

Decision Matrix:
    Current Score > 85  → TAKE IT (opportunity cost of waiting too high)
    Current Score 70-85 → TAKE IT if no better setup expected soon
    Current Score 55-70 → WAIT if better setup expected within 1 hour
    Current Score < 55  → WAIT (not good enough)

Usage:
    analyzer = OpportunityCostAnalyzer()

    decision = analyzer.evaluate(
        current_score=72,
        expected_better_setup_minutes=30,
        capital_idle_cost_pct=0.01,
        setup_frequency_per_day=8,
        historical_avg_score=65,
    )
    # decision = {
    #     "action": "take_trade",
    #     "reason": "Good setup, no better expected soon",
    #     "opportunity_cost_of_waiting": 0.15,
    #     "expected_value_of_waiting": -0.05,
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.opportunity_cost_analyzer")


class OpportunityDecision(str, Enum):
    TAKE_TRADE = "take_trade"
    WAIT = "wait"
    SKIP = "skip"


@dataclass
class OpportunityCostResult:
    """Opportunity cost analysis result."""
    decision: OpportunityDecision = OpportunityDecision.TAKE_TRADE
    current_score: float = 0.0
    expected_better_score: float = 0.0
    expected_wait_minutes: float = 0.0

    # Cost analysis
    opportunity_cost_of_waiting: float = 0.0   # R we lose by not trading now
    expected_value_of_waiting: float = 0.0     # R we might gain by waiting
    capital_idle_cost_pct: float = 0.0         # % cost of capital sitting idle
    risk_of_missing: float = 0.0               # P(this setup won't come again)

    # Recommendation
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "current_score": round(self.current_score, 1),
            "expected_better_score": round(self.expected_better_score, 1),
            "expected_wait_minutes": round(self.expected_wait_minutes, 0),
            "opportunity_cost_of_waiting": round(self.opportunity_cost_of_waiting, 3),
            "expected_value_of_waiting": round(self.expected_value_of_waiting, 3),
            "capital_idle_cost_pct": round(self.capital_idle_cost_pct, 4),
            "risk_of_missing": round(self.risk_of_missing, 3),
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
        }


class OpportunityCostAnalyzer:
    """Evaluates opportunity cost of taking vs waiting."""

    def __init__(self,
                 min_score_to_take: float = 70.0,
                 excellent_score: float = 85.0,
                 poor_score: float = 55.0,
                 capital_daily_cost_pct: float = 0.02):
        """Initialize analyzer.

        Args:
            min_score_to_take: minimum setup score to consider taking
            excellent_score: above this, always take
            poor_score: below this, always wait
            capital_daily_cost_pct: daily cost of capital sitting idle (%)
        """
        self.min_score = min_score_to_take
        self.excellent_score = excellent_score
        self.poor_score = poor_score
        self.capital_daily_cost = capital_daily_cost_pct

    def evaluate(self,
                 current_score: float,
                 expected_better_setup_minutes: float = 60.0,
                 capital_idle_cost_pct: float = 0.01,
                 setup_frequency_per_day: float = 8.0,
                 historical_avg_score: float = 65.0,
                 historical_avg_r: float = 0.3,
                 market_cycle: str = "unknown") -> OpportunityCostResult:
        """Evaluate opportunity cost.

        Args:
            current_score: setup quality score (0-100)
            expected_better_setup_minutes: expected time until a better setup
            capital_idle_cost_pct: % cost of capital sitting idle while waiting
            setup_frequency_per_day: how many setups per day typically
            historical_avg_score: average historical setup score
            historical_avg_r: average R per trade historically
            market_cycle: current market cycle phase

        Returns:
            OpportunityCostResult with decision + reasoning
        """
        result = OpportunityCostResult(
            current_score=current_score,
            expected_wait_minutes=expected_better_setup_minutes,
            capital_idle_cost_pct=capital_idle_cost_pct,
        )

        # === Expected better setup score ===
        # If setups are frequent, a better one is likely soon
        if setup_frequency_per_day > 10:
            result.expected_better_score = min(100, historical_avg_score + 10)
        elif setup_frequency_per_day > 5:
            result.expected_better_score = min(100, historical_avg_score + 5)
        else:
            result.expected_better_score = historical_avg_score

        # === Opportunity cost of waiting ===
        # If we wait, we lose the EV of this trade
        current_ev_r = (current_score / 100) * historical_avg_r
        result.opportunity_cost_of_waiting = current_ev_r

        # === Expected value of waiting ===
        # If we wait, we might get a better setup
        better_ev_r = (result.expected_better_score / 100) * historical_avg_r
        # But we pay capital idle cost
        wait_cost_r = (capital_idle_cost_pct / 100) * (expected_better_setup_minutes / 1440) * 100
        # And risk that the better setup doesn't come
        probability_better_comes = min(1.0, setup_frequency_per_day * expected_better_setup_minutes / 1440)
        result.expected_value_of_waiting = (
            probability_better_comes * (better_ev_r - current_ev_r) - wait_cost_r
        )

        # === Risk of missing ===
        # How likely is it that this exact setup won't come again today?
        if setup_frequency_per_day > 10:
            result.risk_of_missing = 0.2  # will probably come again
        elif setup_frequency_per_day > 5:
            result.risk_of_missing = 0.5
        else:
            result.risk_of_missing = 0.8  # rare setup, don't miss it

        # === Decision logic ===
        result.decision, result.reason, result.confidence = self._decide(
            result, market_cycle)

        return result

    def _decide(self, r: OpportunityCostResult,
                market_cycle: str) -> tuple:
        """Make the take/wait/skip decision.

        Returns (decision, reason, confidence)
        """
        # Excellent setup — always take
        if r.current_score >= self.excellent_score:
            return (
                OpportunityDecision.TAKE_TRADE,
                f"Excellent setup ({r.current_score:.0f}) — opportunity cost of waiting too high "
                f"(would lose {r.opportunity_cost_of_waiting:.2f}R)",
                0.9,
            )

        # Poor setup — always wait
        if r.current_score < self.poor_score:
            return (
                OpportunityDecision.WAIT,
                f"Poor setup ({r.current_score:.0f}) — wait for better "
                f"(expected {r.expected_better_score:.0f} in {r.expected_wait_minutes:.0f}min)",
                0.8,
            )

        # Good setup (55-85) — compare EV of waiting vs taking
        if r.expected_value_of_waiting > 0.1:
            # Waiting is clearly better
            return (
                OpportunityDecision.WAIT,
                f"Good setup ({r.current_score:.0f}) but better expected soon "
                f"(EV of waiting: +{r.expected_value_of_waiting:.2f}R)",
                0.7,
            )

        if r.expected_value_of_waiting < -0.05:
            # Taking is clearly better
            return (
                OpportunityDecision.TAKE_TRADE,
                f"Good setup ({r.current_score:.0f}) — no better expected soon "
                f"(EV of waiting: {r.expected_value_of_waiting:.2f}R)",
                0.75,
            )

        # Marginal — consider risk of missing
        if r.risk_of_missing > 0.7:
            return (
                OpportunityDecision.TAKE_TRADE,
                f"Good setup ({r.current_score:.0f}) — rare opportunity, "
                f"risk of missing = {r.risk_of_missing:.0%}",
                0.65,
            )

        # Default: take if above min score
        if r.current_score >= self.min_score:
            return (
                OpportunityDecision.TAKE_TRADE,
                f"Acceptable setup ({r.current_score:.0f}) — take it",
                0.6,
            )

        return (
            OpportunityDecision.WAIT,
            f"Setup ({r.current_score:.0f}) below minimum ({self.min_score}) — wait",
            0.7,
        )
