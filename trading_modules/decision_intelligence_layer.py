"""trading_modules/decision_intelligence_layer.py
=====================================================================
Decision Intelligence Layer (Principle #198)
=====================================================================
The final decision layer that combines ALL factors before a trade.
Focuses on DECISION QUALITY, not prediction accuracy.

Core Philosophy (Principle #198):
    "Don't ask 'what will the next candle do?'
     Ask 'what is the best decision given uncertainty?'"

Decision Theory Approach:
    1. Gather all available information (8 dimensions)
    2. Estimate probability distribution of outcomes
    3. Compute expected value of each action (trade vs wait vs skip)
    4. Choose action with best risk-adjusted EV
    5. Score the DECISION (not the outcome)

8 Decision Dimensions:
    1. MARKET CONTEXT  — regime, cycle, session, news
    2. TREND           — direction, strength, persistence
    3. LIQUIDITY       — spread, depth, volume, slippage
    4. ORDER FLOW      — delta, absorption, smart money
    5. VOLATILITY      — ATR percentile, regime, trend
    6. CORRELATION     — with open positions, benchmark
    7. EXECUTION       — recent fill quality, latency
    8. PORTFOLIO RISK  — exposure, heat, budget remaining

Decision Output:
    - Action: TRADE / WAIT / SKIP
    - Decision quality score (0-100) — how good was THIS decision?
    - Expected value (R)
    - Confidence interval
    - Regret potential (how bad if wrong)

Usage:
    layer = DecisionIntelligenceLayer()

    decision = layer.decide(
        market_context_score=0.8,
        trend_score=0.75,
        liquidity_score=0.7,
        order_flow_score=0.6,
        volatility_score=0.65,
        correlation_score=0.8,
        execution_score=0.75,
        portfolio_risk_score=0.85,
        expected_r_if_win=2.0,
        probability_win=0.65,
    )
    # decision = {
    #     "action": "TRADE",
    #     "decision_quality": 78,
    #     "expected_value_r": 0.95,
    #     "confidence": 0.7,
    #     "regret_potential": -1.0,
    #     "reason": "Strong 8-factor consensus, positive EV",
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.decision_intelligence_layer")


class DecisionAction(str, Enum):
    TRADE = "TRADE"
    WAIT = "WAIT"
    SKIP = "SKIP"


@dataclass
class DecisionResult:
    """Final decision from the intelligence layer."""
    action: DecisionAction = DecisionAction.SKIP
    decision_quality: float = 0.0     # 0-100 (how good is THIS decision?)
    expected_value_r: float = 0.0     # EV in R multiples
    confidence: float = 0.0           # 0-1
    regret_potential: float = 0.0     # R if wrong (negative)
    position_size_mult: float = 0.0   # 0-1.5

    # Per-dimension scores
    dimensions: Dict[str, float] = field(default_factory=dict)

    # Decision analysis
    probability_win: float = 0.0
    expected_r_if_win: float = 0.0
    expected_r_if_loss: float = -1.0
    kelly_fraction: float = 0.0

    # Reasoning
    reason: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "decision_quality": round(self.decision_quality, 1),
            "expected_value_r": round(self.expected_value_r, 3),
            "confidence": round(self.confidence, 3),
            "regret_potential": round(self.regret_potential, 3),
            "position_size_mult": round(self.position_size_mult, 2),
            "dimensions": {k: round(v, 3) for k, v in self.dimensions.items()},
            "probability_win": round(self.probability_win, 3),
            "expected_r_if_win": round(self.expected_r_if_win, 2),
            "expected_r_if_loss": round(self.expected_r_if_loss, 2),
            "kelly_fraction": round(self.kelly_fraction, 3),
            "reason": self.reason,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


class DecisionIntelligenceLayer:
    """Master decision intelligence layer."""

    # Dimension weights (must sum to 1.0)
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "market_context": 0.15,
        "trend": 0.15,
        "liquidity": 0.15,
        "order_flow": 0.10,
        "volatility": 0.10,
        "correlation": 0.10,
        "execution": 0.10,
        "portfolio_risk": 0.15,
    }

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 min_quality_to_trade: float = 60.0,
                 min_ev_to_trade: float = 0.2,
                 max_regret: float = -2.0):
        """Initialize decision layer.

        Args:
            weights: dimension weights
            min_quality_to_trade: minimum decision quality to trade
            min_ev_to_trade: minimum EV (R) to trade
            max_regret: max acceptable regret (R if wrong)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.min_quality = min_quality_to_trade
        self.min_ev = min_ev_to_trade
        self.max_regret = max_regret

    def decide(self,
               market_context_score: float = 0.5,
               trend_score: float = 0.5,
               liquidity_score: float = 0.5,
               order_flow_score: float = 0.5,
               volatility_score: float = 0.5,
               correlation_score: float = 0.5,
               execution_score: float = 0.5,
               portfolio_risk_score: float = 0.5,
               expected_r_if_win: float = 2.0,
               probability_win: float = 0.5,
               expected_r_if_loss: float = -1.0) -> DecisionResult:
        """Make the final trading decision.

        Args:
            8 dimension scores (0-1 each)
            expected_r_if_win: R if trade wins
            probability_win: P(win)
            expected_r_if_loss: R if trade loses (negative)

        Returns:
            DecisionResult with action + quality + EV
        """
        result = DecisionResult(
            probability_win=probability_win,
            expected_r_if_win=expected_r_if_win,
            expected_r_if_loss=expected_r_if_loss,
        )

        # === Store dimension scores ===
        result.dimensions = {
            "market_context": market_context_score,
            "trend": trend_score,
            "liquidity": liquidity_score,
            "order_flow": order_flow_score,
            "volatility": volatility_score,
            "correlation": correlation_score,
            "execution": execution_score,
            "portfolio_risk": portfolio_risk_score,
        }

        # === Compute decision quality (0-100) ===
        weighted_sum = sum(
            score * self.weights.get(dim, 0.1)
            for dim, score in result.dimensions.items()
        )
        result.decision_quality = weighted_sum * 100

        # === Compute expected value ===
        result.expected_value_r = (
            probability_win * expected_r_if_win +
            (1 - probability_win) * expected_r_if_loss
        )

        # === Regret potential (what we lose if wrong) ===
        result.regret_potential = expected_r_if_loss

        # === Kelly fraction ===
        b = expected_r_if_win / max(abs(expected_r_if_loss), 0.01)
        result.kelly_fraction = max(0, (b * probability_win - (1 - probability_win)) / max(b, 0.01))

        # === Confidence (how aligned are dimensions?) ===
        scores = list(result.dimensions.values())
        result.confidence = 1.0 - (np.std(scores) / 0.5) if len(scores) > 1 else 0.5

        # === Strengths + weaknesses ===
        for dim, score in result.dimensions.items():
            if score >= 0.7:
                result.strengths.append(f"{dim}: {score:.0%}")
            elif score < 0.4:
                result.weaknesses.append(f"{dim}: {score:.0%}")

        # === Decision logic ===
        # SKIP if any critical dimension is too low
        if liquidity_score < 0.2:
            result.action = DecisionAction.SKIP
            result.position_size_mult = 0.0
            result.reason = "SKIP — liquidity too poor"
        elif portfolio_risk_score < 0.2:
            result.action = DecisionAction.SKIP
            result.position_size_mult = 0.0
            result.reason = "SKIP — portfolio risk too high"
        # WAIT if quality or EV insufficient
        elif result.decision_quality < self.min_quality:
            result.action = DecisionAction.WAIT
            result.position_size_mult = 0.0
            result.reason = f"WAIT — quality {result.decision_quality:.0f} < {self.min_quality}"
        elif result.expected_value_r < self.min_ev:
            result.action = DecisionAction.WAIT
            result.position_size_mult = 0.0
            result.reason = f"WAIT — EV {result.expected_value_r:.2f}R < {self.min_ev}R"
        elif result.regret_potential < self.max_regret:
            result.action = DecisionAction.WAIT
            result.position_size_mult = 0.0
            result.reason = f"WAIT — regret {result.regret_potential:.1f}R too high"
        # TRADE
        else:
            result.action = DecisionAction.TRADE
            # Size based on Kelly (capped at 1.5x)
            result.position_size_mult = min(1.5, max(0.25, result.kelly_fraction * 4))
            result.reason = (
                f"TRADE — quality={result.decision_quality:.0f}, "
                f"EV={result.expected_value_r:.2f}R, "
                f"P(win)={probability_win:.0%}"
            )

        return result

    # ------------------------------------------------------------------
    # Decision quality evaluation (separate from outcome)
    # ------------------------------------------------------------------
    def evaluate_decision_quality(self,
                                  decision: DecisionResult,
                                  actual_outcome: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the quality of a decision AFTER the outcome is known.

        Key insight: A good decision can have a bad outcome, and vice versa.
        We score the PROCESS, not the RESULT.

        Args:
            decision: the original DecisionResult
            actual_outcome: "win", "loss", or None (still open)

        Returns:
            Evaluation with process score + outcome + learning
        """
        eval_result = {
            "decision_quality_score": decision.decision_quality,
            "expected_value_r": decision.expected_value_r,
            "actual_outcome": actual_outcome,
            "process_grade": "",
            "was_good_decision": False,
            "learning_note": "",
        }

        # Grade the process
        if decision.decision_quality >= 80:
            eval_result["process_grade"] = "A"
        elif decision.decision_quality >= 65:
            eval_result["process_grade"] = "B"
        elif decision.decision_quality >= 50:
            eval_result["process_grade"] = "C"
        else:
            eval_result["process_grade"] = "D"

        # Was it a good decision? (regardless of outcome)
        # Good decision = high quality + positive EV
        eval_result["was_good_decision"] = (
            decision.decision_quality >= 60 and
            decision.expected_value_r > 0
        )

        # Learning note
        if actual_outcome == "win" and eval_result["was_good_decision"]:
            eval_result["learning_note"] = "Good decision + good outcome — repeat"
        elif actual_outcome == "win" and not eval_result["was_good_decision"]:
            eval_result["learning_note"] = "Bad decision + lucky outcome — don't repeat"
        elif actual_outcome == "loss" and eval_result["was_good_decision"]:
            eval_result["learning_note"] = "Good decision + unlucky outcome — maintain approach"
        elif actual_outcome == "loss" and not eval_result["was_good_decision"]:
            eval_result["learning_note"] = "Bad decision + bad outcome — fix process"
        else:
            eval_result["learning_note"] = "Decision pending outcome"

        return eval_result
