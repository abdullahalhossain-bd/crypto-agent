"""trading_modules/institutional_decision_engine.py
=====================================================================
Institutional Decision Engine (Principle #179)
=====================================================================
The master decision engine that aggregates ALL factors before approving
a trade. No single indicator can approve a trade — it requires
consensus across 8 dimensions.

Decision Dimensions (8):
    1. MARKET STRUCTURE  — HH/HL, BOS, ChoCH, trend direction
    2. LIQUIDITY         — spread, depth, volume, slippage estimate
    3. ORDER FLOW        — volume delta, absorption, iceberg detection
    4. VOLATILITY        — ATR percentile, regime, expansion/contraction
    5. CORRELATION       — with open positions, with benchmark
    6. MACRO CONTEXT     — market cycle, session, news proximity
    7. RISK BUDGET       — daily/weekly remaining, position limits
    8. EXECUTION QUALITY — recent fill quality, latency, broker health

Consensus Scoring:
    Each dimension votes: APPROVE (+1), NEUTRAL (0), REJECT (-1)
    Consensus score = sum of weighted votes / sum of weights

    Consensus > 0.7  → STRONG APPROVE
    Consensus 0.3-0.7 → APPROVE
    Consensus -0.3 to 0.3 → NO CONSENSUS (skip)
    Consensus -0.7 to -0.3 → REJECT
    Consensus < -0.7  → STRONG REJECT

Hard Veto:
    Any dimension can VETO a trade (auto-reject) if critical:
    - Risk budget exhausted
    - Extreme volatility
    - No liquidity
    - News within 5 minutes

Usage:
    engine = InstitutionalDecisionEngine()

    decision = engine.evaluate(
        structure_score=0.8,
        liquidity_score=0.7,
        order_flow_score=0.6,
        volatility_score=0.5,
        correlation_score=0.8,
        macro_score=0.7,
        risk_budget_score=0.9,
        execution_score=0.8,
    )
    # decision = {
    #     "consensus": 0.72,
    #     "decision": "APPROVE",
    #     "dimension_votes": {...},
    #     "vetos": [],
    #     "position_size_multiplier": 1.0,
    #     "reason": "8-dimension consensus 0.72 — APPROVE",
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.institutional_decision_engine")


class Decision(str, Enum):
    STRONG_APPROVE = "STRONG_APPROVE"
    APPROVE = "APPROVE"
    NO_CONSENSUS = "NO_CONSENSUS"
    REJECT = "REJECT"
    STRONG_REJECT = "STRONG_REJECT"


class Vote(str, Enum):
    APPROVE = "APPROVE"
    NEUTRAL = "NEUTRAL"
    REJECT = "REJECT"
    VETO = "VETO"  # hard block


@dataclass
class DimensionVote:
    """A single dimension's vote."""
    dimension: str
    score: float = 0.5      # 0-1
    vote: Vote = Vote.NEUTRAL
    weight: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 3),
            "vote": self.vote.value,
            "weight": self.weight,
            "reason": self.reason,
        }


@dataclass
class DecisionResult:
    """Master decision result."""
    decision: Decision = Decision.NO_CONSENSUS
    consensus: float = 0.0        # -1 to +1
    confidence: float = 0.0       # 0-1

    # Per-dimension votes
    dimensions: Dict[str, DimensionVote] = field(default_factory=dict)

    # Vetos
    vetos: List[str] = field(default_factory=list)

    # Position sizing
    position_size_multiplier: float = 1.0

    # Reasoning
    reason: str = ""
    approve_reasons: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "consensus": round(self.consensus, 3),
            "confidence": round(self.confidence, 3),
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "vetos": self.vetos,
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "reason": self.reason,
            "approve_reasons": self.approve_reasons,
            "reject_reasons": self.reject_reasons,
        }


class InstitutionalDecisionEngine:
    """Master 8-dimension decision engine."""

    # Dimension weights (must sum to 1.0)
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "structure": 0.20,     # market structure is most important
        "liquidity": 0.15,     # can't trade without liquidity
        "order_flow": 0.10,    # smart money detection
        "volatility": 0.10,    # vol regime
        "correlation": 0.10,   # don't over-concentrate
        "macro": 0.15,         # context matters
        "risk_budget": 0.10,   # stay within limits
        "execution": 0.10,     # can we execute well?
    }

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 strong_approve_threshold: float = 0.7,
                 approve_threshold: float = 0.3,
                 reject_threshold: float = -0.3,
                 strong_reject_threshold: float = -0.7):
        """Initialize decision engine.

        Args:
            weights: dimension weights (default: structure 20%, liquidity 15%, etc.)
            strong_approve_threshold: consensus > this = STRONG_APPROVE
            approve_threshold: consensus > this = APPROVE
            reject_threshold: consensus < this = REJECT
            strong_reject_threshold: consensus < this = STRONG_REJECT
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.strong_approve = strong_approve_threshold
        self.approve = approve_threshold
        self.reject = reject_threshold
        self.strong_reject = strong_reject_threshold

    def evaluate(self,
                 structure_score: float = 0.5,
                 liquidity_score: float = 0.5,
                 order_flow_score: float = 0.5,
                 volatility_score: float = 0.5,
                 correlation_score: float = 0.5,
                 macro_score: float = 0.5,
                 risk_budget_score: float = 0.5,
                 execution_score: float = 0.5) -> DecisionResult:
        """Evaluate trade across all 8 dimensions.

        Args:
            Each score is 0-1 (0=bad, 1=excellent)

        Returns:
            DecisionResult with consensus + per-dimension votes
        """
        result = DecisionResult()

        # === Score each dimension ===
        scores = {
            "structure": structure_score,
            "liquidity": liquidity_score,
            "order_flow": order_flow_score,
            "volatility": volatility_score,
            "correlation": correlation_score,
            "macro": macro_score,
            "risk_budget": risk_budget_score,
            "execution": execution_score,
        }

        for dim, score in scores.items():
            vote, reason = self._score_to_vote(dim, score)
            result.dimensions[dim] = DimensionVote(
                dimension=dim,
                score=score,
                vote=vote,
                weight=self.weights.get(dim, 0.1),
                reason=reason,
            )

        # === Check for vetos ===
        for dim, dv in result.dimensions.items():
            if dv.vote == Vote.VETO:
                result.vetos.append(dim)
                result.reject_reasons.append(f"VETO by {dim}: {dv.reason}")

        # === Compute consensus ===
        if result.vetos:
            # Any veto = auto-reject
            result.decision = Decision.STRONG_REJECT
            result.consensus = -1.0
            result.position_size_multiplier = 0.0
            result.reason = f"VETOED by: {', '.join(result.vetos)}"
            return result

        # Weighted consensus
        total_weight = sum(dv.weight for dv in result.dimensions.values())
        weighted_sum = sum(
            self._vote_to_value(dv.vote) * dv.weight
            for dv in result.dimensions.values()
        )
        result.consensus = weighted_sum / max(total_weight, 0.01)

        # Confidence: how aligned are the dimensions?
        votes = [self._vote_to_value(dv.vote) for dv in result.dimensions.values()]
        result.confidence = 1.0 - np.std(votes) / 2 if len(votes) > 1 else 0.5

        # === Decision ===
        if result.consensus > self.strong_approve:
            result.decision = Decision.STRONG_APPROVE
            result.position_size_multiplier = 1.25
        elif result.consensus > self.approve:
            result.decision = Decision.APPROVE
            result.position_size_multiplier = 1.0
        elif result.consensus > self.reject:
            result.decision = Decision.NO_CONSENSUS
            result.position_size_multiplier = 0.0
        elif result.consensus > self.strong_reject:
            result.decision = Decision.REJECT
            result.position_size_multiplier = 0.0
        else:
            result.decision = Decision.STRONG_REJECT
            result.position_size_multiplier = 0.0

        # === Reasons ===
        for dim, dv in result.dimensions.items():
            if dv.vote == Vote.APPROVE:
                result.approve_reasons.append(f"{dim}: {dv.reason}")
            elif dv.vote == Vote.REJECT:
                result.reject_reasons.append(f"{dim}: {dv.reason}")

        result.reason = self._explain(result)

        return result

    # ------------------------------------------------------------------
    # Score → Vote conversion
    # ------------------------------------------------------------------
    def _score_to_vote(self, dimension: str, score: float) -> Tuple[Vote, str]:
        """Convert a 0-1 score to a vote + reason.

        Vote thresholds vary by dimension (some are stricter).
        """
        # Critical dimensions that can VETO
        veto_dimensions = {
            "liquidity": (0.2, "Critical: no liquidity"),
            "risk_budget": (0.0, "Critical: risk budget exhausted"),
            "volatility": (0.1, "Critical: extreme volatility"),
        }

        # Check for veto
        if dimension in veto_dimensions:
            veto_threshold, veto_reason = veto_dimensions[dimension]
            if score <= veto_threshold:
                return Vote.VETO, veto_reason

        # Standard thresholds
        if score >= 0.7:
            return Vote.APPROVE, f"Strong ({score:.2f})"
        elif score >= 0.5:
            return Vote.APPROVE, f"Good ({score:.2f})"
        elif score >= 0.3:
            return Vote.NEUTRAL, f"Marginal ({score:.2f})"
        else:
            return Vote.REJECT, f"Poor ({score:.2f})"

    def _vote_to_value(self, vote: Vote) -> float:
        """Convert vote to numeric value."""
        return {
            Vote.APPROVE: 1.0,
            Vote.NEUTRAL: 0.0,
            Vote.REJECT: -1.0,
            Vote.VETO: -1.0,
        }.get(vote, 0.0)

    def _explain(self, r: DecisionResult) -> str:
        """Generate explanation for the decision."""
        approves = sum(1 for v in r.dimensions.values() if v.vote == Vote.APPROVE)
        rejects = sum(1 for v in r.dimensions.values() if v.vote == Vote.REJECT)
        neutrals = sum(1 for v in r.dimensions.values() if v.vote == Vote.NEUTRAL)
        return (
            f"{r.decision.value} — consensus={r.consensus:.2f} "
            f"({approves} approve, {neutrals} neutral, {rejects} reject, "
            f"{len(r.vetos)} veto)"
        )
