"""
Swarm Intelligence — Multi-Strategy Cooperation
=================================================

Multiple independent strategies "vote" on each trade.
The swarm's collective decision is often better than any individual.

Each strategy agent:
  1. Independently evaluates the market
  2. Votes (BUY/SELL/HOLD) with confidence
  3. Swarm aggregates votes weighted by recent accuracy
  4. Consensus threshold required for execution

Source: Orallexa (review #27) — 20-agent micro swarm
        NexusQuant (review #29) — opportunity scoring

Usage:
    from trading_modules.swarm_intelligence import StrategySwarm

    swarm = StrategySwarm()

    # Register strategy agents
    swarm.register("trend_follower", trend_strategy_fn)
    swarm.register("mean_reverter", mean_rev_strategy_fn)
    swarm.register("breakout_hunter", breakout_strategy_fn)

    # Get swarm decision
    decision = swarm.vote(df, symbol="BTCUSD")
    if decision.action == "BUY" and decision.confidence > 0.7:
        execute_trade()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict, deque

@dataclass
class AgentVote:
    """A single agent's vote."""
    agent_name: str
    action: str  # "BUY" / "SELL" / "HOLD"
    confidence: float  # 0-1
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {"agent": self.agent_name, "action": self.action, "confidence": round(self.confidence, 2)}


@dataclass
class SwarmDecision:
    """Aggregated swarm decision."""
    action: str = "HOLD"
    confidence: float = 0.0
    consensus: float = 0.0  # 0-1, how much agents agree
    votes: list = field(default_factory=list)
    buy_votes: int = 0
    sell_votes: int = 0
    hold_votes: int = 0
    weighted_score: float = 0.0  # -1 to +1

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "consensus": round(self.consensus, 4),
            "buy_votes": self.buy_votes,
            "sell_votes": self.sell_votes,
            "hold_votes": self.hold_votes,
            "weighted_score": round(self.weighted_score, 4),
            "votes": [v.to_dict() for v in self.votes],
        }


class StrategySwarm:
    """
    Multi-strategy swarm intelligence.

    Each agent votes independently. Swarm aggregates using:
      - Accuracy-weighted voting (better agents get more weight)
      - Consensus threshold (need >60% agreement to act)
      - Confidence-weighted score

    Agents that are consistently wrong get their weight reduced.
    """

    CONSENSUS_THRESHOLD = 0.60  # 60% agreement needed to execute
    MIN_AGENTS = 2              # Need at least 2 agents to vote

    def __init__(self):
        self._agents: dict[str, Callable] = {}
        self._agent_accuracy: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._agent_weights: dict[str, float] = {}

    def register(self, name: str, strategy_fn: Callable[[pd.DataFrame, str], AgentVote]) -> None:
        """Register a strategy agent."""
        self._agents[name] = strategy_fn
        self._agent_weights[name] = 1.0  # Start with equal weight

    def vote(self, df: pd.DataFrame, symbol: str = "") -> SwarmDecision:
        """
        Collect votes from all agents and aggregate.

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            SwarmDecision with aggregated result
        """
        if len(self._agents) < self.MIN_AGENTS:
            return SwarmDecision()

        # Collect votes
        votes = []
        for name, fn in self._agents.items():
            try:
                vote = fn(df, symbol)
                votes.append(vote)
            except Exception as e:
                votes.append(AgentVote(agent_name=name, action="HOLD", confidence=0.0,
                                       reasoning=f"Error: {e}"))

        # Calculate weighted scores
        buy_weight = sell_weight = hold_weight = 0.0
        total_weight = 0.0

        for vote in votes:
            weight = self._agent_weights.get(vote.agent_name, 1.0) * vote.confidence
            total_weight += weight

            if vote.action == "BUY":
                buy_weight += weight
            elif vote.action == "SELL":
                sell_weight += weight
            else:
                hold_weight += weight

        # Weighted score: -1 (all sell) to +1 (all buy)
        if total_weight > 0:
            weighted_score = (buy_weight - sell_weight) / total_weight
        else:
            weighted_score = 0.0

        # Count votes
        buy_count = sum(1 for v in votes if v.action == "BUY")
        sell_count = sum(1 for v in votes if v.action == "SELL")
        hold_count = sum(1 for v in votes if v.action == "HOLD")

        # Consensus: fraction of agents agreeing on majority direction
        max_votes = max(buy_count, sell_count, hold_count)
        consensus = max_votes / len(votes) if votes else 0

        # Decision
        if buy_count > sell_count and consensus >= self.CONSENSUS_THRESHOLD:
            action = "BUY"
            confidence = buy_weight / total_weight if total_weight > 0 else 0
        elif sell_count > buy_count and consensus >= self.CONSENSUS_THRESHOLD:
            action = "SELL"
            confidence = sell_weight / total_weight if total_weight > 0 else 0
        else:
            action = "HOLD"
            confidence = hold_weight / total_weight if total_weight > 0 else 0.5

        return SwarmDecision(
            action=action,
            confidence=float(confidence),
            consensus=float(consensus),
            votes=votes,
            buy_votes=buy_count,
            sell_votes=sell_count,
            hold_votes=hold_count,
            weighted_score=float(weighted_score),
        )

    def record_outcome(self, agent_name: str, was_correct: bool) -> None:
        """Record whether an agent's vote was correct."""
        self._agent_accuracy[agent_name].append(1 if was_correct else 0)

        # Update weight based on rolling accuracy
        history = self._agent_accuracy[agent_name]
        if len(history) >= 10:
            accuracy = sum(history) / len(history)
            # Weight = accuracy (0.3 = 0.3x weight, 0.9 = 0.9x weight)
            self._agent_weights[agent_name] = max(0.1, min(2.0, accuracy * 2))

    def get_agent_status(self) -> dict:
        """Get status of all agents."""
        status = {}
        for name in self._agents:
            history = self._agent_accuracy[name]
            accuracy = sum(history) / len(history) if len(history) > 0 else 0
            status[name] = {
                "weight": round(self._agent_weights.get(name, 1.0), 4),
                "accuracy": round(accuracy, 4),
                "n_predictions": len(history),
            }
        return status

    def get_summary(self) -> dict:
        """Get swarm summary."""
        return {
            "n_agents": len(self._agents),
            "agents": list(self._agents.keys()),
            "agent_status": self.get_agent_status(),
            "consensus_threshold": self.CONSENSUS_THRESHOLD,
        }
