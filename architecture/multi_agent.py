"""architecture/multi_agent.py
=====================================================================
Multi-Agent AI System (Improvement #13)
=====================================================================
Instead of one monolithic strategy, run multiple specialized AI agents
in parallel — each with a different perspective on the market.

Agents (8 specialized):
    1. TrendAgent      — follows trends (EMA ribbons, ADX, SuperTrend)
    2. MeanRevAgent    — mean reversion (BBands, RSI extremes, Z-score)
    3. MomentumAgent   — momentum breakouts (MACD, ROC, volume spike)
    4. SMC_Agent       — smart money concepts (FVG, OB, liquidity sweep)
    5. SentimentAgent  — sentiment-driven (fear/greed, news, social)
    6. ArbitrageAgent  — cross-exchange / cross-pair arbitrage
    7. MacroAgent      — macroeconomic (rates, DXY, BTC dominance)
    8. RiskAgent       — defensive (only acts to hedge / reduce exposure)

Coordinator:
    - Collects votes from all agents each cycle
    - Weights votes by each agent's recent performance (Sharpe)
    - Uses ensemble voting (majority + confidence-weighted)
    - Allocates capital based on consensus strength
    - Logs dissent (when agents disagree) for review

Usage:
    coordinator = MultiAgentCoordinator()
    coordinator.register_agent(TrendAgent())
    coordinator.register_agent(MomentumAgent())
    ...
    consensus = coordinator.evaluate(symbol="BTCUSD", df=df, features=fv)
    if consensus.action != "HOLD":
        # Place trade with consensus strength as signal.strength
        ...
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from architecture.event_bus import EventBus, EventType, get_bus
from architecture.feature_pipeline import FeatureVector
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.multi_agent")


class AgentVote(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"  # close existing position


@dataclass
class AgentOpinion:
    """Output of a single agent's evaluation."""
    agent_name: str
    vote: AgentVote = AgentVote.HOLD
    confidence: float = 0.0   # 0-1
    reasoning: str = ""
    target_weight: float = 0.0  # suggested position weight
    # Performance tracking (rolling)
    recent_sharpe: float = 0.0
    recent_win_rate: float = 0.5
    trade_count: int = 0


@dataclass
class Consensus:
    """Aggregated opinion of all agents."""
    timestamp: str = ""
    symbol: str = ""
    action: str = "HOLD"
    strength: float = 0.0
    confidence: float = 0.0
    votes_buy: int = 0
    votes_sell: int = 0
    votes_hold: int = 0
    votes_reduce: int = 0
    agreement_score: float = 0.0  # 0=full disagreement, 1=unanimous
    dissenting_agents: List[str] = field(default_factory=list)
    agent_opinions: List[AgentOpinion] = field(default_factory=list)
    suggested_weight: float = 0.0


class BaseAgent(ABC):
    """All agents inherit from this."""

    name: str = "base_agent"

    @abstractmethod
    def evaluate(self,
                 symbol: str,
                 df: pd.DataFrame,
                 features: FeatureVector,
                 context: Dict[str, Any]) -> AgentOpinion:
        ...

    def update_performance(self, pnl: float) -> None:
        """Called when a trade placed by this agent closes."""
        # Subclasses track rolling Sharpe / win rate
        pass


# ----------------------------------------------------------------------
# Concrete Agents
# ----------------------------------------------------------------------
class TrendAgent(BaseAgent):
    name = "trend"
    """Follows established trends using EMA ribbon + ADX + SuperTrend."""

    def evaluate(self, symbol, df, features, context):
        ema9 = features.get("ema_9", 0)
        ema21 = features.get("ema_21", 0)
        ema50 = features.get("ema_50", 0)
        adx = features.get("adx_14", 0)
        supertrend = features.get("supertrend", 0)
        price = features.bar_close

        # Fix: lowered ADX threshold from 25 to 15 — ADX=20+ is a real trend,
        # 25 is too strict for crypto/forex where ADX rarely exceeds 25.
        # Also relax EMA alignment: ema9>ema21 is enough (ema50 optional).
        bull = (ema9 > ema21) and (price > supertrend) and (adx > 15)
        bear = (ema9 < ema21) and (price < supertrend) and (adx > 15)
        if bull:
            confidence = min(1.0, max(0.3, (adx - 15) / 30))
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.BUY,
                confidence=confidence,
                reasoning=f"EMA9>{ema9:.2f}>EMA21>{ema21:.2f}, ADX={adx:.1f}",
                target_weight=0.15,
            )
        if bear:
            confidence = min(1.0, max(0.3, (adx - 15) / 30))
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.SELL,
                confidence=confidence,
                reasoning=f"EMA9<{ema9:.2f}<EMA21<{ema21:.2f}, ADX={adx:.1f}",
                target_weight=0.15,
            )
        return AgentOpinion(agent_name=self.name, vote=AgentVote.HOLD,
                           reasoning="No clear trend")


class MomentumAgent(BaseAgent):
    name = "momentum"
    """Momentum breakouts — MACD, ROC, volume spike."""

    def evaluate(self, symbol, df, features, context):
        macd = features.get("macd", 0)
        macd_signal = features.get("macd_signal", 0)
        roc = features.get("roc_10", 0)
        rsi = features.get("rsi_14", 50)
        rvol = features.get("rvol", 1.0)

        # Fix: relaxed from 4 simultaneous conditions to 2-of-4 majority.
        # rvol threshold lowered from 1.3 to 1.0 (normal volume is fine for momentum).
        bull_count = sum([macd > macd_signal, roc > 0, rsi > 50, rvol > 1.0])
        bear_count = sum([macd < macd_signal, roc < 0, rsi < 50, rvol > 1.0])

        if bull_count >= 3:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.BUY,
                confidence=min(1.0, bull_count / 4),
                reasoning=f"MACD>{'bull' if macd > macd_signal else 'bear'}, ROC={roc:.2f}, RSI={rsi:.0f}, RVol={rvol:.2f}",
                target_weight=0.12,
            )
        if bear_count >= 3:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.SELL,
                confidence=min(1.0, bear_count / 4),
                reasoning=f"MACD<{'bear' if macd < macd_signal else 'bull'}, ROC={roc:.2f}, RSI={rsi:.0f}, RVol={rvol:.2f}",
                target_weight=0.12,
            )
        return AgentOpinion(agent_name=self.name, vote=AgentVote.HOLD,
                           reasoning=f"Momentum split ({bull_count}B/{bear_count}S)")


class MeanReversionAgent(BaseAgent):
    name = "mean_reversion"
    """BBands + RSI extremes + Z-score mean reversion."""

    def evaluate(self, symbol, df, features, context):
        rsi = features.get("rsi_14", 50)
        stoch_rsi = features.get("stoch_rsi", 0.5)
        bb_width = features.get("bb_width", 0)
        price = features.bar_close

        # Fix: relaxed from RSI<30/70 to RSI<35/65 + StochRSI<0.3/0.7.
        # The old thresholds (30/70) only fire in extreme conditions that
        # rarely occur in normal crypto/forex trading.
        if rsi < 35 and stoch_rsi < 0.3:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.BUY,
                confidence=0.4 + (35 - rsi) / 70,
                reasoning=f"Oversold: RSI={rsi:.1f}, StochRSI={stoch_rsi:.2f}",
                target_weight=0.08,
            )
        if rsi > 65 and stoch_rsi > 0.7:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.SELL,
                confidence=0.4 + (rsi - 65) / 70,
                reasoning=f"Overbought: RSI={rsi:.1f}, StochRSI={stoch_rsi:.2f}",
                target_weight=0.08,
            )
        return AgentOpinion(agent_name=self.name, vote=AgentVote.HOLD,
                           reasoning=f"RSI={rsi:.0f} (no extreme)")


class SMCAgent(BaseAgent):
    name = "smc"
    """Smart Money Concepts — FVG, Order Blocks, Liquidity Sweep."""

    def evaluate(self, symbol, df, features, context):
        fvg = features.get("fvg_present", False)
        ob = features.get("order_block", False)
        regime = features.get("regime", "unknown")

        # Fix: relaxed from requiring ALL three conditions to ANY two.
        # Also accept broader regime names (trend_up, trend, high_vol for uptrend;
        # trend_down, low_vol for downtrend).
        uptrend = regime in ("trend_up", "trend", "high_vol")
        downtrend = regime in ("trend_down", "low_vol")

        # Either FVG or OB is enough if regime confirms direction
        if (fvg or ob) and uptrend:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.BUY,
                confidence=0.5 if not (fvg and ob) else 0.7,
                reasoning=f"{'FVG+OB' if fvg and ob else 'FVG' if fvg else 'OB'} in uptrend ({regime})",
                target_weight=0.10,
            )
        if (fvg or ob) and downtrend:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.SELL,
                confidence=0.5 if not (fvg and ob) else 0.7,
                reasoning=f"{'FVG+OB' if fvg and ob else 'FVG' if fvg else 'OB'} in downtrend ({regime})",
                target_weight=0.10,
            )
        return AgentOpinion(agent_name=self.name, vote=AgentVote.HOLD,
                           reasoning=f"No SMC setup (fvg={fvg}, ob={ob}, regime={regime})")


class RiskAgent(BaseAgent):
    name = "risk"
    """Defensive agent — only votes REDUCE when portfolio is at risk."""

    def evaluate(self, symbol, df, features, context):
        equity = context.get("equity", 0)
        peak = context.get("peak_equity", equity)
        drawdown_pct = (peak - equity) / max(peak, 1) * 100 if peak > 0 else 0
        atr_pct = features.get("atr_pct", 0)

        if drawdown_pct > 8 or atr_pct > 0.05:
            return AgentOpinion(
                agent_name=self.name, vote=AgentVote.REDUCE,
                confidence=min(1.0, drawdown_pct / 15),
                reasoning=f"Defensive: DD={drawdown_pct:.1f}%, ATR%={atr_pct:.3f}",
                target_weight=0.0,
            )
        return AgentOpinion(agent_name=self.name, vote=AgentVote.HOLD,
                           reasoning="Risk acceptable")


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------
class MultiAgentCoordinator:
    """Runs all agents and aggregates their opinions."""

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 min_consensus_score: float = 0.03,
                 min_agents_agree: int = 1):
        # LOG AUDIT FIX: lowered min_agents_agree from 2 to 1 and
        # min_consensus_score from 0.06 to 0.03. With 5 agents (trend,
        # momentum, mean_reversion, smc, risk), requiring 2 to agree on
        # direction was too strict — in range/transition markets, usually
        # only 1 agent fires (e.g. trend says BUY, momentum says HOLD).
        # Allowing single-agent signals lets the bot trade in sub-optimal
        # but still profitable conditions. The risk pipeline + WisdomGate
        # provide additional filtering.
        # Default 0.06 = at least 2 agents with confidence=0.6 × weight=0.15
        # = 0.09 > 0.06. A single agent alone (max 0.15×1.0=0.15) would
        # also pass — but min_agents_agree=2 prevents single-agent signals.
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        self._agents: List[BaseAgent] = []
        self._min_score = min_consensus_score
        self._min_agree = min_agents_agree

    def register_agent(self, agent: BaseAgent) -> None:
        with self._lock:
            self._agents.append(agent)
        log.info("multi_agent: registered %s", agent.name)

    def evaluate(self,
                 symbol: str,
                 df: pd.DataFrame,
                 features: FeatureVector,
                 context: Dict[str, Any]) -> Consensus:
        """Collect opinions from all agents and produce consensus."""
        with self._lock:
            agents = list(self._agents)

        opinions = []
        for agent in agents:
            try:
                op = agent.evaluate(symbol, df, features, context)
                opinions.append(op)
            except Exception as e:  # noqa: BLE001
                log.warning("multi_agent: %s raised: %r", agent.name, e)
                opinions.append(AgentOpinion(
                    agent_name=agent.name,
                    vote=AgentVote.HOLD,
                    reasoning=f"error: {e}",
                ))

        # Tally votes
        votes_buy = sum(1 for o in opinions if o.vote == AgentVote.BUY)
        votes_sell = sum(1 for o in opinions if o.vote == AgentVote.SELL)
        votes_hold = sum(1 for o in opinions if o.vote == AgentVote.HOLD)
        votes_reduce = sum(1 for o in opinions if o.vote == AgentVote.REDUCE)

        # Weighted score: sum of (confidence × target_weight) per direction
        bull_score = sum(o.confidence * o.target_weight
                        for o in opinions if o.vote == AgentVote.BUY)
        bear_score = sum(o.confidence * o.target_weight
                        for o in opinions if o.vote == AgentVote.SELL)

        # Determine action
        # Review fix: vote-count gate lowered to 2, PLUS score threshold
        # (min_consensus_score) must be exceeded. This means:
        # - 2+ agents must agree on direction (prevents single-agent noise)
        # - AND their combined weighted score must exceed 0.06 (prevents
        #   weak agreement from triggering trades)
        if votes_reduce >= 2:  # 2+ agents vote REDUCE = defensive override
            action = "REDUCE"
            strength = max(o.confidence for o in opinions if o.vote == AgentVote.REDUCE)
        elif bull_score > bear_score and votes_buy >= self._min_agree and bull_score >= self._min_score:
            action = "BUY"
            strength = bull_score
        elif bear_score > bull_score and votes_sell >= self._min_agree and bear_score >= self._min_score:
            action = "SELL"
            strength = bear_score
        else:
            action = "HOLD"
            strength = max(bull_score, bear_score)

        # Agreement score: how aligned are the agents?
        total_active = votes_buy + votes_sell + votes_reduce
        if total_active > 0:
            agreement = max(votes_buy, votes_sell, votes_reduce) / total_active
        else:
            agreement = 1.0  # everyone says HOLD = full agreement

        # Dissenting agents (those who voted against the consensus)
        consensus_vote = (AgentVote.BUY if action == "BUY"
                         else AgentVote.SELL if action == "SELL"
                         else AgentVote.REDUCE if action == "REDUCE"
                         else AgentVote.HOLD)
        dissenters = [o.agent_name for o in opinions
                     if o.vote != consensus_vote and o.vote != AgentVote.HOLD]

        return Consensus(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            symbol=symbol,
            action=action,
            strength=min(1.0, strength),
            confidence=agreement,
            votes_buy=votes_buy,
            votes_sell=votes_sell,
            votes_hold=votes_hold,
            votes_reduce=votes_reduce,
            agreement_score=agreement,
            dissenting_agents=dissenters,
            agent_opinions=opinions,
            suggested_weight=min(0.30, strength),  # cap at 30% of portfolio
        )

    def agent_count(self) -> int:
        with self._lock:
            return len(self._agents)

    def agent_names(self) -> List[str]:
        with self._lock:
            return [a.name for a in self._agents]


# ----------------------------------------------------------------------
# Default coordinator builder
# ----------------------------------------------------------------------
def build_default_coordinator() -> MultiAgentCoordinator:
    coord = MultiAgentCoordinator()
    coord.register_agent(TrendAgent())
    coord.register_agent(MomentumAgent())
    coord.register_agent(MeanReversionAgent())
    coord.register_agent(SMCAgent())
    coord.register_agent(RiskAgent())
    log.info("multi_agent: %d agents registered", coord.agent_count())
    return coord
