"""agents package — Multi-Agent Trading Framework (inspired by TradingAgents).

Deploys specialized LLM-powered agents that collaborate like a real
trading firm:
  - Analyst Team   : fundamentals, news, sentiment, technical
  - Research Team   : bull vs bear debate
  - Research Manager: synthesizes debate into investment plan
  - Trader          : turns research into transaction proposal
  - Risk Team       : aggressive / conservative / neutral debate
  - Portfolio Mgr   : final approve/reject decision
  - Memory Log      : decision log with deferred reflection

Unlike TradingAgents (which uses LangGraph + LangChain), we use a
lightweight pure-Python graph — no external graph framework needed.
The LLM calls go through our existing external.llm_provider.
"""
from agents.schemas import (  # noqa: F401
    PortfolioRating, TraderAction, ResearchPlan, TraderProposal,
    PortfolioDecision, parse_rating,
)
from agents.analysts import (  # noqa: F401
    AnalystTeam, FundamentalsAnalyst, NewsAnalyst,
    SentimentAnalyst, TechnicalAnalyst, AnalystReport,
)
from agents.researchers import BullBearDebate, DebateResult  # noqa: F401
from agents.research_manager import ResearchManager  # noqa: F401
from agents.risk_debators import RiskDebate, RiskDebateResult  # noqa: F401
from agents.trader import Trader  # noqa: F401
from agents.portfolio_manager import PortfolioManager  # noqa: F401
from agents.memory_log import TradingMemoryLog, MemoryEntry  # noqa: F401
from agents.agent_graph import MultiAgentTradingGraph, TradingGraphResult  # noqa: F401

__all__ = [
    "PortfolioRating", "TraderAction", "ResearchPlan", "TraderProposal",
    "PortfolioDecision", "parse_rating",
    "AnalystTeam", "FundamentalsAnalyst", "NewsAnalyst",
    "SentimentAnalyst", "TechnicalAnalyst", "AnalystReport",
    "BullBearDebate", "DebateResult",
    "ResearchManager",
    "RiskDebate", "RiskDebateResult",
    "Trader", "PortfolioManager",
    "TradingMemoryLog", "MemoryEntry",
    "MultiAgentTradingGraph", "TradingGraphResult",
]
