"""enhancements package — Phase 8 institutional enhancements (Day 141-160).

Fills the remaining gaps between "trading platform" and "production
trading desk":

  - position_manager     : trailing stops, breakeven, partial profits, scaling
  - notification_system  : Slack / Discord / email / webhook alerts
  - trade_journal        : post-trade analysis + lessons learned
  - monte_carlo          : 1000s of alternative equity curves
  - walk_forward_optimizer : strategy parameter optimization
  - news_calendar        : economic events calendar integration
  - benchmarker          : compare vs buy-and-hold + benchmarks
  - trade_replay         : forensic decision replay
  - strategy_ensemble    : intelligent multi-strategy signal combination
  - drawdown_recovery    : post-drawdown trading protocol
  - trading_as_git       : stage/commit/push trade operations (OpenAlice)
  - sector_rotation      : cross-sectional crypto sector momentum (OpenAlice)
  - as_of_snapshot       : point-in-time no-lookahead market snapshot (OpenAlice)
  - trade_simulator      : what-if entry→exit path-dependent backtest (OpenAlice)
  - ai_agent_tools       : AI-agent-callable trading + analysis tools (OpenAlice)
  - rss_news_collector   : periodic RSS feed ingestion (OpenAlice)
"""
from enhancements.position_manager import (  # noqa: F401
    PositionManager, PositionAction, TrailingStopType,
)
from enhancements.notification_system import (  # noqa: F401
    NotificationSystem, NotificationChannel, NotificationMessage,
)
from enhancements.trade_journal import (  # noqa: F401
    TradeJournal, JournalEntry, JournalAnalysis,
)
from enhancements.monte_carlo import (  # noqa: F401
    MonteCarloSimulator, MonteCarloResult,
)
from enhancements.walk_forward_optimizer import (  # noqa: F401
    WalkForwardOptimizer, OptimizationResult,
)
from enhancements.news_calendar import (  # noqa: F401
    NewsCalendar, NewsEvent, NewsImpact,
)
from enhancements.benchmarker import (  # noqa: F401
    Benchmarker, BenchmarkResult,
)
from enhancements.trade_replay import (  # noqa: F401
    TradeReplay, ReplayResult,
)
from enhancements.strategy_ensemble import (  # noqa: F401
    StrategyEnsemble, EnsembleResult, CombinationMethod,
)
from enhancements.drawdown_recovery import (  # noqa: F401
    DrawdownRecoveryProtocol, RecoveryState, RecoveryPhase,
)
from enhancements.trading_as_git import (  # noqa: F401
    TradingGit, Operation, OperationAction, OperationStatus, CommitEntry,
)
from enhancements.sector_rotation import (  # noqa: F401
    SectorRotationAnalyzer, SectorRotationRow, CRYPTO_SECTORS,
)
from enhancements.as_of_snapshot import (  # noqa: F401
    AsOfSnapshot, SnapshotResult, SnapshotBar,
)
from enhancements.trade_simulator import (  # noqa: F401
    TradeSimulator, SimulateResult,
)
from enhancements.ai_agent_tools import (  # noqa: F401
    AIAgentTools, ToolResult,
)
from enhancements.rss_news_collector import (  # noqa: F401
    RSSNewsCollector, NewsItem,
)

__all__ = [
    "PositionManager", "PositionAction", "TrailingStopType",
    "NotificationSystem", "NotificationChannel", "NotificationMessage",
    "TradeJournal", "JournalEntry", "JournalAnalysis",
    "MonteCarloSimulator", "MonteCarloResult",
    "WalkForwardOptimizer", "OptimizationResult",
    "NewsCalendar", "NewsEvent", "NewsImpact",
    "Benchmarker", "BenchmarkResult",
    "TradeReplay", "ReplayResult",
    "StrategyEnsemble", "EnsembleResult", "CombinationMethod",
    "DrawdownRecoveryProtocol", "RecoveryState", "RecoveryPhase",
    "TradingGit", "Operation", "OperationAction", "OperationStatus", "CommitEntry",
    "SectorRotationAnalyzer", "SectorRotationRow", "CRYPTO_SECTORS",
    "AsOfSnapshot", "SnapshotResult", "SnapshotBar",
    "TradeSimulator", "SimulateResult",
    "AIAgentTools", "ToolResult",
    "RSSNewsCollector", "NewsItem",
]
