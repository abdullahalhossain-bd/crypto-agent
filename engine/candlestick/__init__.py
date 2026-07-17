"""engine.candlestick package — Candlestick Confluence Layer (Day 126-140).

Inspired by The Candlestick Trading Bible. EVERY concept here is a
feature source, NOT a trading rule. The ML layer / confluence engine
decides weights; nothing in this package emits BUY/SELL on its own.

Modules:
  - pattern_detector     : pin bar / inside bar / engulfing / doji / hammer
  - rejection_strength   : wick/body/ATR/volume scoring
  - market_state         : TREND / RANGE / CHOPPY classifier
  - confluence_engine    : weighted multi-factor final confidence
  - pullback_detector    : pullback-within-trend detection
  - false_breakout       : false breakout probability
  - sr_confidence        : support/resistance confidence scoring
  - pattern_statistics   : per-pattern historical stats DB
  - pattern_decay        : per-pattern win-rate decay tracking
  - trade_quality        : A+ / A / B / C / Reject grading
  - entry_style          : aggressive / conservative / adaptive
  - multi_timeframe      : D1 → H4 → H1 → M15 confirmation
  - candlestick_features : 25+ features for the ML feature store
"""
from engine.candlestick.pattern_detector import (  # noqa: F401
    PatternDetector, PatternResult, PatternType, TREND_REQUIREMENTS,
)
from engine.candlestick.rejection_strength import (  # noqa: F401
    RejectionStrengthScorer, RejectionScore,
)
from engine.candlestick.market_state import (  # noqa: F401
    MarketStateClassifier, MarketState,
)
from engine.candlestick.confluence_engine import (  # noqa: F401
    ConfluenceEngine, ConfluenceResult, ConfluenceFactor,
)
from engine.candlestick.pullback_detector import (  # noqa: F401
    PullbackDetector, PullbackResult,
)
from engine.candlestick.false_breakout import (  # noqa: F401
    FalseBreakoutDetector, FalseBreakoutResult,
)
from engine.candlestick.sr_confidence import (  # noqa: F401
    SupportResistanceConfidence, SRLevel, SRConfidenceResult,
)
from engine.candlestick.pattern_statistics import (  # noqa: F401
    PatternStatisticsDB, PatternStats,
)
from engine.candlestick.pattern_decay import (  # noqa: F401
    PatternDecayTracker, PatternDecayReport,
)
from engine.candlestick.trade_quality import (  # noqa: F401
    TradeQualityScorer, TradeQuality, QualityGrade,
)
from engine.candlestick.entry_style import (  # noqa: F401
    EntryStyleSelector, EntryStyle,
)
from engine.candlestick.multi_timeframe import (  # noqa: F401
    MultiTimeframeConfirmator, MTFResult,
)
from engine.candlestick.candlestick_features import (  # noqa: F401
    CandlestickFeatureExtractor,
)

__all__ = [
    "PatternDetector", "PatternResult", "PatternType",
    "RejectionStrengthScorer", "RejectionScore",
    "MarketStateClassifier", "MarketState",
    "ConfluenceEngine", "ConfluenceResult", "ConfluenceFactor",
    "PullbackDetector", "PullbackResult",
    "FalseBreakoutDetector", "FalseBreakoutResult",
    "SupportResistanceConfidence", "SRLevel", "SRConfidenceResult",
    "PatternStatisticsDB", "PatternStats",
    "PatternDecayTracker", "PatternDecayReport",
    "TradeQualityScorer", "TradeQuality", "QualityGrade",
    "EntryStyleSelector", "EntryStyle",
    "MultiTimeframeConfirmator", "MTFResult",
    "CandlestickFeatureExtractor",
]
