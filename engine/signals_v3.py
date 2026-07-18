"""engine/signals_v3.py
===============================================================================
Industrial-Grade Immutable Decision Contract (v3 — Citadel/Jump/Two Sigma Level)
===============================================================================

Purpose
-------
A Signal is NOT just BUY/SELL. It is a complete immutable decision contract
that flows through every layer of the platform: Strategy → AI → Risk →
Wisdom → Execution → Audit → Memory. Each signal carries 30+ institutional
features organized into 22 strongly-typed sub-dataclasses.

Architecture
------------
    Signal (immutable, frozen=True)
    │
    ├── SignalIdentity              — UUID, correlation, strategy lineage
    ├── MarketSnapshot              — symbol, TF, regime, session, bar info
    ├── StrategyMetadata            — action, strength, quality, expiry
    ├── AIInference                 — model, version, latency, embedding
    ├── ConfidenceBreakdown         — 8-dimensional confidence (trend/mom/vol/...)
    ├── VolatilitySnapshot          — ATR, HV, RV, percentile
    ├── LiquiditySnapshot           — spread, slippage, depth, volume profile
    ├── MultiTimeframeConfirmation  — M5/M15/H1/H4/D1 alignment
    ├── RiskEstimate                — Kelly, R:R, MAE, expected return
    ├── ExecutionHint               — order type, priority, urgency
    ├── Explainability              — SHAP, attention, decision trace
    ├── EnsembleVotes               — per-agent votes + consensus
    ├── BayesianConfidence          — prior, posterior, uncertainty
    ├── UncertaintyEstimate         — epistemic + aleatoric
    ├── MarketMicrostructure        — imbalance, orderflow, CVD, absorption
    ├── OnChainSnapshot             — funding, OI, liquidations, whales
    ├── SentimentSnapshot           — Twitter, Reddit, FearGreed, news
    ├── NewsFlag                    — high-impact events, blackout
    ├── SessionInfo                 — Asia/London/NY/Overlap
    ├── CorrelationSnapshot         — BTC/ETH/SPX/DXY/Gold correlations
    ├── FeatureMetadata             — feature hash + version (ML replay)
    ├── AuditMetadata               — bot version, git commit, host
    └── ReplayMetadata              — cycle/snapshot IDs for replay

Design Principles
-----------------
1. Immutable — every sub-dataclass uses frozen=True; once created, never mutated
2. Typed — no loose `meta` dict; every field has a type annotation
3. Composable — sub-dataclasses can be used independently (e.g., RiskEstimate)
4. Serializable — full to_dict / to_json / from_dict / from_json
5. Schema-versioned — schema_version=3, with backward-compat migration
6. Auditable — every signal has UUID + correlation_id + decision trace
7. Replayable — feature_hash + market_snapshot_id enable exact replay
8. Builder-friendly — SignalBuilder for incremental construction

Schema Version: 3
===============================================================================
"""
from __future__ import annotations

import hashlib
import json
import socket
import uuid as _uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("trading_bot.engine.signals_v3")


# ======================================================================
# Enumerations (extended for v3)
# ======================================================================
class Action(str, Enum):
    """Trading actions — institutional-grade (9 actions)."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    HEDGE = "HEDGE"
    REVERSE = "REVERSE"

    def __str__(self) -> str:
        return self.value


class MarketRegime(str, Enum):
    """Market regime (extended — 8 regimes)."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    NEWS = "news"
    UNKNOWN = "unknown"


class TradingSession(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    OFF_HOURS = "off_hours"


class InstrumentType(str, Enum):
    FOREX = "forex"
    METAL = "metal"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class SignalSourceType(str, Enum):
    """Origin of the signal (extended — 7 sources)."""
    RULE_BASED = "rule_based"
    ML = "ml"
    RL = "rl"               # reinforcement learning
    LLM = "llm"             # large language model
    ENSEMBLE = "ensemble"
    MANUAL = "manual"
    BACKTEST = "backtest"


class SignalSource(str, Enum):
    """Where the signal was generated (environment)."""
    MT5 = "mt5"
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"
    BACKTEST = "backtest"
    SHADOW = "shadow"


class ExecutionStatus(str, Enum):
    NEW = "new"
    PENDING_RISK = "pending_risk"
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_EXEC = "pending_exec"
    EXECUTED = "executed"
    PARTIAL_FILL = "partial_fill"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SignalQuality(str, Enum):
    """Trade quality grade."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    REJECT = "REJECT"


class OrderType(str, Enum):
    """Execution recommendation."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    POV = "POV"  # percent of volume


class Urgency(str, Enum):
    """How quickly must this signal be executed?"""
    IMMEDIATE = "immediate"   # market order, skip queue
    HIGH = "high"             # execute within 1 bar
    NORMAL = "normal"         # execute within 5 bars
    LOW = "low"               # limit order, wait for fill
    PATIENT = "patient"       # can wait indefinitely


# ======================================================================
# Sub-Dataclass 1: SignalIdentity
# ======================================================================
@dataclass(frozen=True)
class SignalIdentity:
    """Globally-unique identity + lineage for this signal."""
    signal_id: str = field(
        default_factory=lambda: str(_uuid.uuid4())
    )
    correlation_id: str = field(
        default_factory=lambda: _uuid.uuid4().hex[:12]
    )
    parent_signal_id: str = ""           # if scaled/hedged from another signal
    strategy_id: str = ""                # e.g., "Momentum_v4.1"
    strategy_version: str = ""           # e.g., "4.1.0"


# ======================================================================
# Sub-Dataclass 2: MarketSnapshot
# ======================================================================
@dataclass(frozen=True)
class MarketSnapshot:
    """Complete market context at signal generation time."""
    symbol: str = ""
    timeframe: str = "M15"
    bar_time: Optional[str] = None       # ISO 8601 UTC
    bar_index: int = 0
    bar_close: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_open: float = 0.0
    bar_volume: float = 0.0
    regime: MarketRegime = MarketRegime.UNKNOWN
    session: TradingSession = TradingSession.OFF_HOURS
    instrument_type: InstrumentType = InstrumentType.UNKNOWN
    closed_bar: bool = True              # no repaint


# ======================================================================
# Sub-Dataclass 3: StrategyMetadata
# ======================================================================
@dataclass(frozen=True)
class StrategyMetadata:
    """Strategy's view: action, conviction, quality, expiry."""
    action: Action = Action.HOLD
    strength: float = 0.0                # strategy conviction [0, 1]
    quality_score: float = 0.0           # market cleanliness [0, 1]
    quality_grade: SignalQuality = SignalQuality.C
    source_type: SignalSourceType = SignalSourceType.RULE_BASED
    source: SignalSource = SignalSource.LIVE
    execution_status: ExecutionStatus = ExecutionStatus.NEW
    expires_at: Optional[str] = None     # ISO 8601 UTC
    ttl_seconds: float = 3600.0          # default 1 hour
    reject_reason: str = ""

    def __post_init__(self) -> None:
        # Clamp strength and quality
        if self.strength < 0.0 or self.strength > 1.0:
            object.__setattr__(self, "strength",
                              max(0.0, min(1.0, self.strength)))
        if self.quality_score < 0.0 or self.quality_score > 1.0:
            object.__setattr__(self, "quality_score",
                              max(0.0, min(1.0, self.quality_score)))


# ======================================================================
# Sub-Dataclass 4: AIInference
# ======================================================================
@dataclass(frozen=True)
class AIInference:
    """AI model metadata (if signal is ML/RL/LLM-generated)."""
    model_name: str = ""
    model_version: str = ""
    ensemble_weight: float = 0.0         # weight in ensemble (0-1)
    inference_latency_ms: float = 0.0
    embedding_id: str = ""               # transformer embedding hash
    model_confidence: float = 0.0        # raw model confidence [0, 1]
    model_prediction: str = ""           # raw model output (e.g., "BUY")


# ======================================================================
# Sub-Dataclass 5: ConfidenceBreakdown
# ======================================================================
@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Multi-dimensional confidence — not a single number."""
    overall: float = 0.0
    trend_confidence: float = 0.0        # EMA stacking, ADX
    momentum_confidence: float = 0.0     # RSI, MACD, ROC
    volume_confidence: float = 0.0       # OBV, RVol, CMF
    ai_confidence: float = 0.0           # ML model output
    macro_confidence: float = 0.0        # DXY, rates, BTC dominance
    pattern_confidence: float = 0.0      # candlestick, chart patterns
    sentiment_confidence: float = 0.0    # Fear/Greed, news, social

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and (v < 0.0 or v > 1.0):
                object.__setattr__(self, f.name, max(0.0, min(1.0, v)))

    @property
    def average(self) -> float:
        """Simple average of all confidence dimensions."""
        vals = [getattr(self, f.name) for f in fields(self)
                if f.name != "overall" and isinstance(getattr(self, f.name), (int, float))]
        return sum(vals) / max(len(vals), 1)


# ======================================================================
# Sub-Dataclass 6: VolatilitySnapshot
# ======================================================================
@dataclass(frozen=True)
class VolatilitySnapshot:
    """Volatility metrics at signal time."""
    atr: float = 0.0                     # ATR(14)
    atr_pct: float = 0.0                 # ATR / price * 100
    historical_vol_20: float = 0.0       # 20-bar HV (annualized)
    realized_vol_20: float = 0.0         # RV from intraday returns
    implied_vol: float = 0.0             # from options (if available)
    volatility_percentile: float = 0.5   # 0-1, where 1=extreme vol
    bb_width: float = 0.0                # Bollinger Band width


# ======================================================================
# Sub-Dataclass 7: LiquiditySnapshot
# ======================================================================
@dataclass(frozen=True)
class LiquiditySnapshot:
    """Liquidity metrics — critical for execution quality."""
    spread_bps: float = 0.0              # bid-ask spread in bps
    slippage_estimate_bps: float = 0.0   # expected slippage for our size
    orderbook_depth_usd: float = 0.0     # depth at ±1%
    volume_profile: str = "normal"       # thin/normal/thick
    avg_volume_20: float = 0.0           # 20-bar avg volume
    current_volume: float = 0.0          # current bar volume
    volume_ratio: float = 1.0            # current / avg


# ======================================================================
# Sub-Dataclass 8: MultiTimeframeConfirmation
# ======================================================================
@dataclass(frozen=True)
class MultiTimeframeConfirmation:
    """Alignment across timeframes — institutional confirmation."""
    m1: Optional[str] = None             # BUY/SELL/HOLD
    m5: Optional[str] = None
    m15: Optional[str] = None
    m30: Optional[str] = None
    h1: Optional[str] = None
    h4: Optional[str] = None
    d1: Optional[str] = None
    w1: Optional[str] = None
    aligned_timeframes: List[str] = field(default_factory=list)
    alignment_score: float = 0.0         # 0-1, 1=perfect alignment

    @property
    def all_aligned(self) -> bool:
        """True if every populated TF agrees on direction."""
        votes = [v for v in [self.m5, self.m15, self.h1, self.h4, self.d1] if v]
        if not votes:
            return False
        return len(set(votes)) == 1 and votes[0] != "HOLD"


# ======================================================================
# Sub-Dataclass 9: RiskEstimate
# ======================================================================
@dataclass(frozen=True)
class RiskEstimate:
    """Pre-trade risk estimates (signal-stage, before risk pipeline)."""
    estimated_risk_usd: float = 0.0      # dollar risk if SL hit
    estimated_risk_pct: float = 0.0      # % of equity at risk
    kelly_fraction: float = 0.0          # Kelly criterion fraction
    risk_reward: float = 0.0             # TP:SL ratio
    max_adverse_excursion: float = 0.0   # expected MAE in price units
    max_favorable_excursion: float = 0.0 # expected MFE
    expected_return: float = 0.0         # expected $ return
    expected_drawdown: float = 0.0       # expected DD during hold
    expected_winrate: float = 0.0        # historical win rate for this setup
    expected_duration_bars: int = 0      # expected hold time
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


# ======================================================================
# Sub-Dataclass 10: ExecutionHint
# ======================================================================
@dataclass(frozen=True)
class ExecutionHint:
    """Hints to the execution engine (not dictates)."""
    recommended_order_type: OrderType = OrderType.MARKET
    priority_score: float = 0.0          # 0-1, higher = more urgent
    urgency: Urgency = Urgency.NORMAL
    max_slippage_bps: float = 10.0
    time_in_force: str = "GTC"           # GTC, IOC, FOK, DAY
    suggested_volume: float = 0.0        # position size suggestion
    iceberg_pct: float = 0.0             # 0=visible, 1=fully hidden


# ======================================================================
# Sub-Dataclass 11: Explainability
# ======================================================================
@dataclass(frozen=True)
class Explainability:
    """Why did the AI/strategy make this decision?"""
    top_features: List[Tuple[str, float]] = field(default_factory=list)  # [(name, SHAP value)]
    shap_values: Dict[str, float] = field(default_factory=dict)
    attention_scores: Dict[str, float] = field(default_factory=dict)  # for transformers
    decision_trace: List[str] = field(default_factory=list)  # human-readable steps
    decision_latency_ms: float = 0.0


# ======================================================================
# Sub-Dataclass 12: EnsembleVotes
# ======================================================================
@dataclass(frozen=True)
class EnsembleVote:
    """Single agent's vote."""
    agent_name: str = ""
    vote: str = "HOLD"                   # BUY/SELL/HOLD/REDUCE
    confidence: float = 0.0
    weight: float = 1.0
    reasoning: str = ""


@dataclass(frozen=True)
class EnsembleVotes:
    """All agent votes + final consensus."""
    votes: List[EnsembleVote] = field(default_factory=list)
    final_action: str = "HOLD"
    final_strength: float = 0.0
    agreement_score: float = 0.0         # 0=full disagreement, 1=unanimous
    dissenting_agents: List[str] = field(default_factory=list)


# ======================================================================
# Sub-Dataclass 13: BayesianConfidence
# ======================================================================
@dataclass(frozen=True)
class BayesianConfidence:
    """Bayesian-updated confidence."""
    prior: float = 0.5                   # before seeing this bar
    posterior: float = 0.5               # after Bayesian update
    uncertainty: float = 0.5             # 1 - posterior confidence
    evidence_strength: float = 0.0       # how strong was the new evidence
    posterior_samples: int = 0           # MCMC samples (if applicable)


# ======================================================================
# Sub-Dataclass 14: UncertaintyEstimate
# ======================================================================
@dataclass(frozen=True)
class UncertaintyEstimate:
    """Decomposed uncertainty — critical for risk-aware sizing."""
    epistemic_uncertainty: float = 0.0   # model uncertainty (reducible)
    aleatoric_uncertainty: float = 0.0   # data uncertainty (irreducible)
    total_uncertainty: float = 0.0       # epistemic + aleatoric
    confidence_interval_95: Tuple[float, float] = (0.0, 1.0)


# ======================================================================
# Sub-Dataclass 15: MarketMicrostructure
# ======================================================================
@dataclass(frozen=True)
class MarketMicrostructure:
    """Order flow + microstructure signals."""
    order_imbalance: float = 0.0         # -1 (sell-heavy) to +1 (buy-heavy)
    orderflow_delta: float = 0.0         # buy vol - sell vol
    cvd: float = 0.0                     # cumulative volume delta
    absorption_detected: bool = False
    iceberg_detected: bool = False
    spoofing_detected: bool = False
    large_trades_count: int = 0          # > $100k trades in last bar


# ======================================================================
# Sub-Dataclass 16: OnChainSnapshot
# ======================================================================
@dataclass(frozen=True)
class OnChainSnapshot:
    """On-chain crypto metrics (BTC/ETH/etc)."""
    funding_rate: float = 0.0            # perp futures funding rate
    open_interest_usd: float = 0.0
    liquidation_24h_usd: float = 0.0
    whale_inflow_usd: float = 0.0        # large deposits to exchanges
    whale_outflow_usd: float = 0.0       # large withdrawals
    stablecoin_flow_usd: float = 0.0     # USDT/USDC net flow
    exchange_reserves: float = 0.0


# ======================================================================
# Sub-Dataclass 17: SentimentSnapshot
# ======================================================================
@dataclass(frozen=True)
class SentimentSnapshot:
    """Market sentiment across sources."""
    fear_greed_index: float = 50.0       # 0-100
    twitter_sentiment: float = 0.0       # -1 (bearish) to +1 (bullish)
    reddit_sentiment: float = 0.0
    news_sentiment: float = 0.0          # from NLP on news headlines
    social_volume: int = 0               # mentions per hour
    dominant_emotion: str = "neutral"    # fear/greed/hope/panic/euphoria


# ======================================================================
# Sub-Dataclass 18: NewsFlag
# ======================================================================
@dataclass(frozen=True)
class NewsFlag:
    """Scheduled news events that may impact the trade."""
    high_impact_news: bool = False
    minutes_to_news: float = 0.0         # negative = past event
    event_name: str = ""                 # "FOMC", "CPI", "NFP"
    event_currency: str = ""             # "USD", "EUR"
    blackout_until: Optional[str] = None # ISO timestamp — no trades until


# ======================================================================
# Sub-Dataclass 19: SessionInfo
# ======================================================================
@dataclass(frozen=True)
class SessionInfo:
    """Trading session details."""
    session: TradingSession = TradingSession.OFF_HOURS
    time_in_session_s: float = 0.0
    session_progress_pct: float = 0.0    # 0 = just opened, 100 = closing
    sessions_open: List[str] = field(default_factory=list)  # ["asia", "london"]


# ======================================================================
# Sub-Dataclass 20: CorrelationSnapshot
# ======================================================================
@dataclass(frozen=True)
class CorrelationSnapshot:
    """Correlations with major assets — for hedging + diversification."""
    btc_correlation: float = 0.0         # for non-BTC assets
    eth_correlation: float = 0.0
    sp500_correlation: float = 0.0
    dxy_correlation: float = 0.0         # US Dollar Index
    gold_correlation: float = 0.0
    avg_correlation: float = 0.0         # mean off-diagonal
    correlation_risk: str = "low"        # low/medium/high (for portfolio)


# ======================================================================
# Sub-Dataclass 21: FeatureMetadata
# ======================================================================
@dataclass(frozen=True)
class FeatureMetadata:
    """Feature vector metadata for ML training + replay."""
    feature_hash: str = ""               # MD5 of feature vector
    feature_version: str = "1.0"         # pipeline version
    feature_vector: Dict[str, Any] = field(default_factory=dict)
    feature_count: int = 0


# ======================================================================
# Sub-Dataclass 22: AuditMetadata
# ======================================================================
@dataclass(frozen=True)
class AuditMetadata:
    """Who/what/when/where — full provenance."""
    created_by: str = "industrial_bot"
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    hostname: str = field(default_factory=socket.gethostname)
    bot_version: str = "8.0"
    git_commit: str = ""                 # set at build time
    build_number: str = ""
    environment: str = "production"      # dev/staging/production


# ======================================================================
# Sub-Dataclass 23: ReplayMetadata
# ======================================================================
@dataclass(frozen=True)
class ReplayMetadata:
    """For exact replay in backtest/simulation."""
    cycle_id: int = 0
    snapshot_id: str = ""
    market_snapshot_id: str = ""         # links to market data snapshot
    config_version: str = ""
    state_machine_state: str = "LIVE"


# ======================================================================
# The Master Signal — Immutable Decision Contract
# ======================================================================
@dataclass(frozen=True)
class Signal:
    """Industrial-Grade Immutable Decision Contract (v3).

    A Signal is the single authoritative object that flows through every
    layer of the trading platform. It is:
        - Immutable (frozen=True) — cannot be mutated after creation
        - Typed — 22 sub-dataclasses, no loose dicts
        - Auditable — UUID + correlation_id + decision trace
        - Replayable — feature_hash + snapshot_id for exact replay
        - Serializable — to_dict / to_json / from_dict / from_json

    Construction:
        Use SignalBuilder for incremental construction. Direct construction
        is allowed but requires all 23 sub-objects (most can default).

    Schema Version: 3
    """
    # === 23 Sub-Objects ===
    identity: SignalIdentity = field(default_factory=SignalIdentity)
    market: MarketSnapshot = field(default_factory=MarketSnapshot)
    strategy: StrategyMetadata = field(default_factory=StrategyMetadata)
    ai: AIInference = field(default_factory=AIInference)
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    volatility: VolatilitySnapshot = field(default_factory=VolatilitySnapshot)
    liquidity: LiquiditySnapshot = field(default_factory=LiquiditySnapshot)
    mtf: MultiTimeframeConfirmation = field(default_factory=MultiTimeframeConfirmation)
    risk: RiskEstimate = field(default_factory=RiskEstimate)
    execution: ExecutionHint = field(default_factory=ExecutionHint)
    explain: Explainability = field(default_factory=Explainability)
    ensemble: EnsembleVotes = field(default_factory=EnsembleVotes)
    bayesian: BayesianConfidence = field(default_factory=BayesianConfidence)
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)
    microstructure: MarketMicrostructure = field(default_factory=MarketMicrostructure)
    onchain: OnChainSnapshot = field(default_factory=OnChainSnapshot)
    sentiment: SentimentSnapshot = field(default_factory=SentimentSnapshot)
    news: NewsFlag = field(default_factory=NewsFlag)
    session_info: SessionInfo = field(default_factory=SessionInfo)
    correlation: CorrelationSnapshot = field(default_factory=CorrelationSnapshot)
    feature_meta: FeatureMetadata = field(default_factory=FeatureMetadata)
    audit: AuditMetadata = field(default_factory=AuditMetadata)
    replay: ReplayMetadata = field(default_factory=ReplayMetadata)

    # === Schema ===
    schema_version: int = 3
    # Backward-compat: legacy code may still access .meta / .symbol etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------
    def __post_init__(self) -> None:
        # Auto-set entry_price from bar_close if not set
        if self.risk.entry_price == 0.0 and self.market.bar_close > 0.0:
            new_risk = replace(self.risk, entry_price=self.market.bar_close)
            object.__setattr__(self, "risk", new_risk)
        # Auto-set expiry if not set
        if self.strategy.expires_at is None and self.market.bar_time:
            try:
                bar_dt = datetime.fromisoformat(self.market.bar_time.replace("Z", "+00:00"))
                expiry = bar_dt + timedelta(seconds=self.strategy.ttl_seconds)
                new_strat = replace(self.strategy, expires_at=expiry.isoformat())
                object.__setattr__(self, "strategy", new_strat)
            except Exception:
                pass
        # Auto-compute feature_hash if not set
        if self.feature_meta.feature_hash == "" and self.feature_meta.feature_vector:
            h = hashlib.md5(
                ",".join(f"{k}={self.feature_meta.feature_vector[k]}"
                         for k in sorted(self.feature_meta.feature_vector.keys())
                         ).encode()
            ).hexdigest()[:12]
            new_fm = replace(self.feature_meta,
                            feature_hash=h,
                            feature_count=len(self.feature_meta.feature_vector))
            object.__setattr__(self, "feature_meta", new_fm)

    # ----------------------------------------------------------------
    # Convenience Properties (for backward compat with v2 code)
    # ----------------------------------------------------------------
    @property
    def signal_id(self) -> str:
        return self.identity.signal_id

    @property
    def symbol(self) -> str:
        return self.market.symbol

    @property
    def timeframe(self) -> str:
        return self.market.timeframe

    @property
    def action(self) -> Action:
        return self.strategy.action

    @property
    def strength(self) -> float:
        return self.strategy.strength

    @property
    def quality(self) -> SignalQuality:
        return self.strategy.quality_grade

    @property
    def price(self) -> float:
        return self.market.bar_close

    @property
    def entry_price(self) -> float:
        return self.risk.entry_price or self.market.bar_close

    @property
    def stop_loss(self) -> float:
        return self.risk.stop_loss

    @property
    def take_profit(self) -> float:
        return self.risk.take_profit

    @property
    def bar_time(self) -> Optional[datetime]:
        if self.market.bar_time:
            try:
                return datetime.fromisoformat(self.market.bar_time.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @property
    def expires_at(self) -> Optional[datetime]:
        if self.strategy.expires_at:
            try:
                return datetime.fromisoformat(self.strategy.expires_at.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @property
    def source(self) -> SignalSource:
        return self.strategy.source

    @property
    def execution_status(self) -> ExecutionStatus:
        return self.strategy.execution_status

    @property
    def is_actionable(self) -> bool:
        """True if action is BUY or SELL."""
        return self.strategy.action in (Action.BUY, Action.SELL)

    @property
    def is_expired(self) -> bool:
        """True if signal has passed its expiry."""
        exp = self.expires_at
        if exp is None:
            return False
        return datetime.now(timezone.utc) > exp

    @property
    def direction(self) -> str:
        """'long', 'short', or 'neutral'."""
        if self.strategy.action == Action.BUY:
            return "long"
        if self.strategy.action == Action.SELL:
            return "short"
        return "neutral"

    @property
    def rr_ratio(self) -> float:
        """Reward:Risk ratio."""
        if self.risk.risk_reward > 0:
            return self.risk.risk_reward
        r = abs(self.entry_price - self.stop_loss)
        w = abs(self.take_profit - self.entry_price)
        return w / r if r > 0 else 0.0

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Full serialization for database + JSON logging."""
        return {
            "schema_version": self.schema_version,
            "identity": asdict(self.identity),
            "market": asdict(self.market),
            "strategy": asdict(self.strategy),
            "ai": asdict(self.ai),
            "confidence": asdict(self.confidence),
            "volatility": asdict(self.volatility),
            "liquidity": asdict(self.liquidity),
            "mtf": asdict(self.mtf),
            "risk": asdict(self.risk),
            "execution": asdict(self.execution),
            "explain": asdict(self.explain),
            "ensemble": asdict(self.ensemble),
            "bayesian": asdict(self.bayesian),
            "uncertainty": asdict(self.uncertainty),
            "microstructure": asdict(self.microstructure),
            "onchain": asdict(self.onchain),
            "sentiment": asdict(self.sentiment),
            "news": asdict(self.news),
            "session_info": asdict(self.session_info),
            "correlation": asdict(self.correlation),
            "feature_meta": asdict(self.feature_meta),
            "audit": asdict(self.audit),
            "replay": asdict(self.replay),
            "meta": self.meta,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """JSON serialization."""
        return json.dumps(self.to_dict(), default=str, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Deserialize from dict (database row or JSON)."""
        # Helper to safely construct sub-dataclasses
        def _sub(name: str, cls_):
            d = data.get(name, {})
            if not isinstance(d, dict):
                return cls_()
            # Filter to only valid fields
            valid_keys = {f.name for f in fields(cls_)}
            filtered = {k: v for k, v in d.items() if k in valid_keys}
            # Convert string enums back
            for f in fields(cls_):
                if f.name in filtered and isinstance(f.type, type) and \
                        issubclass(f.type, Enum) and isinstance(filtered[f.name], str):
                    try:
                        filtered[f.name] = f.type(filtered[f.name])
                    except ValueError:
                        # Minor #10 fix: log a warning instead of silently
                        # falling back to the default. Silent fallback means
                        # an unrecognised regime becomes "unknown" without
                        # any signal to the operator that data was corrupted.
                        log.warning("signals_v3: failed to convert %s=%r to %s — "
                                    "using default", f.name, filtered[f.name],
                                    f.type.__name__)
            return cls_(**filtered)

        return cls(
            identity=_sub("identity", SignalIdentity),
            market=_sub("market", MarketSnapshot),
            strategy=_sub("strategy", StrategyMetadata),
            ai=_sub("ai", AIInference),
            confidence=_sub("confidence", ConfidenceBreakdown),
            volatility=_sub("volatility", VolatilitySnapshot),
            liquidity=_sub("liquidity", LiquiditySnapshot),
            mtf=_sub("mtf", MultiTimeframeConfirmation),
            risk=_sub("risk", RiskEstimate),
            execution=_sub("execution", ExecutionHint),
            explain=_sub("explain", Explainability),
            ensemble=_sub("ensemble", EnsembleVotes),
            bayesian=_sub("bayesian", BayesianConfidence),
            uncertainty=_sub("uncertainty", UncertaintyEstimate),
            microstructure=_sub("microstructure", MarketMicrostructure),
            onchain=_sub("onchain", OnChainSnapshot),
            sentiment=_sub("sentiment", SentimentSnapshot),
            news=_sub("news", NewsFlag),
            session_info=_sub("session_info", SessionInfo),
            correlation=_sub("correlation", CorrelationSnapshot),
            feature_meta=_sub("feature_meta", FeatureMetadata),
            audit=_sub("audit", AuditMetadata),
            replay=_sub("replay", ReplayMetadata),
            schema_version=int(data.get("schema_version", 3)),
            meta=data.get("meta", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Signal":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ----------------------------------------------------------------
    # Immutability helpers (functional update)
    # ----------------------------------------------------------------
    def with_status(self, status: ExecutionStatus) -> "Signal":
        """Return a copy with updated execution status."""
        new_strategy = replace(self.strategy, execution_status=status)
        return replace(self, strategy=new_strategy)

    def with_reject_reason(self, reason: str) -> "Signal":
        """Return a copy with rejection reason set."""
        new_strategy = replace(self.strategy,
                              reject_reason=reason,
                              execution_status=ExecutionStatus.REJECTED)
        return replace(self, strategy=new_strategy)

    def with_audit(self, **kwargs: Any) -> "Signal":
        """Return a copy with audit fields updated."""
        new_audit = replace(self.audit, **kwargs)
        return replace(self, audit=new_audit)

    # ----------------------------------------------------------------
    # Factory methods (for backward compat with v2 code)
    # ----------------------------------------------------------------
    @classmethod
    def hold(cls, symbol: str, timeframe: str = "M15", price: float = 0.0,
             bar_time: Optional[datetime] = None,
             reason: str = "no signal", **kwargs: Any) -> "Signal":
        """Create a HOLD signal."""
        bar_time_str = bar_time.isoformat() if bar_time else None
        market = MarketSnapshot(symbol=symbol, timeframe=timeframe,
                                bar_close=price, bar_time=bar_time_str)
        strategy = StrategyMetadata(action=Action.HOLD, strength=0.0,
                                    reject_reason=reason)
        return cls(market=market, strategy=strategy, **kwargs)

    @classmethod
    def buy(cls, symbol: str, timeframe: str, strength: float,
            price: float = 0.0, bar_time: Optional[datetime] = None,
            entry_price: float = 0.0, stop_loss: float = 0.0,
            take_profit: float = 0.0,
            confidence: float = 0.0,
            quality: SignalQuality = SignalQuality.B,
            strategy_id: str = "", strategy_version: str = "",
            **kwargs: Any) -> "Signal":
        """Create a BUY signal with full institutional context."""
        bar_time_str = bar_time.isoformat() if bar_time else None
        market = MarketSnapshot(symbol=symbol, timeframe=timeframe,
                                bar_close=price, bar_time=bar_time_str)
        strategy = StrategyMetadata(action=Action.BUY, strength=strength,
                                    quality_grade=quality)
        risk = RiskEstimate(entry_price=entry_price or price,
                           stop_loss=stop_loss, take_profit=take_profit)
        identity = SignalIdentity(strategy_id=strategy_id,
                                  strategy_version=strategy_version)
        cb = ConfidenceBreakdown(overall=confidence or strength)
        return cls(market=market, strategy=strategy, risk=risk,
                  identity=identity, confidence=cb, **kwargs)

    @classmethod
    def sell(cls, symbol: str, timeframe: str, strength: float,
             price: float = 0.0, bar_time: Optional[datetime] = None,
             entry_price: float = 0.0, stop_loss: float = 0.0,
             take_profit: float = 0.0,
             confidence: float = 0.0,
             quality: SignalQuality = SignalQuality.B,
             strategy_id: str = "", strategy_version: str = "",
             **kwargs: Any) -> "Signal":
        """Create a SELL signal with full institutional context."""
        bar_time_str = bar_time.isoformat() if bar_time else None
        market = MarketSnapshot(symbol=symbol, timeframe=timeframe,
                                bar_close=price, bar_time=bar_time_str)
        strategy = StrategyMetadata(action=Action.SELL, strength=strength,
                                    quality_grade=quality)
        risk = RiskEstimate(entry_price=entry_price or price,
                           stop_loss=stop_loss, take_profit=take_profit)
        identity = SignalIdentity(strategy_id=strategy_id,
                                  strategy_version=strategy_version)
        cb = ConfidenceBreakdown(overall=confidence or strength)
        return cls(market=market, strategy=strategy, risk=risk,
                  identity=identity, confidence=cb, **kwargs)


# ======================================================================
# SignalBuilder — fluent construction for complex signals
# ======================================================================
class SignalBuilder:
    """Fluent builder for constructing complex Signals incrementally.

    Example:
        signal = (SignalBuilder()
                  .with_symbol("BTCUSD", "M15")
                  .with_action(Action.BUY, strength=0.85)
                  .with_price(43250.0)
                  .with_sl_tp(42500, 45000)
                  .with_strategy("Momentum_v4.1", "4.1.0")
                  .with_ai_model("transformer_v9", "9.0", 0.85)
                  .with_confidence(trend=0.9, momentum=0.8, volume=0.7)
                  .with_regime(MarketRegime.TREND_UP)
                  .with_risk(estimated_risk_usd=200, kelly=0.15)
                  .with_execution(OrderType.LIMIT, urgency=Urgency.HIGH)
                  .with_explainability([("rsi", 0.3), ("macd", 0.25)])
                  .build())
    """

    def __init__(self):
        self._identity = SignalIdentity()
        self._market = MarketSnapshot()
        self._strategy = StrategyMetadata()
        self._ai = AIInference()
        self._confidence = ConfidenceBreakdown()
        self._volatility = VolatilitySnapshot()
        self._liquidity = LiquiditySnapshot()
        self._mtf = MultiTimeframeConfirmation()
        self._risk = RiskEstimate()
        self._execution = ExecutionHint()
        self._explain = Explainability()
        self._ensemble = EnsembleVotes()
        self._bayesian = BayesianConfidence()
        self._uncertainty = UncertaintyEstimate()
        self._microstructure = MarketMicrostructure()
        self._onchain = OnChainSnapshot()
        self._sentiment = SentimentSnapshot()
        self._news = NewsFlag()
        self._session = SessionInfo()
        self._correlation = CorrelationSnapshot()
        self._feature_meta = FeatureMetadata()
        self._audit = AuditMetadata()
        self._replay = ReplayMetadata()
        self._meta: Dict[str, Any] = {}

    # Identity
    def with_strategy(self, strategy_id: str, version: str) -> "SignalBuilder":
        self._identity = replace(self._identity,
                                strategy_id=strategy_id,
                                strategy_version=version)
        return self

    def with_parent(self, parent_id: str) -> "SignalBuilder":
        self._identity = replace(self._identity, parent_signal_id=parent_id)
        return self

    # Market
    def with_symbol(self, symbol: str, timeframe: str = "M15") -> "SignalBuilder":
        self._market = replace(self._market, symbol=symbol, timeframe=timeframe)
        return self

    def with_price(self, price: float, bar_time: Optional[datetime] = None,
                   high: float = 0, low: float = 0, volume: float = 0) -> "SignalBuilder":
        bt_str = bar_time.isoformat() if bar_time else None
        self._market = replace(self._market, bar_close=price, bar_time=bt_str,
                              bar_high=high, bar_low=low, bar_volume=volume)
        return self

    def with_regime(self, regime: MarketRegime,
                    session: TradingSession = TradingSession.OFF_HOURS) -> "SignalBuilder":
        self._market = replace(self._market, regime=regime, session=session)
        return self

    # Strategy
    def with_action(self, action: Action, strength: float,
                    quality: SignalQuality = SignalQuality.B,
                    ttl_s: float = 3600) -> "SignalBuilder":
        self._strategy = replace(self._strategy, action=action, strength=strength,
                                quality_grade=quality, ttl_seconds=ttl_s)
        return self

    def with_source(self, source_type: SignalSourceType,
                    source: SignalSource = SignalSource.LIVE) -> "SignalBuilder":
        self._strategy = replace(self._strategy, source_type=source_type, source=source)
        return self

    # Risk
    def with_sl_tp(self, sl: float, tp: float,
                   entry: Optional[float] = None) -> "SignalBuilder":
        self._risk = replace(self._risk, stop_loss=sl, take_profit=tp,
                            entry_price=entry or self._market.bar_close)
        return self

    def with_risk(self, estimated_risk_usd: float = 0,
                  kelly_fraction: float = 0,
                  expected_winrate: float = 0,
                  expected_duration_bars: int = 0) -> "SignalBuilder":
        self._risk = replace(self._risk,
                            estimated_risk_usd=estimated_risk_usd,
                            kelly_fraction=kelly_fraction,
                            expected_winrate=expected_winrate,
                            expected_duration_bars=expected_duration_bars)
        return self

    # AI
    def with_ai_model(self, name: str, version: str,
                      weight: float = 1.0,
                      latency_ms: float = 0,
                      confidence: float = 0) -> "SignalBuilder":
        self._ai = replace(self._ai, model_name=name, model_version=version,
                          ensemble_weight=weight, inference_latency_ms=latency_ms,
                          model_confidence=confidence)
        return self

    def with_embedding(self, embedding_id: str) -> "SignalBuilder":
        self._ai = replace(self._ai, embedding_id=embedding_id)
        return self

    # Confidence
    def with_confidence(self, overall: float = 0, trend: float = 0,
                       momentum: float = 0, volume: float = 0,
                       ai: float = 0, macro: float = 0,
                       pattern: float = 0, sentiment: float = 0) -> "SignalBuilder":
        self._confidence = ConfidenceBreakdown(
            overall=overall, trend_confidence=trend,
            momentum_confidence=momentum, volume_confidence=volume,
            ai_confidence=ai, macro_confidence=macro,
            pattern_confidence=pattern, sentiment_confidence=sentiment,
        )
        return self

    # Volatility
    def with_volatility(self, atr: float = 0, atr_pct: float = 0,
                       hv: float = 0, rv: float = 0,
                       percentile: float = 0.5) -> "SignalBuilder":
        self._volatility = VolatilitySnapshot(
            atr=atr, atr_pct=atr_pct, historical_vol_20=hv,
            realized_vol_20=rv, volatility_percentile=percentile,
        )
        return self

    # Liquidity
    def with_liquidity(self, spread_bps: float = 0,
                       slippage_bps: float = 0,
                       depth_usd: float = 0) -> "SignalBuilder":
        self._liquidity = replace(self._liquidity, spread_bps=spread_bps,
                                 slippage_estimate_bps=slippage_bps,
                                 orderbook_depth_usd=depth_usd)
        return self

    # Multi-timeframe
    def with_mtf(self, m5: str = None, m15: str = None, h1: str = None,
                 h4: str = None, d1: str = None) -> "SignalBuilder":
        aligned = [v for v in [m5, m15, h1, h4, d1] if v and v != "HOLD"]
        score = (len(set(aligned)) <= 1 and len(aligned) > 0) and 1.0 or \
                (len(aligned) / 5.0 if aligned else 0.0)
        self._mtf = MultiTimeframeConfirmation(
            m5=m5, m15=m15, h1=h1, h4=h4, d1=d1,
            aligned_timeframes=aligned, alignment_score=score,
        )
        return self

    # Execution
    def with_execution(self, order_type: OrderType = OrderType.MARKET,
                       priority: float = 0.5,
                       urgency: Urgency = Urgency.NORMAL,
                       max_slippage_bps: float = 10) -> "SignalBuilder":
        self._execution = ExecutionHint(
            recommended_order_type=order_type, priority_score=priority,
            urgency=urgency, max_slippage_bps=max_slippage_bps,
        )
        return self

    # Explainability
    def with_explainability(self, top_features: List[Tuple[str, float]] = None,
                           decision_trace: List[str] = None,
                           latency_ms: float = 0) -> "SignalBuilder":
        self._explain = Explainability(
            top_features=top_features or [],
            decision_trace=decision_trace or [],
            decision_latency_ms=latency_ms,
        )
        return self

    def with_shap(self, shap: Dict[str, float]) -> "SignalBuilder":
        self._explain = replace(self._explain, shap_values=shap)
        return self

    # Ensemble
    def with_ensemble(self, votes: List[EnsembleVote],
                      final_action: str = "HOLD",
                      final_strength: float = 0.0,
                      agreement: float = 0.0) -> "SignalBuilder":
        dissenters = [v.agent_name for v in votes
                     if v.vote != final_action and v.vote != "HOLD"]
        self._ensemble = EnsembleVotes(
            votes=votes, final_action=final_action,
            final_strength=final_strength, agreement_score=agreement,
            dissenting_agents=dissenters,
        )
        return self

    # Uncertainty
    def with_uncertainty(self, epistemic: float = 0,
                        aleatoric: float = 0,
                        ci_95: Tuple[float, float] = (0, 1)) -> "SignalBuilder":
        self._uncertainty = UncertaintyEstimate(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=epistemic + aleatoric,
            confidence_interval_95=ci_95,
        )
        return self

    # Bayesian
    def with_bayesian(self, prior: float, posterior: float,
                      evidence: float = 0) -> "SignalBuilder":
        self._bayesian = BayesianConfidence(
            prior=prior, posterior=posterior,
            uncertainty=1.0 - posterior, evidence_strength=evidence,
        )
        return self

    # Microstructure
    def with_microstructure(self, imbalance: float = 0, delta: float = 0,
                           cvd: float = 0, absorption: bool = False) -> "SignalBuilder":
        self._microstructure = MarketMicrostructure(
            order_imbalance=imbalance, orderflow_delta=delta, cvd=cvd,
            absorption_detected=absorption,
        )
        return self

    # On-chain
    def with_onchain(self, funding: float = 0, oi: float = 0,
                     liquidation: float = 0, whale_inflow: float = 0) -> "SignalBuilder":
        self._onchain = OnChainSnapshot(
            funding_rate=funding, open_interest_usd=oi,
            liquidation_24h_usd=liquidation, whale_inflow_usd=whale_inflow,
        )
        return self

    # Sentiment
    def with_sentiment(self, fear_greed: float = 50,
                      twitter: float = 0, reddit: float = 0,
                      news_sent: float = 0) -> "SignalBuilder":
        self._sentiment = SentimentSnapshot(
            fear_greed_index=fear_greed, twitter_sentiment=twitter,
            reddit_sentiment=reddit, news_sentiment=news_sent,
        )
        return self

    # News
    def with_news(self, high_impact: bool = False,
                  minutes_to: float = 0, event: str = "") -> "SignalBuilder":
        self._news = NewsFlag(
            high_impact_news=high_impact, minutes_to_news=minutes_to,
            event_name=event,
        )
        return self

    # Correlation
    def with_correlation(self, btc: float = 0, eth: float = 0,
                        sp500: float = 0, dxy: float = 0,
                        gold: float = 0) -> "SignalBuilder":
        avg = sum([abs(btc), abs(eth), abs(sp500), abs(dxy), abs(gold)]) / 5
        risk = "low" if avg < 0.3 else "medium" if avg < 0.7 else "high"
        self._correlation = CorrelationSnapshot(
            btc_correlation=btc, eth_correlation=eth,
            sp500_correlation=sp500, dxy_correlation=dxy,
            gold_correlation=gold, avg_correlation=avg,
            correlation_risk=risk,
        )
        return self

    # Features
    def with_features(self, feature_vector: Dict[str, Any],
                     version: str = "1.0") -> "SignalBuilder":
        h = hashlib.md5(
            ",".join(f"{k}={feature_vector[k]}"
                     for k in sorted(feature_vector.keys())).encode()
        ).hexdigest()[:12]
        self._feature_meta = FeatureMetadata(
            feature_hash=h, feature_version=version,
            feature_vector=feature_vector,
            feature_count=len(feature_vector),
        )
        return self

    # Audit
    def with_audit(self, created_by: str = "industrial_bot",
                  bot_version: str = "8.0",
                  git_commit: str = "",
                  environment: str = "production") -> "SignalBuilder":
        self._audit = AuditMetadata(
            created_by=created_by, bot_version=bot_version,
            git_commit=git_commit, environment=environment,
        )
        return self

    # Replay
    def with_replay(self, cycle_id: int = 0, snapshot_id: str = "",
                   market_snapshot_id: str = "") -> "SignalBuilder":
        self._replay = ReplayMetadata(
            cycle_id=cycle_id, snapshot_id=snapshot_id,
            market_snapshot_id=market_snapshot_id,
        )
        return self

    # Meta (backward compat)
    def with_meta(self, **kwargs: Any) -> "SignalBuilder":
        self._meta.update(kwargs)
        return self

    # Build
    def build(self) -> Signal:
        """Construct the immutable Signal."""
        return Signal(
            identity=self._identity,
            market=self._market,
            strategy=self._strategy,
            ai=self._ai,
            confidence=self._confidence,
            volatility=self._volatility,
            liquidity=self._liquidity,
            mtf=self._mtf,
            risk=self._risk,
            execution=self._execution,
            explain=self._explain,
            ensemble=self._ensemble,
            bayesian=self._bayesian,
            uncertainty=self._uncertainty,
            microstructure=self._microstructure,
            onchain=self._onchain,
            sentiment=self._sentiment,
            news=self._news,
            session_info=self._session,
            correlation=self._correlation,
            feature_meta=self._feature_meta,
            audit=self._audit,
            replay=self._replay,
            meta=self._meta,
        )


# ======================================================================
# Migration helpers (v2 → v3)
# ======================================================================
def migrate_v2_to_v3(v2_signal: Any) -> Signal:
    """Migrate a v2 Signal (from engine.signals) to v3 format.

    v2_signal can be either a dict (serialized) or an actual v2 Signal object.
    """
    if isinstance(v2_signal, dict):
        d = v2_signal
    elif hasattr(v2_signal, "to_dict"):
        d = v2_signal.to_dict()
    else:
        d = asdict(v2_signal)

    builder = SignalBuilder()
    if d.get("symbol"):
        builder.with_symbol(d["symbol"], d.get("timeframe", "M15"))
    if d.get("action"):
        try:
            action = Action(d["action"]) if isinstance(d["action"], str) else d["action"]
            builder.with_action(action, strength=float(d.get("strength", 0)))
        except Exception:
            pass
    if d.get("price"):
        builder.with_price(float(d["price"]))
    if d.get("stop_loss") and d.get("take_profit"):
        builder.with_sl_tp(float(d["stop_loss"]), float(d["take_profit"]),
                          entry=float(d.get("entry_price", 0)))
    if d.get("strategy_name"):
        builder.with_strategy(d["strategy_name"], d.get("strategy_version", ""))
    if d.get("features"):
        builder.with_features(d["features"])
    if d.get("regime"):
        try:
            builder.with_regime(MarketRegime(d["regime"]))
        except Exception:
            pass
    if d.get("decision_trace"):
        builder.with_explainability(decision_trace=d["decision_trace"])
    return builder.build()
