"""engine.signals
===============================================================================
Day 90 — Institutional Signal Object + Factory (v2 — Hedge-Fund Grade)

Purpose
-------
A Signal is the immutable currency that flows between Strategy → Risk Engine →
Portfolio Manager → Execution Layer. It encapsulates a strategy's view at a
single closed candle with full institutional context.

This is NOT a signal generator. It is a typed, auditable, versioned DTO
(Data Transfer Object) that carries:
  - Strategy identity (name, version)
  - Market context (regime, session, instrument type)
  - Entry/SL/TP levels
  - Risk metrics (R:R, pip distance, risk flags)
  - Confluence data (confirmation count, ensemble vote, ML score)
  - Audit trail (decision trace, signal UUID, latency)
  - Lifecycle tracking (execution status, expiry, source)
  - Feature snapshot (for ML training replay)

Design Principles
-----------------
- Immutable (frozen dataclass) — prevents downstream mutation
- Typed context (SignalContext) — not loose dict
- Unique UUID per signal — duplicate detection + audit
- Signal expiry — stale signals auto-expire
- Closed-candle confirmation only — no repaint
- Full serialization — database + JSON compatible
- Schema versioning — future compatibility

Signal Lifecycle
----------------
Strategy.generate()  →  Signal (immutable, UUID-tagged)
        │
        ▼
Risk Engine          →  ApprovedTrade or None (reject_reason set)
        │
        ▼
Wisdom Gate (120)    →  approved/rejected + position_multiplier
        │
        ▼
Execution Engine     →  OrderResult (MT5 order) + execution_status updated
        │
        ▼
Database             →  persisted with full features snapshot for ML replay
===============================================================================
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("engine.signals")


# ======================================================================
# Enumerations
# ======================================================================
class Action(str, Enum):
    """Institutional trading actions.

    Extended beyond simple BUY/SELL to support position management:
        BUY           — enter long
        SELL          — enter short
        HOLD          — no action (strategy neutral)
        CLOSE_LONG    — close existing long position
        CLOSE_SHORT   — close existing short position
        PARTIAL_CLOSE — partially close position
        SCALE_IN      — add to existing position (pyramid)
        SCALE_OUT     — reduce existing position
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"

    def __str__(self) -> str:
        return self.value


class MarketRegime(str, Enum):
    """Market regime classification at signal time."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class TradingSession(str, Enum):
    """Trading session at signal time."""
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    OFF_HOURS = "off_hours"


class InstrumentType(str, Enum):
    """Instrument classification."""
    FOREX = "forex"
    METAL = "metal"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class SignalSource(str, Enum):
    """Where the signal originated."""
    MT5 = "mt5"
    BACKTEST = "backtest"
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"


class ExecutionStatus(str, Enum):
    """Signal lifecycle status."""
    NEW = "new"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SignalQuality(str, Enum):
    """Trade quality grade (A+ = best, REJECT = do not trade)."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    REJECT = "REJECT"


# ======================================================================
# Typed Signal Context
# ======================================================================
@dataclass(frozen=True)
class SignalContext:
    """Typed market context at signal time — replaces loose meta dict.

    This carries everything an institutional system needs to:
    - Audit why a signal was generated
    - Replay the signal in backtest with identical context
    - Train ML models on feature snapshots
    - Track confluence and ensemble votes
    """
    # ── Strategy identity ──
    strategy_name: str = ""
    strategy_version: str = ""

    # ── Market context ──
    regime: str = "unknown"
    session: str = "off_hours"
    instrument_type: str = "unknown"

    # ── Confluence ──
    confirmation_count: int = 0
    ensemble_vote: str = ""         # e.g., "4/5"
    confirmations: list = field(default_factory=list)  # ["EMA", "RSI", "Volume", ...]

    # ── Risk metrics ──
    rr_ratio: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    risk_flags: list = field(default_factory=list)     # ["high_spread", "news_event", ...]

    # ── ML / AI ──
    ml_probability: float = 0.0
    features: dict = field(default_factory=dict)       # {"EMA20": 65000, "RSI": 54.2, ...}

    # ── Audit ──
    decision_trace: list = field(default_factory=list)  # ["EMA Bullish", "ADX>25", "RSI Rising"]
    decision_latency_ms: float = 0.0
    reject_reason: str = ""

    # ── Candle info ──
    bar_index: int = 0
    closed_bar: bool = True


# ======================================================================
# Signal — Immutable DTO
# ======================================================================
@dataclass(frozen=True)
class Signal:
    """Immutable institutional-grade signal object.

    Carries full context from strategy through risk, wisdom gate, execution,
    and database persistence. Every signal has a unique UUID for audit trail.

    Attributes
    ----------
    signal_id : str
        Unique UUID — for duplicate detection, execution tracking, DB lookup
    symbol : str
        Trading instrument (e.g., "BTCUSD")
    timeframe : str
        Chart timeframe (e.g., "M15")
    action : Action
        BUY, SELL, HOLD, CLOSE_LONG, SCALE_IN, etc.
    strength : float
        Strategy conviction in [0, 1]. Used for position sizing.
    confidence : float
        AI confidence in [0, 1]. May differ from strength (ML-adjusted).
    quality : SignalQuality
        A+, A, B, C, or REJECT — trade quality grade
    price : float
        Close price of the signal bar (entry reference)
    entry_price : float
        Intended entry price (may differ from bar close for limit orders)
    stop_loss : float
        Stop loss price
    take_profit : float
        Take profit price
    bar_time : datetime
        Timestamp of the signal bar (UTC)
    expires_at : datetime
        Signal expiry time — stale signals auto-expire
    source : SignalSource
        MT5, backtest, simulation, paper, or live
    execution_status : ExecutionStatus
        Lifecycle: NEW → APPROVED → EXECUTED (or REJECTED/EXPIRED)
    context : SignalContext
        Typed market context (regime, session, confluence, features, audit)
    schema_version : int
        Serialization schema version (future compatibility)
    """
    # ── Identity ──
    signal_id: str = field(default_factory=lambda: _uuid.uuid4().hex[:16])

    # ── Core ──
    symbol: str = ""
    timeframe: str = "M15"
    action: Action = Action.HOLD
    strength: float = 0.0
    confidence: float = 0.0
    quality: SignalQuality = SignalQuality.C

    # ── Price levels ──
    price: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # ── Timing ──
    bar_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # ── Source & lifecycle ──
    source: SignalSource = SignalSource.LIVE
    execution_status: ExecutionStatus = ExecutionStatus.NEW

    # ── Typed context ──
    context: SignalContext = field(default_factory=SignalContext)

    # ── Meta ──
    schema_version: int = 2
    meta: dict[str, Any] = field(default_factory=dict)  # backward compat

    # ── Phase 5 extensions (added in place — no schema_version bump to 3) ──
    # These fields let the canonical Signal carry the signal-quality telemetry
    # that Phase 5 wires in (confluence, MTF, regime, session, EV, explanation).
    # Defaults preserve backward compat with existing callers.
    confluence_score: float = 0.0   # 0..1 — from ConfluenceScorer
    mtf_alignment: float = 0.0      # -1 (full disagreement) to +1 (full alignment)
    expected_value_r: float = 0.0   # expected R-multiple from EV calculator
    explanation: list = field(default_factory=list)  # human-readable reasons

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------
    def __post_init__(self) -> None:
        # Clamp strength to [0, 1]
        if self.strength < 0.0 or self.strength > 1.0:
            object.__setattr__(self, "strength", max(0.0, min(1.0, self.strength)))
        # Clamp confidence to [0, 1]
        if self.confidence < 0.0 or self.confidence > 1.0:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))
        # Auto-set entry_price if not specified
        if self.entry_price == 0.0 and self.price > 0.0:
            object.__setattr__(self, "entry_price", self.price)
        # Auto-set expiry if not specified (default: 1 hour from bar_time)
        if self.expires_at is None and self.bar_time is not None:
            object.__setattr__(self, "expires_at",
                               self.bar_time + timedelta(hours=1))
        # H2 fix: validate that stop_loss and take_profit are on the
        # correct side of entry_price for the given action. A BUY signal
        # must have SL < entry < TP; a SELL signal must have TP < entry < SL.
        # If SL/TP are 0 (unset), skip validation. If they're on the wrong
        # side, log a warning and zero them out so downstream gates can
        # recompute (rather than placing an order with invalid stops).
        if self.action == Action.BUY and self.entry_price > 0:
            if self.stop_loss > 0 and self.stop_loss >= self.entry_price:
                log.warning("Signal %s BUY: stop_loss %.5f >= entry %.5f — zeroing",
                            self.symbol, self.stop_loss, self.entry_price)
                object.__setattr__(self, "stop_loss", 0.0)
            if self.take_profit > 0 and self.take_profit <= self.entry_price:
                log.warning("Signal %s BUY: take_profit %.5f <= entry %.5f — zeroing",
                            self.symbol, self.take_profit, self.entry_price)
                object.__setattr__(self, "take_profit", 0.0)
        elif self.action == Action.SELL and self.entry_price > 0:
            if self.stop_loss > 0 and self.stop_loss <= self.entry_price:
                log.warning("Signal %s SELL: stop_loss %.5f <= entry %.5f — zeroing",
                            self.symbol, self.stop_loss, self.entry_price)
                object.__setattr__(self, "stop_loss", 0.0)
            if self.take_profit > 0 and self.take_profit >= self.entry_price:
                log.warning("Signal %s SELL: take_profit %.5f >= entry %.5f — zeroing",
                            self.symbol, self.take_profit, self.entry_price)
                object.__setattr__(self, "take_profit", 0.0)

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @property
    def is_actionable(self) -> bool:
        """True if signal is BUY or SELL (not HOLD)."""
        return self.action in (Action.BUY, Action.SELL)

    @property
    def is_expired(self) -> bool:
        """True if signal has passed its expiry time."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def confidence_pct(self) -> float:
        """Confidence as percentage (0-100)."""
        return self.confidence * 100.0

    @property
    def strength_pct(self) -> float:
        """Strength as percentage (0-100)."""
        return self.strength * 100.0

    @property
    def rr_ratio(self) -> float:
        """Reward:Risk ratio (computed from SL/TP if not in context)."""
        if self.context.rr_ratio > 0:
            return self.context.rr_ratio
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk > 0:
            return reward / risk
        return 0.0

    @property
    def direction(self) -> str:
        """'long', 'short', or 'neutral'."""
        if self.action == Action.BUY:
            return "long"
        if self.action == Action.SELL:
            return "short"
        return "neutral"

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Full serialization for database + JSON logging."""
        d = {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "action": self.action.value,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "quality": self.quality.value,
            "price": self.price,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source": self.source.value,
            "execution_status": self.execution_status.value,
            "schema_version": self.schema_version,
            # Context
            "strategy_name": self.context.strategy_name,
            "strategy_version": self.context.strategy_version,
            "regime": self.context.regime,
            "session": self.context.session,
            "instrument_type": self.context.instrument_type,
            "confirmation_count": self.context.confirmation_count,
            "ensemble_vote": self.context.ensemble_vote,
            "rr_ratio": self.rr_ratio,
            "sl_pips": self.context.sl_pips,
            "tp_pips": self.context.tp_pips,
            "ml_probability": self.context.ml_probability,
            "decision_latency_ms": self.context.decision_latency_ms,
            "bar_index": self.context.bar_index,
            "closed_bar": self.context.closed_bar,
            "reject_reason": self.context.reject_reason,
            "confirmations": self.context.confirmations,
            "risk_flags": self.context.risk_flags,
            "decision_trace": self.context.decision_trace,
            "features": self.context.features,
            # Backward compat
            "meta": self.meta,
        }
        return d

    # ----------------------------------------------------------------
    # Factory methods — enforce construction discipline
    # ----------------------------------------------------------------
    @classmethod
    def hold(cls, symbol: str, timeframe: str, price: float = 0.0,
             bar_time: Optional[datetime] = None,
             reason: str = "no signal",
             **kwargs: Any) -> "Signal":
        """Create a HOLD signal (no action)."""
        return cls(symbol=symbol, timeframe=timeframe, action=Action.HOLD,
                   strength=0.0, price=price, bar_time=bar_time,
                   meta={"reason": reason}, **kwargs)

    @classmethod
    def buy(cls, symbol: str, timeframe: str, strength: float,
            price: float = 0.0, bar_time: Optional[datetime] = None,
            entry_price: float = 0.0, stop_loss: float = 0.0,
            take_profit: float = 0.0,
            confidence: float = 0.0,
            quality: SignalQuality = SignalQuality.B,
            context: Optional[SignalContext] = None,
            source: SignalSource = SignalSource.LIVE,
            **meta: Any) -> "Signal":
        """Create a BUY signal with full institutional context."""
        return cls(
            symbol=symbol, timeframe=timeframe, action=Action.BUY,
            strength=strength, confidence=confidence or strength,
            quality=quality, price=price, entry_price=entry_price or price,
            stop_loss=stop_loss, take_profit=take_profit,
            bar_time=bar_time, source=source,
            context=context or SignalContext(),
            meta=meta,
        )

    @classmethod
    def sell(cls, symbol: str, timeframe: str, strength: float,
             price: float = 0.0, bar_time: Optional[datetime] = None,
             entry_price: float = 0.0, stop_loss: float = 0.0,
             take_profit: float = 0.0,
             confidence: float = 0.0,
             quality: SignalQuality = SignalQuality.B,
             context: Optional[SignalContext] = None,
             source: SignalSource = SignalSource.LIVE,
             **meta: Any) -> "Signal":
        """Create a SELL signal with full institutional context."""
        return cls(
            symbol=symbol, timeframe=timeframe, action=Action.SELL,
            strength=strength, confidence=confidence or strength,
            quality=quality, price=price, entry_price=entry_price or price,
            stop_loss=stop_loss, take_profit=take_profit,
            bar_time=bar_time, source=source,
            context=context or SignalContext(),
            meta=meta,
        )


# ======================================================================
# Major #5 fix: Backward-compatibility adapter for signals_v3 migration.
# ======================================================================
# `signals_v3.py` defines a richer Signal with 23 sub-dataclasses. Code
# that imports from `engine.signals` (v2) may receive an object that
# doesn't match the v3 schema. This adapter provides a `to_v3()` method
# on the v2 Signal so it can be converted to the v3 format when passed
# to modules that expect the richer schema. The conversion is lossy —
# v3-only fields default to their empty/zero values.
#
# Usage:
#   from engine.signals import Signal as SignalV2
#   v3_signal = SignalV2.to_v3(v2_signal)
# ======================================================================

def _signal_to_v3(v2_signal: "Signal") -> Any:
    """Convert a v2 Signal to a v3 Signal (best-effort, lossy).

    Major #5 fix: provides a bridge between the two signal schemas so
    that code expecting signals_v3.Signal can accept engine.signals.Signal
    without attribute errors.
    """
    try:
        from engine.signals_v3 import Signal as SignalV3, SignalBuilder
        builder = SignalBuilder(
            symbol=v2_signal.symbol,
            timeframe=v2_signal.timeframe,
        )
        # Map v2 fields to v3 builder
        if hasattr(builder, "action"):
            builder.action(v2_signal.action.value if hasattr(v2_signal.action, "value") else str(v2_signal.action))
        if hasattr(builder, "strength"):
            builder.strength(v2_signal.strength)
        if hasattr(builder, "price"):
            builder.price(v2_signal.price)
        if hasattr(builder, "entry_price"):
            builder.entry_price(v2_signal.entry_price)
        if hasattr(builder, "stop_loss"):
            builder.stop_loss(v2_signal.stop_loss)
        if hasattr(builder, "take_profit"):
            builder.take_profit(v2_signal.take_profit)
        if hasattr(builder, "build"):
            return builder.build()
        return builder
    except ImportError:
        # signals_v3 not available — return the v2 signal as-is.
        return v2_signal
    except Exception:
        # Any conversion error — return the v2 signal as-is rather than
        # crashing the caller.
        return v2_signal


# Attach the adapter as a method on Signal.
Signal.to_v3 = _signal_to_v3  # type: ignore[attr-defined]
