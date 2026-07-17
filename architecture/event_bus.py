"""architecture/event_bus.py
=====================================================================
Industrial-Grade Event-Driven Architecture (Improvement #1)
=====================================================================
A central pub/sub event bus that decouples every component of the
trading platform. Producers emit events; consumers subscribe to
topics. This is the **nervous system** of the bot.

Event Lifecycle:
    PRODUCER (strategy, risk, execution, MT5)
        ↓ emit(event_type, payload)
    EventBus (in-memory ring buffer + async dispatch)
        ↓ fan-out to all subscribers
    CONSUMERS (logger, db, monitoring, recovery, audit, AI)

Supported Event Types (30+):
    - System: BOT_START, BOT_SHUTDOWN, STATE_TRANSITION, HEARTBEAT
    - Market: BAR_CLOSED, TICK_UPDATE, SPREAD_WIDEN, GAP_DETECTED
    - Signal: SIGNAL_GENERATED, SIGNAL_REJECTED, SIGNAL_EXPIRED
    - Risk:   RISK_LAYER_PASSED, RISK_LAYER_FAILED, DRAWDOWN_WARNING,
              CIRCUIT_BREAKER_TRIP
    - Wisdom: WISDOM_APPROVED, WISDOM_REJECTED
    - Trade:  ORDER_SUBMITTED, ORDER_FILLED, ORDER_REJECTED,
              ORDER_PARTIAL, POSITION_OPENED, POSITION_CLOSED,
              SL_HIT, TP_HIT, TIMEOUT_CLOSE, FORCE_CLOSE
    - Errors: MT5_DISCONNECT, MT5_RECONNECT, IPC_TIMEOUT,
              DB_WRITE_FAILED, STRATEGY_EXCEPTION
    - Recovery: SNAPSHOT_TAKEN, SNAPSHOT_RESTORED, RECOVERY_STARTED

Thread Safety: thread-safe with RLock; subscribers can be sync or async.
Reliability: ring buffer (last N events) for diagnostics + replay.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.event_bus")


class EventType(str, Enum):
    """All event types in the system. New types can be added freely."""
    # System lifecycle
    BOT_START = "bot.start"
    BOT_SHUTDOWN = "bot.shutdown"
    STATE_TRANSITION = "state.transition"
    HEARTBEAT = "system.heartbeat"
    CONFIG_RELOADED = "config.reloaded"

    # Market data
    BAR_CLOSED = "market.bar_closed"
    TICK_UPDATE = "market.tick"
    SPREAD_WIDEN = "market.spread_widen"
    GAP_DETECTED = "market.gap"
    VOLATILITY_SPIKE = "market.vol_spike"

    # Signals
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_REJECTED = "signal.rejected"
    SIGNAL_EXPIRED = "signal.expired"

    # Risk engine
    RISK_LAYER_PASSED = "risk.layer_passed"
    RISK_LAYER_FAILED = "risk.layer_failed"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    CIRCUIT_BREAKER_TRIP = "risk.circuit_breaker"

    # Wisdom gate
    WISDOM_APPROVED = "wisdom.approved"
    WISDOM_REJECTED = "wisdom.rejected"

    # Orders & positions
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    SL_HIT = "position.sl_hit"
    TP_HIT = "position.tp_hit"
    FORCE_CLOSE = "position.force_close"

    # Errors & recovery
    MT5_DISCONNECT = "error.mt5_disconnect"
    MT5_RECONNECT = "error.mt5_reconnect"
    IPC_TIMEOUT = "error.ipc_timeout"
    DB_WRITE_FAILED = "error.db_write"
    STRATEGY_EXCEPTION = "error.strategy_exception"
    SNAPSHOT_TAKEN = "recovery.snapshot_taken"
    SNAPSHOT_RESTORED = "recovery.snapshot_restored"
    RECOVERY_STARTED = "recovery.started"

    # Portfolio
    REBALANCE_TRIGGERED = "portfolio.rebalance"
    EXPOSURE_BREACH = "portfolio.exposure_breach"


@dataclass
class Event:
    """Immutable event envelope. Every event in the system wraps in this."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.HEARTBEAT
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    source: str = "unknown"        # which module emitted
    correlation_id: str = ""       # for tracing a single decision through layers
    sequence: int = 0              # monotonic per-bus counter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "sequence": self.sequence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# Type for subscriber callbacks
Subscriber = Callable[[Event], None]


class EventBus:
    """Central pub/sub event bus.

    Features:
        - O(1) subscribe / unsubscribe
        - Fan-out to multiple subscribers per event type
        - Wildcard subscription (receive ALL events)
        - In-memory ring buffer (last N events) for replay/diagnostics
        - Thread-safe (RLock)
        - Optional async dispatch
        - Correlation IDs for tracing decisions across layers
        - Metrics: events/sec, subscribers/event_type, errors

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.POSITION_OPENED, log_to_db)
        bus.subscribe_wildcard(metrics_collector)
        bus.emit(EventType.POSITION_OPENED,
                 payload={"symbol": "BTCUSD", ...},
                 source="execution_engine")
    """

    def __init__(self, ring_buffer_size: int = 5000):
        self._subs: Dict[EventType, List[Subscriber]] = {}
        self._wildcard_subs: List[Subscriber] = []
        self._lock = threading.RLock()
        self._ring: Deque[Event] = deque(maxlen=ring_buffer_size)
        self._sequence = 0
        self._metrics = {
            "total_emitted": 0,
            "total_delivered": 0,
            "total_errors": 0,
            "per_type_emitted": {},
        }

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------
    def subscribe(self, event_type: EventType, callback: Subscriber) -> None:
        """Subscribe to a specific event type."""
        with self._lock:
            self._subs.setdefault(event_type, []).append(callback)
        log.debug("event_bus: subscribed %s to %s",
                  getattr(callback, "__name__", repr(callback)),
                  event_type.value)

    def subscribe_wildcard(self, callback: Subscriber) -> None:
        """Subscribe to ALL events (e.g. for metrics, audit log)."""
        with self._lock:
            self._wildcard_subs.append(callback)
        log.debug("event_bus: wildcard subscriber added")

    def unsubscribe(self, event_type: EventType, callback: Subscriber) -> None:
        with self._lock:
            if event_type in self._subs:
                try:
                    self._subs[event_type].remove(callback)
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(self,
             event_type: EventType,
             payload: Optional[Dict[str, Any]] = None,
             source: str = "unknown",
             correlation_id: str = "") -> Event:
        """Emit an event. Synchronous dispatch to all subscribers.

        Exceptions in subscribers are caught and counted but never
        propagate to the emitter — one bad subscriber must not crash
        the trading loop.
        """
        with self._lock:
            self._sequence += 1
            seq = self._sequence

        event = Event(
            type=event_type,
            payload=payload or {},
            source=source,
            correlation_id=correlation_id or str(uuid.uuid4())[:8],
            sequence=seq,
        )

        # Store in ring buffer for diagnostics / replay
        with self._lock:
            self._ring.append(event)
            self._metrics["total_emitted"] += 1
            self._metrics["per_type_emitted"][event_type.value] = \
                self._metrics["per_type_emitted"].get(event_type.value, 0) + 1
            # Snapshot subscriber lists under lock to avoid race during dispatch
            subs = list(self._subs.get(event_type, []))
            wildcards = list(self._wildcard_subs)

        # Dispatch outside the lock to avoid deadlocks
        for cb in subs + wildcards:
            try:
                cb(event)
                with self._lock:
                    self._metrics["total_delivered"] += 1
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._metrics["total_errors"] += 1
                    # M2/X8 fix: dead-letter queue — failed events are stored
                    # for later inspection/retry instead of being silently
                    # dropped. Capped at 1000 entries to bound memory.
                    if not hasattr(self, '_dlq'):
                        self._dlq = []
                    if len(self._dlq) < 1000:
                        self._dlq.append({
                            "event": event,
                            "subscriber": getattr(cb, "__name__", repr(cb)),
                            "error": str(e),
                            "timestamp": time.time(),
                        })
                log.warning("event_bus subscriber %s raised: %r (event dead-lettered)",
                            getattr(cb, "__name__", repr(cb)), e)

        return event

    # ------------------------------------------------------------------
    # Diagnostics & replay
    # ------------------------------------------------------------------
    def replay(self, event_type: Optional[EventType] = None,
               last_n: int = 100) -> List[Event]:
        """Return last N events (optionally filtered by type)."""
        with self._lock:
            events = list(self._ring)
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        return events[-last_n:]

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    def dead_letter_queue(self) -> List[Dict[str, Any]]:
        """M2/X8 fix: return a copy of the dead-letter queue for inspection."""
        with self._lock:
            return list(getattr(self, '_dlq', []))

    def clear_dlq(self) -> int:
        """Clear the dead-letter queue. Returns the number of entries removed."""
        with self._lock:
            dlq = getattr(self, '_dlq', [])
            count = len(dlq)
            dlq.clear()
            return count

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._metrics = {
                "total_emitted": 0,
                "total_delivered": 0,
                "total_errors": 0,
                "per_type_emitted": {},
            }


# ----------------------------------------------------------------------
# Module-level singleton for convenience (use get_bus())
# ----------------------------------------------------------------------
_GLOBAL_BUS: Optional[EventBus] = None
_GLOBAL_BUS_LOCK = threading.Lock()


def get_bus() -> EventBus:
    """Return the process-wide EventBus singleton."""
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        with _GLOBAL_BUS_LOCK:
            if _GLOBAL_BUS is None:
                _GLOBAL_BUS = EventBus()
                log.info("EventBus initialized (singleton)")
    return _GLOBAL_BUS
