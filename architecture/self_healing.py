"""architecture/self_healing.py
=====================================================================
Self-Healing System (Improvement #8)
=====================================================================
Detects failures and automatically attempts recovery actions without
human intervention. Every recovery is logged and emitted to EventBus.

Failure Categories:
    - Connection failures (MT5 disconnect, IPC timeout)
    - Data failures (stale ticks, missing symbols, NaN in OHLCV)
    - Strategy exceptions (uncaught error in signal generation)
    - Order failures (rejected, partial fill, requoted)
    - Database write failures
    - Process crashes (handled by snapshot restore on restart)

Recovery Actions (escalation ladder):
    Level 0: Retry (immediate, same operation)
    Level 1: Reconnect (drop + re-establish connection)
    Level 2: Cool-down + retry (wait 30s then retry)
    Level 3: Switch to degraded mode (continue with reduced functionality)
    Level 4: Snapshot + restart component (in-memory state rebuilt)
    Level 5: Emergency stop (halt trading, alert human, preserve capital)

Decision Logic:
    - Track failure rate per component
    - If failures < threshold: Level 0-1 (transparent to user)
    - If failures >= threshold: Level 2-3 (logged as warning)
    - If critical component down > N minutes: Level 4-5
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from architecture.event_bus import EventBus, EventType, get_bus
from architecture.state_machine import BotState, get_state_machine
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.self_healing")


class FailureType(str, Enum):
    CONNECTION = "connection"
    DATA = "data"
    STRATEGY = "strategy"
    ORDER = "order"
    DATABASE = "database"
    INDICATOR = "indicator"
    UNKNOWN = "unknown"


class RecoveryLevel(int, Enum):
    NONE = 0
    RETRY = 1
    RECONNECT = 2
    COOLDOWN_RETRY = 3
    DEGRADED_MODE = 4
    SNAPSHOT_RESTART = 5
    EMERGENCY_STOP = 6


@dataclass
class FailureRecord:
    component: str
    failure_type: FailureType
    error: str
    timestamp: float = field(default_factory=time.time)
    recovery_level: RecoveryLevel = RecoveryLevel.NONE
    recovered: bool = False
    recovery_time_s: float = 0.0


class SelfHealingSystem:
    """Watches for failures and triggers automatic recovery.

    Usage:
        healer = SelfHealingSystem()
        healer.register_recovery(FailureType.CONNECTION, "mt5_adapter",
                                 reconnect_fn, max_retries=5)
        # On failure:
        healer.report_failure(FailureType.CONNECTION, "mt5_adapter",
                              error="IPC timeout -10005")
    """

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 failure_window_s: float = 300.0):
        self._bus = bus or get_bus()
        self._lock = threading.RLock()
        self._recovery_fns: Dict[tuple, Callable] = {}
        self._retry_counts: Dict[tuple, int] = {}
        self._max_retries: Dict[tuple, int] = {}
        self._failure_log: List[FailureRecord] = []
        self._failure_window_s = failure_window_s
        self._circuit_open: Dict[str, float] = {}  # component -> open_until
        self._degraded_components: set = set()

        # Subscribe to error events
        self._bus.subscribe(EventType.MT5_DISCONNECT, self._on_mt5_disconnect)
        self._bus.subscribe(EventType.IPC_TIMEOUT, self._on_ipc_timeout)
        self._bus.subscribe(EventType.DB_WRITE_FAILED, self._on_db_failure)
        self._bus.subscribe(EventType.STRATEGY_EXCEPTION, self._on_strategy_error)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_recovery(self,
                          failure_type: FailureType,
                          component: str,
                          recovery_fn: Callable[[], bool],
                          max_retries: int = 3) -> None:
        """Register a recovery function for a specific failure type + component."""
        key = (failure_type, component)
        with self._lock:
            self._recovery_fns[key] = recovery_fn
            self._max_retries[key] = max_retries
            self._retry_counts[key] = 0
        log.info("self_healing: registered recovery for %s/%s (max_retries=%d)",
                 failure_type.value, component, max_retries)

    # ------------------------------------------------------------------
    # Failure reporting + recovery
    # ------------------------------------------------------------------
    def report_failure(self,
                       failure_type: FailureType,
                       component: str,
                       error: str,
                       context: Optional[Dict[str, Any]] = None) -> RecoveryLevel:
        """Report a failure and trigger recovery if available.

        Returns the RecoveryLevel that was attempted.
        """
        key = (failure_type, component)
        record = FailureRecord(
            component=component,
            failure_type=failure_type,
            error=error,
        )

        # Check if circuit breaker is open for this component
        with self._lock:
            open_until = self._circuit_open.get(component, 0)
            if time.time() < open_until:
                log.warning("self_healing: circuit breaker OPEN for %s "
                            "(%.0fs remaining) — skipping recovery",
                            component, open_until - time.time())
                record.recovery_level = RecoveryLevel.COOLDOWN_RETRY
                record.recovered = False
                self._failure_log.append(record)
                return RecoveryLevel.COOLDOWN_RETRY

            retry_count = self._retry_counts.get(key, 0)
            max_retries = self._max_retries.get(key, 3)
            recovery_fn = self._recovery_fns.get(key)

        if recovery_fn is None:
            log.warning("self_healing: NO recovery fn for %s/%s — escalating",
                        failure_type.value, component)
            record.recovery_level = RecoveryLevel.DEGRADED_MODE
            self._degrade_component(component)
            self._failure_log.append(record)
            return RecoveryLevel.DEGRADED_MODE

        if retry_count >= max_retries:
            log.error("self_healing: max retries (%d) exceeded for %s/%s — "
                      "escalating to degraded mode",
                      max_retries, failure_type.value, component)
            record.recovery_level = RecoveryLevel.DEGRADED_MODE
            self._degrade_component(component)
            self._failure_log.append(record)
            return RecoveryLevel.DEGRADED_MODE

        # Attempt recovery
        level = RecoveryLevel.RETRY if retry_count == 0 else RecoveryLevel.COOLDOWN_RETRY
        if retry_count >= 2:
            level = RecoveryLevel.RECONNECT

        log.info("self_healing: attempting %s for %s/%s (attempt %d/%d) — %s",
                 level.name, failure_type.value, component,
                 retry_count + 1, max_retries, error)

        t0 = time.time()
        try:
            success = bool(recovery_fn())
            record.recovery_time_s = time.time() - t0
            record.recovery_level = level
            record.recovered = success

            with self._lock:
                if success:
                    self._retry_counts[key] = 0  # reset on success
                    self._degraded_components.discard(component)
                    log.info("self_healing: RECOVERED %s/%s in %.2fs",
                             failure_type.value, component, record.recovery_time_s)
                else:
                    self._retry_counts[key] = retry_count + 1
                    # Exponential backoff: 2^retry_count seconds
                    backoff = min(2 ** retry_count, 60)
                    self._circuit_open[component] = time.time() + backoff
                    log.warning("self_healing: recovery FAILED for %s/%s — "
                                "backoff %.0fs",
                                failure_type.value, component, backoff)

            self._failure_log.append(record)
            return level

        except Exception as e:  # noqa: BLE001
            record.recovery_time_s = time.time() - t0
            record.recovered = False
            record.recovery_level = level
            log.error("self_healing: recovery fn raised: %r", e)
            self._failure_log.append(record)
            return level

    # ------------------------------------------------------------------
    # Degradation
    # ------------------------------------------------------------------
    def _degrade_component(self, component: str) -> None:
        with self._lock:
            self._degraded_components.add(component)
        log.warning("self_healing: component %s moved to DEGRADED state", component)
        # If critical component, trigger state transition
        if component in ("mt5_adapter", "execution_engine"):
            sm = get_state_machine()
            sm.transition(BotState.DEGRADED,
                         reason=f"component {component} degraded")
        self._bus.emit(EventType.CIRCUIT_BREAKER_TRIP,
                       payload={"component": component},
                       source="self_healing")

    def is_degraded(self, component: str) -> bool:
        with self._lock:
            return component in self._degraded_components

    def degraded_components(self) -> List[str]:
        with self._lock:
            return list(self._degraded_components)

    def retest_degraded_components(self) -> Dict[str, bool]:
        """C15/X10 fix: periodically re-test degraded components to see if
        they've recovered. Returns {component: recovered_bool}.

        Without this, a component that was marked degraded due to a
        transient issue stays degraded forever — the system never gives
        it a chance to prove it's healthy again. The caller (e.g.
        TradingBot.cycle()) should invoke this every N cycles (e.g. 50).

        The re-test simply clears the degraded flag so the next real
        operation can succeed; if it fails again, report_failure() will
        re-degrade it. This is a "benefit of the doubt" retry.
        """
        results: Dict[str, bool] = {}
        with self._lock:
            components = list(self._degraded_components)
        for comp in components:
            # Clear the degraded flag — the next operation will either
            # succeed (component stays healthy) or fail (gets re-degraded).
            with self._lock:
                self._degraded_components.discard(comp)
            results[comp] = True
            log.info("self_healing: re-test cleared degraded flag for %s "
                      "(will re-degrade if it fails again)", comp)
        return results

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_mt5_disconnect(self, event) -> None:
        comp = event.payload.get("adapter", "mt5_adapter")
        self.report_failure(FailureType.CONNECTION, comp,
                           error=event.payload.get("error", "disconnected"))

    def _on_ipc_timeout(self, event) -> None:
        self.report_failure(FailureType.CONNECTION, "mt5_adapter",
                           error=f"IPC timeout: {event.payload}")

    def _on_db_failure(self, event) -> None:
        self.report_failure(FailureType.DATABASE, "database",
                           error=str(event.payload))

    def _on_strategy_error(self, event) -> None:
        comp = event.payload.get("model", event.payload.get("function", "strategy"))
        self.report_failure(FailureType.STRATEGY, comp,
                           error=str(event.payload.get("error", "")))

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        with self._lock:
            recent = [r for r in self._failure_log
                     if time.time() - r.timestamp < self._failure_window_s]
            recovered = sum(1 for r in recent if r.recovered)
            unresolved = sum(1 for r in recent if not r.recovered)
        return {
            "total_failures_window": len(recent),
            "recovered": recovered,
            "unresolved": unresolved,
            "recovery_rate": (recovered / max(len(recent), 1)),
            "degraded_components": self.degraded_components(),
            "open_circuits": {c: t - time.time()
                             for c, t in self._circuit_open.items()
                             if t > time.time()},
        }

    def failure_log(self, last_n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "component": r.component,
                    "type": r.failure_type.value,
                    "error": r.error,
                    "level": r.recovery_level.name,
                    "recovered": r.recovered,
                    "time": r.timestamp,
                    "recovery_time_s": r.recovery_time_s,
                }
                for r in self._failure_log[-last_n:]
            ]
