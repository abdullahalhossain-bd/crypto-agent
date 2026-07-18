"""architecture/state_machine.py
=====================================================================
Industrial-Grade State Machine for Bot Lifecycle (Improvement #2)
=====================================================================
Defines every possible state the bot can be in, and the legal
transitions between them. The bot can NEVER be in an undefined state —
every transition is validated, audited, and emitted to the EventBus.

State Lifecycle:
    BOOT → CONNECTING → SYNCING → WARMUP → LIVE
                                            ↓
                                         DEGRADED ⇄ RECOVERY
                                            ↓
                                         EMERGENCY → SHUTDOWN

States:
    BOOT        — initial state, components being constructed
    CONNECTING  — MT5 connection in progress (retry/backoff)
    SYNCING     — fetching symbols, contracts, history from broker
    WARMUP      — indicators warming up (need N bars before signals)
    LIVE        — fully operational, placing orders
    DEGRADED    — partial failure (e.g. 1 of N symbols unreachable)
    RECOVERY    — restoring from snapshot after crash
    EMERGENCY   — kill-switch armed or critical loss event
    SHUTDOWN    — graceful shutdown in progress (final state)

Each state has:
    - allowed_transitions: set of states it can move to
    - entry_hooks: functions called when entering this state
    - exit_hooks: functions called when leaving this state
    - max_duration_s: alert if bot stays in this state too long
    - can_trade: whether orders can be placed in this state
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.state_machine")


class BotState(str, Enum):
    BOOT = "BOOT"
    CONNECTING = "CONNECTING"
    SYNCING = "SYNCING"
    WARMUP = "WARMUP"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    RECOVERY = "RECOVERY"
    EMERGENCY = "EMERGENCY"
    SHUTDOWN = "SHUTDOWN"
    HALTED = "HALTED"   # terminal: requires manual restart


# Legal transitions (state-machine edges)
LEGAL_TRANSITIONS: Dict[BotState, Set[BotState]] = {
    BotState.BOOT:        {BotState.CONNECTING, BotState.SHUTDOWN},
    BotState.CONNECTING:  {BotState.SYNCING, BotState.DEGRADED,
                           BotState.EMERGENCY, BotState.SHUTDOWN,
                           BotState.RECOVERY},
    BotState.SYNCING:     {BotState.WARMUP, BotState.DEGRADED,
                           BotState.EMERGENCY, BotState.SHUTDOWN},
    BotState.WARMUP:      {BotState.LIVE, BotState.DEGRADED,
                           BotState.EMERGENCY, BotState.SHUTDOWN},
    BotState.LIVE:        {BotState.DEGRADED, BotState.EMERGENCY,
                           BotState.SHUTDOWN},
    BotState.DEGRADED:    {BotState.LIVE, BotState.RECOVERY,
                           BotState.SYNCING, BotState.WARMUP,
                           BotState.EMERGENCY, BotState.SHUTDOWN},
    BotState.RECOVERY:    {BotState.LIVE, BotState.DEGRADED,
                           BotState.EMERGENCY, BotState.SHUTDOWN},
    BotState.EMERGENCY:   {BotState.RECOVERY, BotState.SHUTDOWN,
                           BotState.HALTED},
    BotState.SHUTDOWN:    {BotState.HALTED},
    BotState.HALTED:      set(),  # terminal
}


@dataclass
class StateMetadata:
    """Per-state config: how long can we stay here, can we trade?"""
    can_trade: bool = False
    can_fetch_data: bool = False
    max_duration_s: float = 0.0  # 0 = unlimited
    description: str = ""
    severity: str = "info"  # info, warning, critical


STATE_META: Dict[BotState, StateMetadata] = {
    BotState.BOOT:       StateMetadata(False, False, 60,
                                       "Bot starting up — constructing components", "info"),
    BotState.CONNECTING: StateMetadata(False, False, 120,
                                       "Connecting to MT5 broker", "warning"),
    BotState.SYNCING:    StateMetadata(False, True, 180,
                                       "Syncing symbols, contracts, history", "info"),
    BotState.WARMUP:     StateMetadata(False, True, 600,
                                       "Indicators warming up — needs N bars", "info"),
    BotState.LIVE:       StateMetadata(True, True, 0,
                                       "Fully operational — placing orders", "info"),
    BotState.DEGRADED:   StateMetadata(False, True, 300,
                                       "Partial failure — running in safe mode", "warning"),
    BotState.RECOVERY:   StateMetadata(False, False, 600,
                                       "Restoring from snapshot", "warning"),
    BotState.EMERGENCY:  StateMetadata(False, False, 0,
                                       "Kill-switch armed or critical loss", "critical"),
    BotState.SHUTDOWN:   StateMetadata(False, False, 60,
                                       "Graceful shutdown in progress", "warning"),
    BotState.HALTED:     StateMetadata(False, False, 0,
                                       "HALTED — requires manual restart", "critical"),
}


@dataclass
class StateSnapshot:
    """Snapshot of the state machine at a point in time."""
    current: BotState = BotState.BOOT
    previous: Optional[BotState] = None
    entered_at: float = field(default_factory=time.time)
    transition_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class StateMachine:
    """Thread-safe state machine for the bot lifecycle.

    Usage:
        sm = StateMachine()
        sm.transition(BotState.CONNECTING)
        if sm.can_trade():
            place_orders()
        sm.transition(BotState.SHUTDOWN)
    """

    def __init__(self, bus: Optional[EventBus] = None):
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        self._snap = StateSnapshot()
        # Hook registry: state -> list of callbacks
        self._entry_hooks: Dict[BotState, List[Callable]] = {}
        self._exit_hooks: Dict[BotState, List[Callable]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current(self) -> BotState:
        with self._lock:
            return self._snap.current

    @property
    def previous(self) -> Optional[BotState]:
        with self._lock:
            return self._snap.previous

    @property
    def entered_at(self) -> float:
        with self._lock:
            return self._snap.entered_at

    @property
    def transition_count(self) -> int:
        with self._lock:
            return self._snap.transition_count

    def time_in_state(self) -> float:
        return time.time() - self.entered_at

    def metadata(self) -> StateMetadata:
        return STATE_META.get(self.current, StateMetadata())

    # ------------------------------------------------------------------
    # Capability queries
    # ------------------------------------------------------------------
    def can_trade(self) -> bool:
        return self.metadata().can_trade

    def can_fetch_data(self) -> bool:
        return self.metadata().can_fetch_data

    def is_terminal(self) -> bool:
        return self.current in (BotState.HALTED,)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def on_entry(self, state: BotState, callback: Callable) -> None:
        self._entry_hooks.setdefault(state, []).append(callback)

    def on_exit(self, state: BotState, callback: Callable) -> None:
        self._exit_hooks.setdefault(state, []).append(callback)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def transition(self, new_state: BotState,
                   reason: str = "") -> bool:
        """Attempt a state transition. Returns True if successful.

        Illegal transitions are rejected and logged. Every successful
        transition emits a STATE_TRANSITION event.
        """
        with self._lock:
            old = self._snap.current
            if new_state == old:
                log.debug("state_machine: already in %s (no-op)", old.value)
                return True
            if new_state not in LEGAL_TRANSITIONS.get(old, set()):
                log.error("state_machine: ILLEGAL transition %s → %s "
                          "(allowed: %s)",
                          old.value, new_state.value,
                          [s.value for s in LEGAL_TRANSITIONS.get(old, set())])
                return False

            # Fire exit hooks for old state
            for cb in self._exit_hooks.get(old, []):
                try:
                    cb(self._snap)
                except Exception as e:  # noqa: BLE001
                    log.warning("exit hook %s raised: %r",
                                getattr(cb, "__name__", "?"), e)

            # Apply transition
            self._snap.previous = old
            self._snap.current = new_state
            self._snap.entered_at = time.time()
            self._snap.transition_count += 1
            self._snap.history.append({
                "from": old.value,
                "to": new_state.value,
                "reason": reason,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "duration_in_old_s": time.time() - (
                    self._snap.history[-1]["timestamp_parsed"]
                    if self._snap.history and "timestamp_parsed" in self._snap.history[-1]
                    else self._snap.entered_at
                ),
            })

            log.info("state_machine: %s → %s (reason: %s) [transition #%d]",
                     old.value, new_state.value, reason or "—",
                     self._snap.transition_count)

            # Emit event
            self._bus.emit(
                EventType.STATE_TRANSITION,
                payload={
                    "from": old.value,
                    "to": new_state.value,
                    "reason": reason,
                    "transition_count": self._snap.transition_count,
                    "can_trade": self.metadata().can_trade,
                },
                source="state_machine",
            )

        # Fire entry hooks (outside lock to avoid deadlocks)
        for cb in self._entry_hooks.get(new_state, []):
            try:
                cb(self._snap)
            except Exception as e:  # noqa: BLE001
                log.warning("entry hook %s raised: %r",
                            getattr(cb, "__name__", "?"), e)

        return True

    # ------------------------------------------------------------------
    # Watchdog: alert if we're stuck in a state too long
    # ------------------------------------------------------------------
    def check_state_health(self, auto_transition: bool = True) -> Optional[str]:
        """Returns a warning string if we've been in the current state
        longer than max_duration_s. None if healthy.

        H10 fix: previously this only returned a warning string and left
        it entirely up to the caller to notice the log line and act.
        Nothing in the main loop actually transitioned the bot out of a
        stuck state — it could sit in WARMUP (or any other state with a
        max_duration_s) indefinitely. Now, when `auto_transition` is True
        (the default) and we've exceeded max_duration_s, we attempt an
        automatic transition to DEGRADED (a safe, non-trading state) IF
        that transition is legal from the current state. If it isn't
        legal (e.g. stuck in BOOT, which can only go to CONNECTING or
        SHUTDOWN), we still return the warning so the operator/caller can
        decide — auto-transition being illegal doesn't mean the warning
        should be suppressed.
        """
        meta = self.metadata()
        if meta.max_duration_s <= 0:
            return None
        elapsed = self.time_in_state()
        if elapsed > meta.max_duration_s:
            msg = (f"Stuck in {self.current.value} for {elapsed:.0f}s "
                    f"(max {meta.max_duration_s}s)")
            if auto_transition and self.current != BotState.DEGRADED:
                with self._lock:
                    can_degrade = BotState.DEGRADED in LEGAL_TRANSITIONS.get(self.current, set())
                if can_degrade:
                    ok = self.transition(
                        BotState.DEGRADED,
                        reason=f"watchdog: {msg} — auto-transitioned to DEGRADED")
                    if ok:
                        msg += " — auto-transitioned to DEGRADED"
            return msg
        return None

    # ------------------------------------------------------------------
    # Snapshot for recovery
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current": self._snap.current.value,
                "previous": self._snap.previous.value if self._snap.previous else None,
                "entered_at": self._snap.entered_at,
                "transition_count": self._snap.transition_count,
                "history_tail": self._snap.history[-20:],
            }


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------
_GLOBAL_SM: Optional[StateMachine] = None


def get_state_machine() -> StateMachine:
    global _GLOBAL_SM
    if _GLOBAL_SM is None:
        _GLOBAL_SM = StateMachine()
    return _GLOBAL_SM