"""enhancements.drawdown_recovery
=====================================================================
Day 160 — Drawdown recovery protocol.

When the system hits a drawdown, the natural human response is to
"trade harder" to recover. This is exactly wrong — it typically
deepens the drawdown. The institutional response is the opposite:
reduce size, slow down, and only resume normal sizing after the
system has PROVEN it can make money again.

Recovery phases:
  PHASE 1: HALT      — no new trades, manage existing positions only
  PHASE 2: REDUCED   — resume at 25% of normal size
  PHASE 3: NORMAL    — resume at 50% of normal size
  PHASE 4: FULL      — resume at 100% (recovered)

Transitions:
  - HALT → REDUCED   : after N cycles at halt with no further loss
  - REDUCED → NORMAL : after profitable for N cycles
  - NORMAL → FULL    : after profitable for N cycles AND drawdown < 50% of original
  - Any → HALT       : if drawdown deepens by another X%

The protocol is STATEFUL and PERSISTED so a restart doesn't reset it.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False

log = get_logger("enhancements.drawdown_recovery")


class RecoveryPhase(str, Enum):
    NORMAL = "normal"          # full size, no drawdown
    REDUCED = "reduced"        # 25% size, recovering
    HALT = "halt"              # no new trades
    FULL = "full"              # 100% size, recovered (same as NORMAL operationally)


@dataclass
class RecoveryState:
    phase: str = RecoveryPhase.FULL.value
    phase_entered_at: float = 0.0
    cycles_in_phase: int = 0
    drawdown_at_trigger: float = 0.0
    peak_drawdown: float = 0.0
    cycles_profitable: int = 0
    cycles_unprofitable: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class DrawdownRecoveryProtocol:
    def __init__(
        self,
        state_path: str = "data/drawdown_recovery.json",
        halt_threshold_pct: float = 0.10,         # 10% drawdown → HALT
        reduced_threshold_pct: float = 0.05,      # 5% drawdown → REDUCED
        cycles_to_resume_reduced: int = 50,       # 50 cycles at HALT with no further loss
        cycles_to_resume_normal: int = 100,       # 100 profitable cycles at REDUCED
        cycles_to_resume_full: int = 200,         # 200 profitable cycles at NORMAL
        size_multiplier_halt: float = 0.0,
        size_multiplier_reduced: float = 0.25,
        size_multiplier_normal: float = 0.50,
        size_multiplier_full: float = 1.0,
    ) -> None:
        self.state_path = state_path
        self.halt_threshold = float(halt_threshold_pct)
        self.reduced_threshold = float(reduced_threshold_pct)
        self.cycles_to_reduced = int(cycles_to_resume_reduced)
        self.cycles_to_normal = int(cycles_to_resume_normal)
        self.cycles_to_full = int(cycles_to_resume_full)
        self.size_halt = float(size_multiplier_halt)
        self.size_reduced = float(size_multiplier_reduced)
        self.size_normal = float(size_multiplier_normal)
        self.size_full = float(size_multiplier_full)
        self.state = RecoveryState(phase_entered_at=time.time())
        self._load()

    # ----------------------------------------------------------------
    def evaluate(
        self,
        current_drawdown_pct: float,
        cycle_profitable: bool,
    ) -> dict[str, Any]:
        """Evaluate the protocol for this cycle. Returns current phase + size multiplier."""
        self.state.cycles_in_phase += 1
        self.state.peak_drawdown = max(self.state.peak_drawdown, current_drawdown_pct)
        if cycle_profitable:
            self.state.cycles_profitable += 1
            self.state.cycles_unprofitable = 0
        else:
            self.state.cycles_unprofitable += 1
            self.state.cycles_profitable = 0

        current_phase = RecoveryPhase(self.state.phase)

        # Check for drawdown deepening → force HALT
        if current_drawdown_pct >= self.halt_threshold:
            if current_phase != RecoveryPhase.HALT:
                self._transition(RecoveryPhase.HALT,
                                  reason=f"drawdown {current_drawdown_pct:.2%} >= {self.halt_threshold:.2%}")
                self.state.drawdown_at_trigger = current_drawdown_pct
        elif current_drawdown_pct >= self.reduced_threshold:
            if current_phase not in (RecoveryPhase.HALT, RecoveryPhase.REDUCED):
                self._transition(RecoveryPhase.REDUCED,
                                  reason=f"drawdown {current_drawdown_pct:.2%} >= {self.reduced_threshold:.2%}")
                self.state.drawdown_at_trigger = current_drawdown_pct
        else:
            # Drawdown is below reduced threshold — check for phase advancement
            if current_phase == RecoveryPhase.HALT:
                # Need N cycles with no further loss
                if (self.state.cycles_in_phase >= self.cycles_to_reduced
                        and current_drawdown_pct < self.state.drawdown_at_trigger):
                    self._transition(RecoveryPhase.REDUCED,
                                      reason=f"halt cooldown complete ({self.cycles_to_reduced} cycles)")
            elif current_phase == RecoveryPhase.REDUCED:
                if (self.state.cycles_profitable >= self.cycles_to_normal
                        and current_drawdown_pct < self.reduced_threshold * 0.5):
                    self._transition(RecoveryPhase.NORMAL,
                                      reason=f"reduced phase profitable for {self.cycles_to_normal} cycles")
            elif current_phase == RecoveryPhase.NORMAL:
                if (self.state.cycles_profitable >= self.cycles_to_full
                        and current_drawdown_pct < self.reduced_threshold * 0.25):
                    self._transition(RecoveryPhase.FULL,
                                      reason=f"normal phase profitable for {self.cycles_to_full} cycles")

        size_mult = self._size_multiplier()
        self._save()
        return {
            "phase": self.state.phase,
            "size_multiplier": size_mult,
            "cycles_in_phase": self.state.cycles_in_phase,
            "current_drawdown_pct": float(current_drawdown_pct),
            "peak_drawdown": self.state.peak_drawdown,
            "cycles_profitable": self.state.cycles_profitable,
            "cycles_unprofitable": self.state.cycles_unprofitable,
        }

    # ----------------------------------------------------------------
    def _size_multiplier(self) -> float:
        phase = RecoveryPhase(self.state.phase)
        if phase == RecoveryPhase.HALT:
            return self.size_halt
        if phase == RecoveryPhase.REDUCED:
            return self.size_reduced
        if phase == RecoveryPhase.NORMAL:
            return self.size_normal
        return self.size_full

    # ----------------------------------------------------------------
    def _transition(self, new_phase: RecoveryPhase, reason: str) -> None:
        old_phase = self.state.phase
        self.state.phase = new_phase.value
        self.state.phase_entered_at = time.time()
        self.state.cycles_in_phase = 0
        self.state.history.append({
            "ts": time.time(),
            "from": old_phase,
            "to": new_phase.value,
            "reason": reason,
        })
        log.warning("DRAWDOWN RECOVERY: %s → %s (%s)", old_phase, new_phase.value, reason)

    # ----------------------------------------------------------------
    def manual_reset(self) -> None:
        """Operator can force a reset to FULL phase."""
        self.state = RecoveryState(phase=RecoveryPhase.FULL.value,
                                    phase_entered_at=time.time())
        self._save()
        log.info("Drawdown recovery manually reset to FULL")

    # ----------------------------------------------------------------
    @property
    def current_phase(self) -> RecoveryPhase:
        return RecoveryPhase(self.state.phase)

    @property
    def size_multiplier(self) -> float:
        return self._size_multiplier()

    # ----------------------------------------------------------------
    def _save(self) -> None:
        """Critical #3 fix: save with file locking to prevent corruption
        from concurrent access (multi-process or multi-thread)."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        lock_path = self.state_path + ".lock"
        try:
            # Acquire cross-process lock.
            with open(lock_path, "a") as lock_fh:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                elif _HAS_PORTALOCKER:
                    portalocker.lock(lock_fh, portalocker.LOCK_EX)
                try:
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(self.state.to_dict(), f, indent=2, default=str)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, self.state_path)
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                    elif _HAS_PORTALOCKER:
                        portalocker.unlock(lock_fh)
        except Exception as e:  # noqa: BLE001
            log.warning("recovery state save failed: %r", e)

    def _load(self) -> None:
        """Critical #3 fix: load with file locking to prevent reading
        a partially-written file."""
        if not os.path.isfile(self.state_path):
            return
        lock_path = self.state_path + ".lock"
        try:
            with open(lock_path, "a") as lock_fh:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                elif _HAS_PORTALOCKER:
                    portalocker.lock(lock_fh, portalocker.LOCK_EX)
                try:
                    with open(self.state_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.state = RecoveryState(**{k: v for k, v in data.items()
                                                    if k in RecoveryState.__dataclass_fields__})
                    log.info("Drawdown recovery loaded: phase=%s cycles=%d",
                             self.state.phase, self.state.cycles_in_phase)
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                    elif _HAS_PORTALOCKER:
                        portalocker.unlock(lock_fh)
        except Exception as e:  # noqa: BLE001
            log.warning("recovery state load failed: %r", e)
