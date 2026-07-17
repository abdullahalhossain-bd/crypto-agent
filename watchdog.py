"""trading_bot.watchdog
=====================================================================
Day 7 — Safety rails.

Three independent safety mechanisms:

  1. KillSwitch    — external file presence halts the bot.
                     Operator can `touch data/KILL_SWITCH` from a shell
                     without needing to attach to the process.

  2. Heartbeat     — main loop writes a timestamp every successful cycle.
                     If no heartbeat for `heartbeat_timeout_s`, the
                     watchdog raises HeartbeatTimeout so the main loop
                     can decide whether to restart or quit.

  3. ErrorBudget   — counter of consecutive exceptions; when it exceeds
                     `max_consecutive_errors`, the bot self-halts.

Audit Batch 1 remediation (C15, H1, M7, L10):
  - The class is now thread-safe (`threading.RLock` around mutable state).
  - `check_heartbeat()` caches the on-disk timestamp with a TTL (L10 fix)
    so we don't hit disk on every call.
  - `record_error()` logs full context (consecutive count, max budget,
    exception traceback) BEFORE raising `ErrorBudgetExceeded` so operators
    have a forensic trail (M7 fix).
  - The class is the single source of truth — `main.py` no longer
    duplicates kill-switch / heartbeat / error-budget logic (C15 fix).
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("watchdog")


class KillSwitchActive(RuntimeError):
    """Raised when the kill-switch file exists."""


class HeartbeatTimeout(RuntimeError):
    """Raised when no successful cycle within the timeout window."""


class ErrorBudgetExceeded(RuntimeError):
    """Raised when too many consecutive errors."""


# ----------------------------------------------------------------------
# Audit-fix C15 / H1 / M7 / L10:
#  - RLock instead of bare Lock (RLock allows re-entrancy from the same
#    thread, which matters because record_error() may be called from
#    within a code path that already holds the lock in a future caller).
#  - Heartbeat file reads are cached with a TTL so check_heartbeat()
#    doesn't become a disk-I/O bottleneck.
#  - record_error() logs full context (traceback included) before raising.
# ----------------------------------------------------------------------
@dataclass
class Watchdog:
    kill_switch_file: str
    heartbeat_file: str
    heartbeat_timeout_s: float
    max_consecutive_errors: int

    _consecutive_errors: int = 0
    _last_ok: float = 0.0
    _heartbeat_cache: Optional[float] = None
    _heartbeat_cache_ts: float = 0.0
    _heartbeat_cache_ttl: float = 1.0  # seconds; L10 fix
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _ks_cache: Optional[bool] = None
    _ks_cache_ts: float = 0.0
    _ks_cache_ttl: float = 0.5  # seconds

    # ---- Kill switch ----
    def check_kill_switch(self) -> None:
        """Raise KillSwitchActive if the kill-switch file exists.

        L10 fix: cache the result for `_ks_cache_ttl` seconds so we don't
        stat() the file on every cycle iteration.
        """
        now = time.monotonic()
        exists: bool
        with self._lock:
            if (self._ks_cache is not None
                    and now - self._ks_cache_ts < self._ks_cache_ttl):
                exists = self._ks_cache
            else:
                exists = os.path.isfile(self.kill_switch_file)
                self._ks_cache = exists
                self._ks_cache_ts = now
        if exists:
            raise KillSwitchActive(
                f"kill switch file present: {self.kill_switch_file}")

    def arm_kill_switch(self, reason: str = "manual") -> None:
        """Operator-side helper: create the kill-switch file."""
        os.makedirs(os.path.dirname(self.kill_switch_file) or ".", exist_ok=True)
        with open(self.kill_switch_file, "w", encoding="utf-8") as f:
            f.write(f"armed at {time.time()} reason={reason}\n")
        with self._lock:
            self._ks_cache = True
            self._ks_cache_ts = time.monotonic()
        log.warning("KILL SWITCH ARMED: %s", reason)

    def disarm_kill_switch(self) -> None:
        try:
            os.remove(self.kill_switch_file)
            log.info("Kill switch disarmed")
        except FileNotFoundError:
            pass
        with self._lock:
            self._ks_cache = False
            self._ks_cache_ts = time.monotonic()

    # ---- Heartbeat ----
    def heartbeat(self) -> None:
        os.makedirs(os.path.dirname(self.heartbeat_file) or ".", exist_ok=True)
        with open(self.heartbeat_file, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        with self._lock:
            self._last_ok = time.time()
            self._heartbeat_cache = self._last_ok
            self._heartbeat_cache_ts = time.monotonic()
            self._consecutive_errors = 0

    def check_heartbeat(self) -> None:
        """If we haven't heartbeated recently, raise.

        L10 fix: cache the on-disk timestamp for `_heartbeat_cache_ttl`
        seconds so we don't re-read the file on every call.
        """
        now_mono = time.monotonic()
        with self._lock:
            if (self._heartbeat_cache is not None
                    and now_mono - self._heartbeat_cache_ts < self._heartbeat_cache_ttl):
                last_ok = self._heartbeat_cache
            elif self._last_ok != 0.0:
                last_ok = self._last_ok
            else:
                # Cold start: read from disk once.
                try:
                    with open(self.heartbeat_file, "r", encoding="utf-8") as f:
                        last_ok = float(f.read().strip() or 0.0)
                except (FileNotFoundError, ValueError):
                    last_ok = time.time()
                self._heartbeat_cache = last_ok
                self._heartbeat_cache_ts = now_mono
                self._last_ok = last_ok
        if time.time() - last_ok > self.heartbeat_timeout_s:
            raise HeartbeatTimeout(
                f"no heartbeat for {time.time() - last_ok:.1f}s "
                f"(> {self.heartbeat_timeout_s}s)"
            )

    # ---- Error budget ----
    def record_error(self, exc: BaseException) -> None:
        """Increment the error streak; raise ErrorBudgetExceeded when over.

        M7 fix: log full context (traceback, current streak, max budget)
        BEFORE raising so operators have a forensic trail in the logs.
        """
        with self._lock:
            self._consecutive_errors += 1
            current = self._consecutive_errors
            max_budget = self.max_consecutive_errors
        log.error("consecutive error %d/%d: %r\n%s",
                  current, max_budget, exc,
                  "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        if current >= max_budget:
            # M7 fix: log a detailed final-state message before raising.
            log.error("ERROR BUDGET EXCEEDED: %d consecutive errors (max=%d). "
                      "Bot will halt. Last exception: %r",
                      current, max_budget, exc)
            raise ErrorBudgetExceeded(
                f"{current} consecutive errors — halting "
                f"(last exc: {exc!r})") from exc

    def record_success(self) -> None:
        with self._lock:
            if self._consecutive_errors > 0:
                log.info("error budget reset (was %d)", self._consecutive_errors)
            self._consecutive_errors = 0

    # ---- Diagnostics ----
    def snapshot(self) -> dict:
        """Return a thread-safe snapshot of the watchdog's internal state."""
        with self._lock:
            return {
                "consecutive_errors": self._consecutive_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
                "last_heartbeat": self._last_ok,
                "kill_switch_armed": (
                    self._ks_cache if self._ks_cache is not None
                    else os.path.isfile(self.kill_switch_file)
                ),
            }
