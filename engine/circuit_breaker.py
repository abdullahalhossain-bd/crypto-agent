"""engine.circuit_breaker
=====================================================================
Day 26 — Circuit breakers.

Three breakers, all derived from the same base class:

  1. `EquityDrawdownBreaker` — opens when rolling DD > threshold
  2. `ErrorRateBreaker`      — opens when error rate in last N cycles > X%
  3. `LatencyBreaker`        — opens when median cycle latency > threshold

When OPEN, downstream code MUST refuse to trade. After a cooldown,
the breaker moves to HALF_OPEN and allows one probe; if the probe
succeeds, the breaker CLOSES; if it fails, it re-OPENS.

Pattern borrowed from Hystrix / Resilience4j — well-tested.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger("engine.circuit_breaker")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    cooldown_s: float = 60.0
    success_threshold: int = 2  # successes needed in HALF_OPEN to close
    _state: BreakerState = BreakerState.CLOSED
    _failure_count: int = 0
    _success_count: int = 0
    _opened_at: float = 0.0

    @property
    def state(self) -> BreakerState:
        # Auto-transition OPEN → HALF_OPEN after cooldown
        if self._state == BreakerState.OPEN:
            if time.time() - self._opened_at >= self.cooldown_s:
                self._state = BreakerState.HALF_OPEN
                self._success_count = 0
                log.info("BREAKER %s OPEN -> HALF_OPEN", self.name)
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def allow(self) -> bool:
        """Should the caller proceed?"""
        s = self.state
        if s == BreakerState.CLOSED:
            return True
        if s == BreakerState.HALF_OPEN:
            return True  # allow a probe
        return False

    def record_success(self) -> None:
        if self._state == BreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = BreakerState.CLOSED
                self._failure_count = 0
                log.info("BREAKER %s HALF_OPEN -> CLOSED", self.name)
        else:
            # Reset failure count on any success while CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        if self._state == BreakerState.HALF_OPEN:
            # Failed probe — reopen
            self._state = BreakerState.OPEN
            self._opened_at = time.time()
            log.warning("BREAKER %s HALF_OPEN -> OPEN (probe failed)", self.name)
            return
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.time()
            log.warning("BREAKER %s CLOSED -> OPEN (failures=%d)",
                        self.name, self._failure_count)

    def reset(self) -> None:
        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "opened_at": self._opened_at,
        }


# ----------------------------------------------------------------------
# Specialised breakers
# ----------------------------------------------------------------------
class EquityDrawdownBreaker(CircuitBreaker):
    """Trips when rolling max drawdown exceeds threshold."""
    def __init__(self, threshold: float = 0.10,
                 cooldown_s: float = 300.0) -> None:
        super().__init__(
            name="equity_drawdown",
            failure_threshold=1,  # 1 breach is enough
            cooldown_s=cooldown_s,
        )
        self.threshold = float(threshold)
        self._equity_history: list[float] = []

    def record_equity(self, equity: float) -> None:
        self._equity_history.append(equity)
        if len(self._equity_history) > 5000:
            self._equity_history = self._equity_history[-5000:]
        if len(self._equity_history) < 2:
            return
        import numpy as np
        eqs = np.array(self._equity_history)
        running_max = np.maximum.accumulate(eqs)
        dd = (eqs - running_max) / np.where(running_max > 0, running_max, 1.0)
        current_dd = abs(float(dd[-1]))
        if current_dd > self.threshold:
            self.record_failure()
        else:
            self.record_success()


class ErrorRateBreaker(CircuitBreaker):
    """Trips when error rate over last N cycles exceeds threshold."""
    def __init__(self, window: int = 50, threshold: float = 0.5,
                 cooldown_s: float = 60.0) -> None:
        super().__init__(name="error_rate",
                         failure_threshold=1, cooldown_s=cooldown_s)
        self.window = int(window)
        self.threshold = float(threshold)
        self._events: deque[bool] = deque(maxlen=self.window)

    def record_cycle(self, ok: bool) -> None:
        self._events.append(ok)
        if len(self._events) < 10:
            return
        err_rate = sum(1 for x in self._events if not x) / len(self._events)
        if err_rate >= self.threshold:
            self.record_failure()
        else:
            self.record_success()


class LatencyBreaker(CircuitBreaker):
    """Trips when median cycle latency (in seconds) exceeds threshold."""
    def __init__(self, threshold_s: float = 5.0,
                 window: int = 50,
                 cooldown_s: float = 60.0) -> None:
        super().__init__(name="latency",
                         failure_threshold=1, cooldown_s=cooldown_s)
        self.threshold_s = float(threshold_s)
        self._latencies: deque[float] = deque(maxlen=window)

    def record_latency(self, latency_s: float) -> None:
        self._latencies.append(latency_s)
        if len(self._latencies) < 10:
            return
        import statistics
        med = statistics.median(self._latencies)
        if med > self.threshold_s:
            self.record_failure()
        else:
            self.record_success()
