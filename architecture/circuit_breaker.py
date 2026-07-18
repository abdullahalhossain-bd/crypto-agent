"""architecture/circuit_breaker.py — top-of-cycle circuit breakers.

Phase 4 consolidation. Merges the 3 breakers from the archived
engine/circuit_breaker.py (EquityDrawdown, ErrorRate, Latency) and adds:
  - ConsecutiveLossBreaker (trips after N consecutive losing trades)
  - BrokerDisconnectBreaker (trips when MT5 connection drops)
  - SlippageBreaker (trips when actual fill price deviates from expected
    by more than X bps)

These run at the TOP of TradingBot.cycle(), BEFORE the risk pipeline.
If any breaker is OPEN, the cycle is skipped and a loud log is emitted.
This is distinct from the risk pipeline gates (which run per-trade):
breakers protect the whole bot from systemic issues, not individual
trade-selection issues.

Pattern: Hystrix-style state machine (CLOSED → OPEN → HALF_OPEN → CLOSED).
After a breaker opens, it auto-transitions to HALF_OPEN after cooldown_s
and allows one probe cycle; if the probe succeeds, it closes; if it fails,
it re-opens.

All breakers expose:
  - .state: BreakerState (CLOSED, OPEN, HALF_OPEN)
  - .is_open: bool (True = block trading)
  - .allow(): bool (True = proceed this cycle)
  - .record_success() / .record_failure(): update internal state
  - .to_dict(): for status/monitoring
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

# Minor #7 fix: move numpy import to module level (was inside record_equity).
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.circuit_breaker")


class BreakerState(str, Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"            # tripped — block all trading
    HALF_OPEN = "half_open"  # cooldown elapsed — allow one probe cycle


@dataclass
class CircuitBreaker:
    """Base circuit breaker — Hystrix-style state machine.

    Subclasses override `record_*()` to feed in telemetry and call
    `record_failure()` / `record_success()` on the base class.
    """
    name: str
    failure_threshold: int = 5       # failures before opening
    cooldown_s: float = 60.0         # OPEN → HALF_OPEN after this many seconds
    success_threshold: int = 2       # successes in HALF_OPEN to close
    # C18 fix: not every open breaker should halt the ENTIRE cycle
    # (including SL/TP position management). Critical breakers (equity
    # drawdown, broker disconnect) block everything; non-critical ones
    # (slippage, latency) only block NEW trade entries.
    critical: bool = True
    _state: BreakerState = BreakerState.CLOSED
    _failure_count: int = 0
    _success_count: int = 0
    _opened_at: float = 0.0
    _last_reason: str = ""

    @property
    def state(self) -> BreakerState:
        # Auto-transition OPEN → HALF_OPEN after cooldown
        if self._state == BreakerState.OPEN:
            if time.time() - self._opened_at >= self.cooldown_s:
                self._state = BreakerState.HALF_OPEN
                self._success_count = 0
                log.info("breaker %s: OPEN → HALF_OPEN (cooldown elapsed)",
                         self.name)
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def allow(self) -> bool:
        """Should the caller proceed this cycle?

        CLOSED → True (normal)
        OPEN   → False (blocked)
        HALF_OPEN → True (probe allowed)
        """
        s = self.state
        return s != BreakerState.OPEN

    def record_success(self) -> None:
        if self._state == BreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = BreakerState.CLOSED
                self._failure_count = 0
                log.info("breaker %s: HALF_OPEN → CLOSED (probe succeeded)",
                         self.name)
        else:
            # Reset failure count on any success while CLOSED
            self._failure_count = 0

    def record_failure(self, reason: str = "") -> None:
        if self._state == BreakerState.HALF_OPEN:
            # Failed probe — reopen
            self._state = BreakerState.OPEN
            self._opened_at = time.time()
            self._last_reason = reason
            log.warning("breaker %s: HALF_OPEN → OPEN (probe failed: %s)",
                        self.name, reason)
            return
        self._failure_count += 1
        self._last_reason = reason
        if self._failure_count >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.time()
            log.warning("breaker %s: CLOSED → OPEN (failures=%d, reason=%s)",
                        self.name, self._failure_count, reason)

    def reset(self) -> None:
        """Manual reset — operator-initiated only."""
        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_reason = ""
        log.info("breaker %s: manual reset → CLOSED", self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "critical": self.critical,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "opened_at": self._opened_at,
            "last_reason": self._last_reason,
        }


# ----------------------------------------------------------------------
# Specialized breakers
# ----------------------------------------------------------------------
class EquityDrawdownBreaker(CircuitBreaker):
    """Trips when rolling max drawdown exceeds threshold.

    Default: 10% drawdown from peak equity trips the breaker for 5 minutes.
    This is a systemic backstop — the per-trade DrawdownGate in RiskPipeline
    handles individual trade rejection; this breaker halts ALL trading when
    the account is in a tailspin.
    """
    def __init__(self, threshold: float = 0.10,
                 cooldown_s: float = 300.0) -> None:
        super().__init__(
            name="equity_drawdown",
            failure_threshold=1,  # 1 breach is enough — drawdown is serious
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
        # Minor #7 fix: numpy imported at module level.
        if not _HAS_NUMPY:
            return
        eqs = np.array(self._equity_history)
        running_max = np.maximum.accumulate(eqs)
        dd = (eqs - running_max) / np.where(running_max > 0, running_max, 1.0)
        current_dd = abs(float(dd[-1]))
        if current_dd > self.threshold:
            self.record_failure(f"drawdown {current_dd*100:.1f}% > {self.threshold*100:.1f}%")
        else:
            self.record_success()


class ErrorRateBreaker(CircuitBreaker):
    """Trips when error rate over last N cycles exceeds threshold.

    Review Point 1: tightened from 50% over 50 cycles to 3 consecutive
    errors (immediate trip) + a rolling-window check. The old default
    allowed 25 errors before tripping — too loose for real capital.

    Default: 3 consecutive cycle errors trips immediately (failure_threshold=3).
    Also checks rolling 20-cycle window at 50% error rate.
    """
    def __init__(self, window: int = 20, threshold: float = 0.5,
                 consecutive_threshold: int = 3,
                 cooldown_s: float = 60.0) -> None:
        super().__init__(name="error_rate",
                         failure_threshold=consecutive_threshold,
                         cooldown_s=cooldown_s)
        self.window = int(window)
        self.threshold = float(threshold)
        self._events: Deque[bool] = deque(maxlen=self.window)
        self._consecutive = 0

    def record_cycle(self, ok: bool) -> None:
        self._events.append(ok)
        if ok:
            self._consecutive = 0
            self.record_success()
        else:
            self._consecutive += 1
            # Immediate trip on N consecutive errors (Review Point 1)
            if self._consecutive >= self.failure_threshold:
                self.record_failure(
                    f"{self._consecutive} consecutive cycle errors")
                self._consecutive = 0  # reset after trip
                return
            # Rolling-window check
            if len(self._events) >= 10:
                err_rate = sum(1 for x in self._events if not x) / len(self._events)
                # C9 fix: strictly greater-than so an exact-threshold rate
                # (e.g. 0.5 == 0.5) does not falsely trip the breaker.
                if err_rate > self.threshold:
                    self.record_failure(
                        f"error rate {err_rate*100:.0f}% > {self.threshold*100:.0f}%")


class LatencyBreaker(CircuitBreaker):
    """Trips when median cycle latency exceeds threshold.

    Review fix: raised from 5s to 60s. With 100+ symbols and MT5 API
    round-trips, a cycle legitimately takes 10-30s even when parallelized.
    The old 5s threshold would trip the breaker on every cycle, halting
    all trading for no real reason. 60s is a genuine "something is wrong"
    threshold — if a cycle takes >60s, MT5 IPC is truly degraded.
    """
    def __init__(self, threshold_s: float = 60.0,
                 window: int = 20,
                 cooldown_s: float = 120.0) -> None:
        super().__init__(name="latency",
                         failure_threshold=1, cooldown_s=cooldown_s,
                         critical=False)
        self.threshold_s = float(threshold_s)
        self._latencies: Deque[float] = deque(maxlen=window)

    def record_latency(self, latency_s: float) -> None:
        self._latencies.append(latency_s)
        # Immediate trip on a single extreme outlier (>3x threshold) so we
        # don't wait for 10 slow cycles to accumulate before reacting.
        if latency_s > self.threshold_s * 3:
            self.record_failure(
                f"latency spike {latency_s:.1f}s > {self.threshold_s*3:.1f}s (3x threshold)")
            return
        if len(self._latencies) < 10:
            return
        med = statistics.median(self._latencies)
        if med > self.threshold_s:
            self.record_failure(f"median latency {med:.1f}s > {self.threshold_s:.1f}s")
        else:
            self.record_success()


class ConsecutiveLossBreaker(CircuitBreaker):
    """Trips after N consecutive losing trades.

    Default: 5 consecutive losses trips for 15 minutes.
    This is a systemic backstop — the per-trade ConsecutiveLossGate in
    RiskPipeline rejects at 3; this breaker halts ALL trading at 5 because
    5 in a row means the strategy or regime is fundamentally broken.

    Distinct from ConsecutiveLossGate: the gate blocks new entries per-trade;
    the breaker blocks the entire cycle (including position management) and
    requires a cooldown or manual reset.
    """
    def __init__(self, threshold: int = 5,
                 cooldown_s: float = 900.0) -> None:
        # 15-min cooldown: 5 consecutive losses = something is deeply wrong.
        # Give the market time to change regime before probing again.
        super().__init__(name="consecutive_loss",
                         failure_threshold=1, cooldown_s=cooldown_s)
        self.threshold = int(threshold)
        self._streak: int = 0

    def record_trade_outcome(self, pnl: float) -> None:
        if pnl < 0:
            self._streak += 1
            if self._streak >= self.threshold:
                self.record_failure(
                    f"{self._streak} consecutive losses >= {self.threshold}")
                # Reset streak after tripping so the cooldown is meaningful
                self._streak = 0
        else:
            self._streak = 0
            self.record_success()


class BrokerDisconnectBreaker(CircuitBreaker):
    """Trips when the broker connection drops.

    Listens to MT5_DISCONNECT events on the EventBus. On disconnect,
    records a failure. On MT5_RECONNECT, records a success.

    Default: 1 disconnect trips the breaker for 30s (short — MT5 often
    has transient blips). After 30s, HALF_OPEN allows a probe cycle to
    test whether the connection has been re-established.
    """
    def __init__(self, cooldown_s: float = 30.0,
                 bus: Optional[EventBus] = None) -> None:
        super().__init__(name="broker_disconnect",
                         failure_threshold=1, cooldown_s=cooldown_s)
        self._bus = bus or get_bus()
        # Subscribe to disconnect/reconnect events
        try:
            self._bus.subscribe(EventType.MT5_DISCONNECT, self._on_disconnect)
            self._bus.subscribe(EventType.MT5_RECONNECT, self._on_reconnect)
        except Exception:
            pass  # bus subscription is best-effort

    def _on_disconnect(self, event) -> None:
        self.record_failure(f"MT5_DISCONNECT: {event.payload.get('error', '')}")

    def _on_reconnect(self, event) -> None:
        self.record_success()


class SlippageBreaker(CircuitBreaker):
    """Trips when actual fill price deviates from expected by > X bps.

    Default: 3 consecutive fills with > 20 bps slippage trips for 60s.
    Catches degraded execution quality (e.g., low liquidity, broker issues)
    that would silently erode edge if trading continued.

    The slippage is computed as |fill_price - expected_price| / expected_price
    * 10000 (bps). Expected_price is the bid/ask at order time.
    """
    def __init__(self, max_slippage_bps: float = 20.0,
                 consecutive_threshold: int = 3,
                 cooldown_s: float = 60.0) -> None:
        super().__init__(name="slippage",
                         failure_threshold=consecutive_threshold,
                         cooldown_s=cooldown_s,
                         critical=False)
        self.max_slippage_bps = float(max_slippage_bps)
        self._recent_slippages: Deque[float] = deque(maxlen=20)

    def record_fill(self, expected_price: float, actual_price: float) -> None:
        if expected_price <= 0:
            return
        slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
        self._recent_slippages.append(slippage_bps)
        if slippage_bps > self.max_slippage_bps:
            self.record_failure(
                f"slippage {slippage_bps:.1f}bps > {self.max_slippage_bps:.1f}bps")
        else:
            self.record_success()


# ----------------------------------------------------------------------
# Top-of-cycle breaker coordinator
# ----------------------------------------------------------------------
class CircuitBreakerCoordinator:
    """Runs all breakers at the top of TradingBot.cycle().

    If ANY breaker is OPEN, the cycle is skipped and a loud log is emitted.
    This is the systemic safety net — distinct from the per-trade RiskPipeline
    gates which run inside _process_symbol.

    Usage in TradingBot.cycle():
        if self.breakers.should_block_cycle():
            log.warning("TradingBot: cycle blocked by open breaker(s): %s",
                        self.breakers.open_breakers())
            return result
    """

    def __init__(self, config: dict, bus: Optional[EventBus] = None, ignore_broker_disconnect: bool = False):
        self._bus = bus or get_bus()
        cfg = config or {}
        breaker_cfg = cfg.get("circuit_breakers", {})
        self.breakers = [
            EquityDrawdownBreaker(
                threshold=float(breaker_cfg.get("drawdown_threshold", 0.10)),
                cooldown_s=float(breaker_cfg.get("drawdown_cooldown_s", 300.0)),
            ),
            ErrorRateBreaker(
                window=int(breaker_cfg.get("error_window", 50)),
                threshold=float(breaker_cfg.get("error_threshold", 0.5)),
                cooldown_s=float(breaker_cfg.get("error_cooldown_s", 60.0)),
            ),
            LatencyBreaker(
                threshold_s=float(breaker_cfg.get("latency_threshold_s", 60.0)),
                cooldown_s=float(breaker_cfg.get("latency_cooldown_s", 120.0)),
            ),
            ConsecutiveLossBreaker(
                threshold=int(breaker_cfg.get("consecutive_loss_threshold", 5)),
                cooldown_s=float(breaker_cfg.get("consecutive_loss_cooldown_s", 900.0)),
            ),
            BrokerDisconnectBreaker(
                cooldown_s=float(breaker_cfg.get("disconnect_cooldown_s", 30.0)),
                bus=self._bus,
            ) if not ignore_broker_disconnect else None,
            SlippageBreaker(
                max_slippage_bps=float(breaker_cfg.get("slippage_max_bps", 20.0)),
                consecutive_threshold=int(breaker_cfg.get("slippage_consecutive", 3)),
                cooldown_s=float(breaker_cfg.get("slippage_cooldown_s", 60.0)),
            ),
        ]
        self.breakers = [b for b in self.breakers if b is not None]
        log.info("CircuitBreakerCoordinator: %d breakers registered",
                 len(self.breakers))

    def should_block_cycle(self) -> bool:
        """Returns True if any CRITICAL breaker is OPEN.

        C18 fix: this used to trip on ANY open breaker, including
        non-critical ones like SlippageBreaker — meaning a slippage trip
        would block SL/TP position management along with new entries.
        Now only critical breakers (equity drawdown, broker disconnect,
        consecutive loss, error rate) block the whole cycle. Use
        `should_block_new_trades()` to also account for non-critical
        breakers when deciding whether to open new positions.
        """
        return any(b.is_open and b.critical for b in self.breakers)

    def should_block_new_trades(self) -> bool:
        """Returns True if ANY breaker (critical or not) is OPEN.

        New trade entries should still respect non-critical breakers
        (e.g. don't open new positions while slippage is elevated), even
        though existing position management is allowed to continue.
        """
        return any(b.is_open for b in self.breakers)

    def open_breakers(self) -> list[dict]:
        """List of currently-open breakers (for logging/status)."""
        return [b.to_dict() for b in self.breakers if b.is_open]

    def all_status(self) -> list[dict]:
        """All breakers' status (for monitoring/dashboard)."""
        return [b.to_dict() for b in self.breakers]

    def record_equity(self, equity: float) -> None:
        for b in self.breakers:
            if isinstance(b, EquityDrawdownBreaker):
                b.record_equity(equity)

    def record_cycle(self, ok: bool, latency_s: float) -> None:
        for b in self.breakers:
            if isinstance(b, ErrorRateBreaker):
                b.record_cycle(ok)
            elif isinstance(b, LatencyBreaker):
                b.record_latency(latency_s)

    def record_trade_outcome(self, pnl: float) -> None:
        for b in self.breakers:
            if isinstance(b, ConsecutiveLossBreaker):
                b.record_trade_outcome(pnl)

    def record_fill(self, expected_price: float, actual_price: float) -> None:
        for b in self.breakers:
            if isinstance(b, SlippageBreaker):
                b.record_fill(expected_price, actual_price)

    def reset_all(self) -> None:
        """Manual reset of all breakers — operator-initiated only."""
        for b in self.breakers:
            b.reset()
        log.warning("CircuitBreakerCoordinator: ALL breakers manually reset")